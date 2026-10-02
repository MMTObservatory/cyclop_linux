# cyclop

[![tests](https://github.com/MMTObservatory/cyclop_linux/actions/workflows/tests.yml/badge.svg)](https://github.com/MMTObservatory/cyclop_linux/actions/workflows/tests.yml)

Linux replacement for Alcor System's Windows-only `SeeingMonitor_Cyclop` software, written for the
MMTO MiniCyclop (no heater or RS232). It drives the camera directly over GigE Vision, measures
Polaris' image motion, and publishes seeing straight to redis.

## How it works

1. When the Sun is below `max_sun_alt` (-5°), it grabs full frames (2592×1944) until Polaris is
   detected in several consecutive frames.
2. It switches to a 640×480 region around the star and centroids every frame (132 fps), moving
   the region when the star drifts near an edge. If the star is lost for `lost_timeout` s
   (clouds), it goes back to full-frame searching. Acquisition and analysis run in parallel: a
   thread copies an 81×81 cutout around the star from each frame into chunks of up to 256 frames
   (`chunk_frames`, or `chunk_seconds`), and the main loop centroids a whole chunk at once with
   vectorized numpy (about 0.1 ms per frame) while the next one fills.

   Positions come from cross-correlating each frame with a Gaussian (`centroid = "xcorr"`, a
   matched filter of sigma `xcorr_sigma`) within `xcorr_reach` px of the star's last position,
   with the peak refined by a 3-point Gaussian fit along each axis. For a ~2 px star this has 2-4×
   less noise than a thresholded centre of mass (`centroid = "moments"`, the original method) and
   no pixel-phase bias: read noise in the wings, which the moments let into σ, is weighted down.
   On sky (2026-10-02) the moments gave seeing ~35% above the MMT wavefront sensors, while the
   cross-correlation agreed with them (2.27/2.36" vs WFS 2.27") and with photutils aperture
   centroids at r ≈ 2 FWHM. Frames are kept when both the star's peak S/N (`min_snr`) and the
   correlation S/N (`xcorr_min_snr`) pass.
3. Every `n_samples` (3000) centroids, it drops samples more than `clip` (5) robust sigmas from
   the drift-removed median (a frame that centroided on noise would otherwise dominate σ), removes
   the slow drift of Polaris (linear fit in time) and computes:

   ```
   local seeing ["] = 13.58031732652 × (σx^1.2 + σy^1.2) / 2      σ in pixels
   zenith seeing    = local × cos(90° − latitude)^0.6
   r0 [mm]          = 550 nm / zenith seeing
   ```

   These relations were recovered from the Windows software's own `Data.bin` history (300,000
   measurements) and reproduce its reported values to ~1e-10 given the same σ.

4. Results go to:
   - `Seeing_Data.txt` / `Last_Seeing_Data.txt` in `output.data_dir`, in the same format as the
     Windows software;
   - redis keys `seeing_monitor_{seeing,flux,r0,measurement_timestamp}` (set + publish), as
     `minicyclop`'s `tcs_logger` did.

## Install

```
conda create -n cyclop -c conda-forge python=3.14 numpy redis-py pytest
pip install -e .
```

Camera access needs Aravis with GObject introspection, built into the same environment
(conda-forge has no Aravis package); see `docs/aravis.md`.

## Use

```
cp config.example.toml config.toml
cyclop -c config.toml camera-info                 # check the camera is reachable
cyclop -c config.toml run --simulate --ignore-sun --no-redis -n 2    # dry run with a fake star
cyclop -c config.toml run                         # the real thing
cyclop -c config.toml run --gain 10 --exposure 2000   # try other camera settings (dB, us)
cyclop replay ~/path/to/*_Motion.txt              # reduce Windows motion files with this code
```

`systemd/cyclop.service` runs it as a user service.

## Frame timing

Each frame is timestamped with the host clock when it arrived (Aravis' buffer system timestamp),
so frames that wait in the queue while the analysis catches up keep their true times. Frames the
camera produced but the analysis never saw (queue full, incomplete) are counted from gaps in the
GigE Vision frame ids. Every block's log line reports the processed rate, the camera rate, the
percentage dropped and the mean analysis time per frame, which must stay under the camera's frame
period (about 7.5 ms at the 640×480 region's maximum of 132 fps) to use every frame.

## Web interface

`cyclop run` also serves read-only status pages at `http://<host>:8080/` (`[web]` in the config,
`--web-port`, or `--no-web`). They loosely follow the Windows GUI:

- **Capture**: the live camera frame with the star marked, its histogram, saturated-pixel count,
  brightest pixel, a zoom on the star, the camera settings, and the processed versus camera frame
  rate with the fraction of frames dropped;
- **Status**: UTC/local/sidereal time, Sun elevation, and the last local and zenith seeing and r0;
- **Plots**: the Windows "Output results" tabs: zenith seeing and flux versus time (1h to 1M,
  history from `<data_dir>/Seeing_Data.txt` plus any `web.history_files`, such as the Windows
  software's log), and the star motion and FWHM of the latest centroids;
- a running log at the bottom.

Tabs can be linked to directly, e.g. `#status` or `#plots/motion`. The page is self-contained (no
external scripts), so it works without internet access. Everything is controlled from the CLI and
config file; the web interface never changes anything.

## Notes

- Only one program can control the camera at a time: stop `SeeingMonitor_Cyclop.exe` before
  running `cyclop` against the real camera.
- The camera is at 192.168.2.59 on the dedicated 192.168.2.0/24 interface. Under WSL2 this works
  with `networkingMode=mirrored` in `.wslconfig`.
- Jumbo frames are not required for the 640×480 tracking region but help full-frame searches.
