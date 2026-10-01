# cyclop

[![tests](https://github.com/MMTObservatory/cyclop_linux/actions/workflows/tests.yml/badge.svg)](https://github.com/MMTObservatory/cyclop_linux/actions/workflows/tests.yml)

Linux replacement for Alcor System's Windows-only `SeeingMonitor_Cyclop` software, written for the
MMTO MiniCyclop (no heater or RS232). It drives the camera directly over GigE Vision, measures
Polaris' image motion, and publishes seeing straight to redis.

## How it works

1. When the Sun is below `max_sun_alt` (-5°), it grabs full frames (2592×1944) until Polaris is
   detected in several consecutive frames.
2. It switches to a 640×480 region around the star and centroids every frame (~60 fps), moving
   the region when the star drifts near an edge. If the star is lost for `lost_timeout` s
   (clouds), it goes back to full-frame searching.
3. Every `n_samples` (3000) centroids, it removes the slow drift of Polaris (linear fit in time)
   and computes:

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
cyclop replay ~/path/to/*_motion.txt              # reduce Windows motion files with this code
```

`systemd/cyclop.service` runs it as a user service.

## Notes

- Only one program can control the camera at a time: stop `SeeingMonitor_Cyclop.exe` before
  running `cyclop` against the real camera.
- The camera is at 192.168.2.59 on the dedicated 192.168.2.0/24 interface. Under WSL2 this works
  with `networkingMode=mirrored` in `.wslconfig`.
- Jumbo frames are not required for the 640×480 tracking region but help full-frame searches.
