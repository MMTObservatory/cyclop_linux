"""
The measurement loop: wait for night, find Polaris in a full frame, track it in a small region of
interest, and reduce each block of `n_samples` centroids to a seeing value.

While tracking, a camera.Acquirer thread fills chunks of cutouts around the star and this loop
centroids each chunk at once (star.measure_cube), so acquisition and analysis run in parallel.
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from cyclop import seeing, star, sun
from cyclop.camera import Acquirer

log = logging.getLogger(__name__)

DAY = "system idle, daytime, no measurements"
SEARCHING = "searching for a valid star"
MEASURING = "measurements pending"


class Monitor:
    def __init__(self, cfg, camera_factory, writer=None, publisher=None, sleep=time.sleep,
                 clock=time.monotonic, now=lambda: datetime.now(timezone.utc)):
        self.cfg = cfg
        self.camera_factory = camera_factory
        self.camera = None
        self.acq = None               # Acquirer while tracking
        self.writer = writer
        self.publisher = publisher
        self.sleep = sleep
        self.clock = clock
        self.now = now
        self.state = None
        self.results = deque(maxlen=20000)   # every block, accepted or not (about two weeks)
        self.n_results = 0
        self._reset_samples()
        # read by the web interface from another thread
        self.lock = threading.Lock()
        self._frame = None            # (timestamp, image, region) of the last grabbed frame
        self.star = None              # (timestamp, Star in full-frame pixels) of the last valid centroid
        self.recent = deque(maxlen=cfg['measurement']['n_samples'])  # (t, x, y, fwhm, flux)
        self.cam_stats = deque(maxlen=11)   # (clock, camera.stats()) about once a second
        self.pos = None               # last star position, full-frame pixels
        self.last_valid = None
        self.last_search = None
        self.last_search_log = None

    # --- helpers -------------------------------------------------------------------------------

    def _reset_samples(self):
        # per sample; delivered/dropped are the camera's running frame counts
        self.samples = {k: [] for k in ('t', 'x', 'y', 'fwhm', 'flux', 'delivered', 'dropped')}
        self.proc_times = []          # (seconds, frames) of analysis for each chunk

    def _set_state(self, state):
        if state != self.state:
            log.info(f'Status changed from "{self.state or "unknown state"}" -> "{state}"')
            self.state = state

    def _is_night(self):
        s = self.cfg['site']
        return sun.is_night(s['latitude'], s['longitude'], s['max_sun_alt'], self.now())

    def _ensure_camera(self):
        if self.camera is None:
            self.camera = self.camera_factory()
        return self.camera

    @property
    def frame(self):
        acq = self.acq
        if acq is not None and acq.latest is not None:
            return acq.latest
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value

    def _drop_camera(self):
        self._stop_tracking()
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.camera = None

    def _start_tracking(self, x, y):
        """Centre the region of interest on (x, y) and start filling chunks around it."""
        self._stop_tracking()
        c, ms = self.cfg['camera'], self.cfg['measurement']
        w, h = c['roi_width'], c['roi_height']
        self.camera.set_region(int(x - w / 2), int(y - h / 2), w, h)
        self.acq = Acquirer(self.camera, (x, y), radius=self.cfg['star']['search'],
                            chunk_frames=ms['chunk_frames'], max_seconds=ms['chunk_seconds']).start()

    def _stop_tracking(self):
        if self.acq is not None:
            self.acq.stop()
            self._frame = self.acq.latest or self._frame
            self.acq = None

    def _valid(self, s):
        return s is not None and 0.5 < s.fwhm <= self.cfg['star']['max_fwhm']

    # --- states --------------------------------------------------------------------------------

    def search(self):
        """Look for Polaris in a full frame; switch to tracking once it is confirmed."""
        st = self.cfg['star']
        now = self.clock()
        if self.last_search is not None and now - self.last_search < self.cfg['measurement']['search_interval']:
            self.sleep(0.5)
            return
        self.last_search = now
        self._set_state(SEARCHING)

        cam = self._ensure_camera()
        w, h = cam.sensor_size
        if tuple(cam.region) != (0, 0, w, h):
            cam.set_region(0, 0, w, h)
        cam.start()

        found = []
        for _ in range(st['confirm_frames'] + 2):
            frame = cam.grab(timeout_s=2.0)
            if frame is None:
                continue
            self.frame = (frame[0], frame[1], tuple(cam.region))
            s = star.measure(frame[1], box=st['box'], min_snr=st['min_snr'])
            if not self._valid(s):
                found = []
                continue
            if found and np.hypot(s.x - found[-1].x, s.y - found[-1].y) > 10:
                found = []
            found.append(s)
            if len(found) >= st['confirm_frames']:
                break

        if len(found) < st['confirm_frames']:
            if self.last_search_log is None or now - self.last_search_log > 120:
                log.info("No valid star found !")
                self.last_search_log = now
            return

        s = found[-1]
        log.info(f"Valid Star Found (X={s.x:.0f} Y={s.y:.0f}, fwhm={s.fwhm:.2f}, peak={s.peak:.0f}, "
                 f"saturated={s.n_saturated}): switching to tracking region")
        if s.n_saturated >= 3:
            log.warning(f"{s.n_saturated} saturated pixels; reduce exposure or gain")
        self.pos = (s.x, s.y)
        self.star = (frame[0], s)
        self._start_tracking(s.x, s.y)
        self._reset_samples()
        self.last_valid = self.clock()
        self.last_search_log = None
        self._set_state(MEASURING)

    def track(self):
        """Centroid the next chunk of frames and add the valid ones to the current block."""
        acq = self.acq                # _analyze may replace it when it moves the region
        chunk = acq.get(timeout=5.0)
        if chunk is None or chunk.n == 0:
            if chunk is not None:
                acq.release(chunk)
            self._check_lost()
            return
        try:
            self._analyze(chunk)
        finally:
            acq.release(chunk)

    def _analyze(self, chunk):
        st, ms = self.cfg['star'], self.cfg['measurement']
        started = time.perf_counter()
        n = chunk.n
        cube, x0, y0 = chunk.cube[:n], chunk.x0[:n], chunk.y0[:n]
        r = star.measure_cube(cube, box=st['box'], min_snr=st['min_snr'])
        t = chunk.t[:n]
        ok = r['ok'] & (r['fwhm'] > 0.5) & (r['fwhm'] <= st['max_fwhm'])
        if st['centroid'] == 'xcorr':
            # positions from the matched filter, searched around where the star was last seen;
            # measure_cube still supplies flux, peak, FWHM and saturation, and vets the frame
            c = star.correlate_cube(cube, self.pos[0] - x0, self.pos[1] - y0,
                                    sigma=st['xcorr_sigma'], reach=st['xcorr_reach'])
            x, y = c['x'] + x0, c['y'] + y0
            ok &= c['snr'] >= st['xcorr_min_snr']
        else:
            x, y = r['x'] + x0, r['y'] + y0
        flux = r['flux'] / star.column_flux_factor(x, st['column_flux_terms'])
        good = np.flatnonzero(ok)
        self._update_cam_stats()
        if len(good) == 0:
            self._check_lost()
            return

        i = good[-1]
        self.pos = (float(x[i]), float(y[i]))
        self.acq.center = self.pos
        self.last_valid = self.clock()
        self.star = (float(t[i]), star.Star(
            x=self.pos[0], y=self.pos[1], flux=float(flux[i]), peak=float(r['peak'][i]),
            fwhm=float(r['fwhm'][i]), n_saturated=int(r['n_saturated'][i]), snr=float(r['snr'][i])))
        cols = {'t': t[good], 'x': x[good], 'y': y[good], 'fwhm': r['fwhm'][good], 'flux': flux[good],
                'delivered': chunk.delivered[:n][good], 'dropped': chunk.dropped[:n][good]}
        smp = self.samples
        for k, v in cols.items():
            smp[k].extend(v.tolist())
        with self.lock:
            self.recent.extend(zip(*(cols[k].tolist() for k in ('t', 'x', 'y', 'fwhm', 'flux'))))
        self.proc_times.append((time.perf_counter() - started, n))

        while len(smp['t']) >= ms['n_samples']:
            self.finish_block()
            smp = self.samples

        rx, ry, w, h = self.camera.region
        m = st['recenter_margin']
        lx, ly = self.pos[0] - rx, self.pos[1] - ry
        if lx < m or ly < m or lx > w - m or ly > h - m:
            log.info(f"Tracking change position -> star at X={self.pos[0]:.0f} Y={self.pos[1]:.0f}")
            self._start_tracking(*self.pos)

    def _check_lost(self):
        ms = self.cfg['measurement']
        if self.clock() - self.last_valid > ms['lost_timeout']:
            log.info(f"Star likely hidden by clouds (last valid measurement "
                     f"{self.clock() - self.last_valid:.0f} s ago), searching full frame")
            self._stop_tracking()
            self._reset_samples()
            self.last_search = None
            self._set_state(SEARCHING)

    def _update_cam_stats(self):
        now = self.clock()
        if not self.cam_stats or now - self.cam_stats[-1][0] >= 1.0:
            stats = self.camera.stats()
            with self.lock:
                self.cam_stats.append((now, stats))

    def _frame_stats(self, smp, span):
        """Dropped frames, camera frame rate and processing time over the block just finished."""
        secs, frames = (sum(v) for v in zip(*self.proc_times)) if self.proc_times else (0, 0)
        out = {'proc_ms': 1e3 * secs / frames if frames else None,
               'dropped': None, 'drop_fraction': None, 'camera_fps': None}
        # frames the camera produced between the block's first and last samples
        delivered = int(smp['delivered'][-1] - smp['delivered'][0])
        dropped = int(smp['dropped'][-1] - smp['dropped'][0])
        if delivered > 0 and dropped >= 0:
            out.update(dropped=dropped, drop_fraction=dropped / (delivered + dropped),
                       camera_fps=(delivered + dropped) / span if span > 0 else None)
        return out

    def finish_block(self):
        """Reduce the first n_samples samples to a seeing value; any beyond start the next block."""
        n = self.cfg['measurement']['n_samples']
        smp = {k: np.asarray(v[:n]) for k, v in self.samples.items()}
        rest = {k: v[n:] for k, v in self.samples.items()}
        span = smp['t'][-1] - smp['t'][0]
        frames = self._frame_stats(smp, span)
        proc = self.proc_times
        self._reset_samples()
        if rest['t']:                 # the chunk that finished this block also starts the next
            self.samples, self.proc_times = rest, proc[-1:]
        t_end = self.now()
        lat = self.cfg['site']['latitude']
        ms = self.cfg['measurement']
        rate = (len(smp['t']) - 1) / span
        keep = seeing.inliers(smp['t'], smp['x'], smp['y'], ms['detrend'], ms['clip'] or None)
        n_clipped = int((~keep).sum())
        smp = {k: v[keep] for k, v in smp.items()}
        r = seeing.compute(smp['t'], smp['x'], smp['y'], lat, detrend=ms['detrend'])
        flux = float(np.mean(smp['flux']))
        accepted = r['zenith'] <= self.cfg['measurement']['max_zenith_seeing']
        r.update(flux=flux, fwhm=float(np.mean(smp['fwhm'])), rate=rate, time=t_end, accepted=accepted,
                 n_clipped=n_clipped, **frames)
        with self.lock:
            self.results.append(r)
        self.n_results += 1

        if not accepted:
            log.info(f"Seeing above threshold ({r['zenith']:.2f} > "
                     f"{self.cfg['measurement']['max_zenith_seeing']} arcsec), value discarded")
            return
        log.info(f"Seeing Zen. Ok : {r['zenith']:.2f} arcsec (local {r['local']:.2f}, "
                 f"sigma {r['sigma_x']:.3f}/{r['sigma_y']:.3f} px, fwhm {r['fwhm']:.2f} px, "
                 f"flux {flux:.0f}, {rate:.1f} fps, {n_clipped} clipped{self._drop_text(r)})")
        if self.writer:
            self.writer.write(t_end, flux, r, samples=smp)
        if self.publisher:
            self.publisher.publish(t_end, flux, r)

    @staticmethod
    def _drop_text(r):
        if r['drop_fraction'] is None:
            return ""
        text = f", dropped {100 * r['drop_fraction']:.1f}% of {r['camera_fps']:.1f} fps"
        return text + (f", {r['proc_ms']:.2f} ms/frame" if r['proc_ms'] is not None else "")

    # --- main loop -----------------------------------------------------------------------------

    def step(self, ignore_sun=False):
        if not ignore_sun and not self._is_night():
            if self.state != DAY:
                if self.state is not None:
                    log.info("Stop measurements because of daytime")
                self._drop_camera()
                self._reset_samples()
                self.frame = None
                self._set_state(DAY)
            self.sleep(60)
            return
        if self.state == DAY:
            log.info("Night is coming, measurements restart, find star in full frame mode")
            self.last_search = None
        if self.state == MEASURING:
            self.track()
        else:
            self.search()

    def run(self, ignore_sun=False, max_results=None, should_stop=lambda: False):
        while not should_stop():
            if max_results is not None and self.n_results >= max_results:
                break
            try:
                self.step(ignore_sun=ignore_sun)
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Camera or processing error; reopening camera in 10 s")
                self._drop_camera()
                self._reset_samples()
                self.state = None
                self.last_search = None
                self.sleep(10)
        self._drop_camera()
