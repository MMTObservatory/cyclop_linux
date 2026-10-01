"""
The measurement loop: wait for night, find Polaris in a full frame, track it in a small region of
interest, and reduce each block of `n_samples` centroids to a seeing value.
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np

from cyclop import seeing, star, sun

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
        self.writer = writer
        self.publisher = publisher
        self.sleep = sleep
        self.clock = clock
        self.now = now
        self.state = None
        self.results = []
        self._reset_samples()
        self.pos = None               # last star position, full-frame pixels
        self.last_valid = None
        self.last_search = None
        self.last_search_log = None

    # --- helpers -------------------------------------------------------------------------------

    def _reset_samples(self):
        self.samples = {'t': [], 'x': [], 'y': [], 'fwhm': [], 'flux': []}

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

    def _drop_camera(self):
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.camera = None

    def _set_roi(self, x, y):
        c = self.cfg['camera']
        w, h = c['roi_width'], c['roi_height']
        self.camera.set_region(int(x - w / 2), int(y - h / 2), w, h)
        self.camera.start()

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
        self._set_roi(s.x, s.y)
        self._reset_samples()
        self.last_valid = self.clock()
        self.last_search_log = None
        self._set_state(MEASURING)

    def track(self):
        """Grab one ROI frame and add its centroid to the current block."""
        st = self.cfg['star']
        ms = self.cfg['measurement']
        cam = self.camera
        frame = cam.grab(timeout_s=1.0)
        s = None
        if frame is not None:
            t, img = frame
            x0, y0, w, h = cam.region
            guess = (self.pos[0] - x0, self.pos[1] - y0)
            s = star.measure(img, box=st['box'], min_snr=st['min_snr'], guess=guess, search=st['search'])

        if not self._valid(s):
            if self.clock() - self.last_valid > ms['lost_timeout']:
                log.info(f"Star likely hidden by clouds (last valid measurement "
                         f"{self.clock() - self.last_valid:.0f} s ago), searching full frame")
                self._reset_samples()
                self.last_search = None
                self._set_state(SEARCHING)
            return

        x, y = s.x + x0, s.y + y0
        self.pos = (x, y)
        self.last_valid = self.clock()
        smp = self.samples
        smp['t'].append(t)
        smp['x'].append(x)
        smp['y'].append(y)
        smp['fwhm'].append(s.fwhm)
        smp['flux'].append(s.flux)

        m = st['recenter_margin']
        if s.x < m or s.y < m or s.x > w - m or s.y > h - m:
            log.info(f"Tracking change position -> star at X={x:.0f} Y={y:.0f}")
            self._set_roi(x, y)

        if len(smp['t']) >= ms['n_samples']:
            self.finish_block()

    def finish_block(self):
        smp = {k: np.asarray(v) for k, v in self.samples.items()}
        self._reset_samples()
        t_end = self.now()
        lat = self.cfg['site']['latitude']
        r = seeing.compute(smp['t'], smp['x'], smp['y'], lat, detrend=self.cfg['measurement']['detrend'])
        flux = float(np.mean(smp['flux']))
        rate = (len(smp['t']) - 1) / (smp['t'][-1] - smp['t'][0])
        r.update(flux=flux, fwhm=float(np.mean(smp['fwhm'])), rate=rate, time=t_end)
        self.results.append(r)

        if r['zenith'] > self.cfg['measurement']['max_zenith_seeing']:
            log.info(f"Seeing above threshold ({r['zenith']:.2f} > "
                     f"{self.cfg['measurement']['max_zenith_seeing']} arcsec), value discarded")
            return
        log.info(f"Seeing Zen. Ok : {r['zenith']:.2f} arcsec (local {r['local']:.2f}, "
                 f"sigma {r['sigma_x']:.3f}/{r['sigma_y']:.3f} px, fwhm {r['fwhm']:.2f} px, "
                 f"flux {flux:.0f}, {rate:.1f} fps)")
        if self.writer:
            self.writer.write(t_end, flux, r, samples=smp)
        if self.publisher:
            self.publisher.publish(t_end, flux, r)

    # --- main loop -----------------------------------------------------------------------------

    def step(self, ignore_sun=False):
        if not ignore_sun and not self._is_night():
            if self.state != DAY:
                if self.state is not None:
                    log.info("Stop measurements because of daytime")
                self._drop_camera()
                self._reset_samples()
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
            if max_results is not None and len(self.results) >= max_results:
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
