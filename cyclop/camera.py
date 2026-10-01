"""
Camera access. `AravisCamera` drives the GigE Vision DMK 33GP031 through Aravis;
`SimCamera` produces synthetic Polaris frames so the pipeline can be exercised without hardware.

Both expose the same small interface:

    sensor_size -> (width, height)
    set_region(x, y, width, height)   # stops acquisition if running
    start() / stop()
    grab(timeout_s) -> (timestamp_s, 2D uint8 ndarray) or None
    region -> (x, y, width, height)
"""

import logging
import math
import time

import numpy as np

log = logging.getLogger(__name__)


class AravisCamera:
    def __init__(self, address=None, exposure_us=3906.0, gain=12.43, frame_rate=None, n_buffers=32):
        import gi
        gi.require_version('Aravis', '0.8')
        from gi.repository import Aravis
        self._arv = Aravis

        self.cam = Aravis.Camera.new(address)
        log.info(f"Opened {self.cam.get_vendor_name()} {self.cam.get_model_name()} "
                 f"s/n {self.cam.get_device_serial_number()} at {address or 'first found'}")
        self.cam.set_pixel_format(Aravis.PIXEL_FORMAT_MONO_8)
        if self.cam.is_gv_device():
            self.cam.gv_auto_packet_size()
        self.n_buffers = n_buffers
        self.frame_rate = frame_rate
        self.stream = None
        self.running = False

        self.sensor_size = tuple(self.cam.get_sensor_size())
        self._x_inc = self._increment('OffsetX', 4)
        self._y_inc = self._increment('OffsetY', 4)
        self._w_inc = self._increment('Width', 16)
        self._h_inc = self._increment('Height', 4)

        self.set_exposure(exposure_us)
        self.set_gain(gain)
        self.region = tuple(self.cam.get_region())

    def _increment(self, feature, default):
        try:
            return max(1, int(self.cam.get_integer_increment(feature)))
        except Exception:
            return default

    def set_exposure(self, exposure_us):
        self.cam.set_exposure_time(float(exposure_us))
        log.info(f"Exposure set to {self.cam.get_exposure_time():.1f} us")

    def set_gain(self, gain):
        self.cam.set_gain(float(gain))
        log.info(f"Gain set to {self.cam.get_gain()}")

    def set_region(self, x, y, width, height):
        was_running = self.running
        self.stop()
        sw, sh = self.sensor_size
        width = min(sw, width - width % self._w_inc)
        height = min(sh, height - height % self._h_inc)
        x = int(np.clip(x, 0, sw - width)) // self._x_inc * self._x_inc
        y = int(np.clip(y, 0, sh - height)) // self._y_inc * self._y_inc
        # shrink first so the new offset is always legal, then move
        self.cam.set_region(0, 0, width, height)
        self.cam.set_region(x, y, width, height)
        self.region = tuple(self.cam.get_region())
        if self.frame_rate:
            try:
                lo, hi = self.cam.get_frame_rate_bounds()
                self.cam.set_frame_rate(min(max(self.frame_rate, lo), hi))
            except Exception as e:
                log.warning(f"Could not set frame rate: {e}")
        log.info(f"Region set to {self.region}, frame rate {self.cam.get_frame_rate():.1f} fps")
        if was_running:
            self.start()

    def start(self):
        if self.running:
            return
        Aravis = self._arv
        self.stream = self.cam.create_stream(None, None)
        payload = self.cam.get_payload()
        for _ in range(self.n_buffers):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.cam.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        self.cam.start_acquisition()
        self.running = True

    def stop(self):
        if not self.running:
            return
        self.cam.stop_acquisition()
        self.stream = None
        self.running = False

    def grab(self, timeout_s=1.0):
        """Next complete frame, skipping incomplete ones (e.g. missing packets) until the timeout."""
        deadline = time.monotonic() + timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            buf = self.stream.timeout_pop_buffer(int(remaining * 1e6))
            if buf is None:
                return None
            try:
                if buf.get_status() != self._arv.BufferStatus.SUCCESS:
                    log.debug(f"Buffer status {buf.get_status()}")
                    continue
                w, h = buf.get_image_width(), buf.get_image_height()
                img = np.frombuffer(buf.get_data(), dtype=np.uint8)[:w * h].reshape(h, w).copy()
                return time.time(), img
            finally:
                self.stream.push_buffer(buf)
        return None

    def close(self):
        self.stop()
        self.cam = None


class SimCamera:
    """
    Synthetic Polaris: a Gaussian star drifting slowly around the pole with Gaussian
    tip-tilt jitter of known rms, plus background and read noise.
    """

    def __init__(self, sensor_size=(2592, 1944), star_xy=(1796.0, 679.0), fwhm=2.2, peak=150.0,
                 jitter_px=0.15, drift_px_per_s=0.15, background=12.0, noise=2.0, frame_rate=60.0,
                 realtime=False, seed=None):
        self.sensor_size = sensor_size
        self.star_xy = np.array(star_xy, dtype=float)
        self.sigma = fwhm / 2.3548
        self.peak = peak
        self.jitter_px = jitter_px
        self.drift = np.array([drift_px_per_s, -drift_px_per_s * 0.5])
        self.background = background
        self.noise = noise
        self.frame_rate = frame_rate
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.region = (0, 0, *sensor_size)
        self.running = False
        self.t = 0.0
        self.visible = True

    def set_region(self, x, y, width, height):
        sw, sh = self.sensor_size
        width, height = min(width, sw), min(height, sh)
        x = int(np.clip(x, 0, sw - width)) // 4 * 4
        y = int(np.clip(y, 0, sh - height)) // 4 * 4
        self.region = (x, y, width, height)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def true_position(self):
        return self.star_xy + self.drift * self.t

    def grab(self, timeout_s=1.0):
        if self.realtime:
            time.sleep(1.0 / self.frame_rate)
        self.t += 1.0 / self.frame_rate
        x0, y0, w, h = self.region
        img = self.background + self.noise * self.rng.standard_normal((h, w))
        if self.visible:
            sx, sy = self.true_position() + self.rng.normal(0, self.jitter_px, 2)
            lx, ly = sx - x0, sy - y0
            r = int(math.ceil(5 * self.sigma))
            xa, xb = max(int(lx) - r, 0), min(int(lx) + r + 1, w)
            ya, yb = max(int(ly) - r, 0), min(int(ly) + r + 1, h)
            if xa < xb and ya < yb:
                yy, xx = np.mgrid[ya:yb, xa:xb]
                img[ya:yb, xa:xb] += self.peak * np.exp(
                    -((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * self.sigma ** 2))
        return self.t, np.clip(img, 0, 255).astype(np.uint8)

    def close(self):
        pass
