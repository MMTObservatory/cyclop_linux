"""
Camera access. `AravisCamera` drives the GigE Vision DMK 33GP031 through Aravis;
`SimCamera` produces synthetic Polaris frames so the pipeline can be exercised without hardware.

Both expose the same small interface:

    sensor_size -> (width, height)
    set_region(x, y, width, height)   # stops acquisition if running
    start() / stop()
    grab(timeout_s) -> (timestamp_s, 2D uint8 ndarray) or None
    region -> (x, y, width, height)
    stats() -> {'delivered': n, 'dropped': n, ...} counted since the camera was opened

Timestamps are when the frame arrived, not when it was taken off the queue, so a backlog in
processing does not distort them; frames the camera produced but we never got count as dropped.
"""

import logging
import math
import queue
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """The camera stopped delivering frames or no longer accepts our commands; reopen it."""


class AravisCamera:
    # Aravis stream counters reported by stats() (reset whenever a stream is created)
    STREAM_COUNTERS = ('n_underruns', 'n_failures', 'n_missing_frames', 'n_missing_packets',
                       'n_resent_packets')

    def __init__(self, address=None, exposure_us=3906.0, gain=12.5, frame_rate=None, n_buffers=128,
                 buffer_mb=64.0, socket_buffer_mb=8.0):
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
        self.buffer_mb = buffer_mb
        # UDP receive buffer for the GigE stream. Aravis's automatic size is about one frame, which
        # a host pause of a few tens of ms overflows (lost packets, resends, incomplete frames);
        # the kernel caps it at net.core.rmem_max
        self.socket_buffer_mb = socket_buffer_mb
        self.frame_rate = frame_rate
        self.stream = None
        self.running = False
        self.delivered = 0
        self.dropped = 0
        self._last_id = None
        self._stream_totals = dict.fromkeys(self.STREAM_COUNTERS, 0)
        # set from Aravis's heartbeat thread when the camera stops accepting us as its controller
        self.control_lost = False
        self.cam.get_device().connect('control-lost', self._on_control_lost)

        self.sensor_size = tuple(self.cam.get_sensor_size())
        self._x_inc = self._increment('OffsetX', 4)
        self._y_inc = self._increment('OffsetY', 4)
        self._w_inc = self._increment('Width', 16)
        self._h_inc = self._increment('Height', 4)

        self.set_exposure(exposure_us)
        self.set_gain(gain)
        self.region = tuple(self.cam.get_region())

    def _on_control_lost(self, device):
        self.control_lost = True
        log.warning("Camera control lost (heartbeat failed); the camera no longer accepts our commands")

    def _increment(self, feature, default):
        try:
            return max(1, int(self.cam.get_integer_increment(feature)))
        except Exception:
            return default

    def set_exposure(self, exposure_us):
        self.cam.set_exposure_time(float(exposure_us))
        self.exposure_us = self.cam.get_exposure_time()      # what the camera actually took
        log.info(f"Exposure set to {self.exposure_us:.1f} us")

    def set_gain(self, gain):
        self.cam.set_gain(float(gain))
        self.gain = self.cam.get_gain()
        log.info(f"Gain set to {self.gain:.2f} dB")

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
        if self.socket_buffer_mb and isinstance(self.stream, Aravis.GvStream):
            self.stream.set_property('socket-buffer', Aravis.GvStreamSocketBuffer.FIXED)
            self.stream.set_property('socket-buffer-size', int(self.socket_buffer_mb * 2 ** 20))
        payload = self.cam.get_payload()
        # cap the memory: 128 full frames would be 645 MB, which the heap keeps after the stream stops
        n = max(4, min(self.n_buffers, int(self.buffer_mb * 2 ** 20 // payload)))
        for _ in range(n):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.cam.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        self._last_id = None
        self.cam.start_acquisition()
        self.running = True

    def stop(self):
        if not self.running:
            return
        self.cam.stop_acquisition()
        for k, v in self._stream_counters().items():
            self._stream_totals[k] += v
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
                self._count(buf.get_frame_id())
                w, h = buf.get_image_width(), buf.get_image_height()
                img = np.frombuffer(buf.get_data(), dtype=np.uint8)[:w * h].reshape(h, w).copy()
                # host clock when the frame arrived; falls back to now if Aravis did not set it
                ns = buf.get_system_timestamp()
                return (ns / 1e9 if ns else time.time()), img
            finally:
                self.stream.push_buffer(buf)
        return None

    def _count(self, frame_id):
        """Count the frames missing between this one and the last we delivered."""
        if self._last_id is not None:
            gap = frame_id - self._last_id - 1
            if self._last_id > 0xFFFF - 10000 and frame_id < 10000 and self._last_id <= 0xFFFF:
                gap %= 0xFFFF          # 16-bit GigE Vision block ids wrap 65535 -> 1
            if 0 <= gap < 10000:       # anything else is a restart, not lost frames
                self.dropped += gap
        self._last_id = frame_id
        self.delivered += 1

    def _stream_counters(self):
        if self.stream is None:
            return dict.fromkeys(self.STREAM_COUNTERS, 0)
        return {k: int(self.stream.get_info_uint64_by_name(k)) for k in self.STREAM_COUNTERS}

    def stats(self):
        out = {'delivered': self.delivered, 'dropped': self.dropped}
        out.update({k: self._stream_totals[k] + v for k, v in self._stream_counters().items()})
        return out

    def close(self):
        self.stop()
        self.cam = None


class SimCamera:
    """
    Synthetic Polaris: a Gaussian star drifting slowly around the pole with Gaussian
    tip-tilt jitter of known rms, plus background and read noise.
    """

    NOISE_PAD = 64

    def __init__(self, sensor_size=(2592, 1944), star_xy=(1796.0, 679.0), fwhm=2.2, peak=150.0,
                 jitter_px=0.15, drift_px_per_s=0.15, background=12.0, noise=2.0, frame_rate=60.0,
                 realtime=False, drop_fraction=0.0, seed=None):
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
        self.drop_fraction = drop_fraction    # chance that each frame is lost before delivery
        self.delivered = 0
        self.dropped = 0
        self.rng = np.random.default_rng(seed)
        self._noise = None
        self.region = (0, 0, *sensor_size)
        self.running = False
        self.t = 0.0
        self.visible = True
        self.stalled = False                  # True: deliver nothing, like a camera that went silent

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

    def _noise_frame(self, h, w):
        """Unit Gaussian noise: a random window of a precomputed field (drawing fresh noise for
        every frame took longer than the analysis we want to exercise)."""
        if self._noise is None:
            sw, sh = self.sensor_size
            self._noise = self.rng.standard_normal((sh + self.NOISE_PAD, sw + self.NOISE_PAD),
                                                   dtype=np.float32)
        dy, dx = self.rng.integers(0, self.NOISE_PAD, 2)
        return self._noise[dy:dy + h, dx:dx + w]

    def true_position(self):
        return self.star_xy + self.drift * self.t

    def grab(self, timeout_s=1.0):
        if self.stalled:
            time.sleep(min(timeout_s, 0.01))
            return None
        if self.realtime:
            time.sleep(1.0 / self.frame_rate)
        self.t += 1.0 / self.frame_rate
        while self.drop_fraction and self.rng.random() < self.drop_fraction:
            self.dropped += 1
            self.t += 1.0 / self.frame_rate
        self.delivered += 1
        x0, y0, w, h = self.region
        img = self.background + self.noise * self._noise_frame(h, w)
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
        stamp = time.time() if self.realtime else self.t   # wall-clock timestamps when running live
        return stamp, np.clip(img, 0, 255).astype(np.uint8)

    def stats(self):
        return {'delivered': self.delivered, 'dropped': self.dropped}

    def close(self):
        pass


class Chunk:
    """A block of frames from `Acquirer`: cutouts around the star and where they came from."""

    def __init__(self, n, size):
        self.cube = np.zeros((n, size, size), dtype=np.uint8)
        self.t = np.zeros(n)
        self.x0 = np.zeros(n, dtype=int)     # full-frame position of each cutout's corner
        self.y0 = np.zeros(n, dtype=int)
        self.delivered = np.zeros(n, dtype=np.int64)   # camera's running counts after each frame
        self.dropped = np.zeros(n, dtype=np.int64)
        self.n = 0                            # frames filled


class Acquirer:
    """
    Grabs frames on a thread of its own and fills chunks of cutouts around `center`, so the
    analysis of one chunk overlaps the acquisition of the next.

    Frames are only copied into a cutout (2 * radius + 1 square) here; the analysis gets each
    chunk from `get()` and hands it back with `release()`. If the analysis falls behind and all
    `n_chunks` are in use, grabbing pauses, frames pile up in the camera's own buffers and any
    it cannot hold are counted as dropped by the camera. A chunk is handed over when full or
    `max_seconds` after its first frame, so a slow camera still delivers chunks regularly.
    """

    def __init__(self, camera, center, radius=40, chunk_frames=256, n_chunks=3, max_seconds=2.0):
        self.camera = camera
        self.center = center                  # full-frame (x, y); updated by the analysis
        self.radius = radius
        self.max_seconds = max_seconds
        size = 2 * radius + 1
        self._free = queue.Queue()
        for _ in range(n_chunks):
            self._free.put(Chunk(chunk_frames, size))
        self._full = queue.Queue()
        self.latest = None                    # (t, image, region) of the newest frame
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="acquire", daemon=True)

    def start(self):
        self.camera.start()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def get(self, timeout=None):
        """Next filled chunk, or None after `timeout` s (re-raises an acquisition error)."""
        try:
            chunk = self._full.get(timeout=timeout)
        except queue.Empty:
            chunk = None
        if self.error is not None:
            raise RuntimeError("acquisition failed") from self.error
        return chunk

    def release(self, chunk):
        chunk.n = 0
        self._free.put(chunk)

    def _origin(self, region):
        """Corner of the cutout around `center`, kept inside the camera region."""
        rx, ry, rw, rh = region
        size = 2 * self.radius + 1
        x0 = int(round(self.center[0])) - self.radius
        y0 = int(round(self.center[1])) - self.radius
        return (min(max(x0, rx), rx + rw - size), min(max(y0, ry), ry + rh - size))

    def _run(self):
        try:
            chunk, started = None, 0.0
            while not self._stop.is_set():
                if chunk is None:
                    try:
                        chunk = self._free.get(timeout=0.2)
                    except queue.Empty:
                        continue
                frame = self.camera.grab(timeout_s=0.5)
                now = time.monotonic()
                if frame is not None:
                    t, img = frame
                    region = tuple(self.camera.region)
                    self.latest = (t, img, region)
                    x0, y0 = self._origin(region)
                    i = chunk.n
                    lx, ly = x0 - region[0], y0 - region[1]
                    chunk.cube[i] = img[ly:ly + chunk.cube.shape[1], lx:lx + chunk.cube.shape[2]]
                    chunk.t[i], chunk.x0[i], chunk.y0[i] = t, x0, y0
                    chunk.delivered[i] = getattr(self.camera, 'delivered', 0)
                    chunk.dropped[i] = getattr(self.camera, 'dropped', 0)
                    if i == 0:
                        started = now
                    chunk.n = i + 1
                if chunk.n and (chunk.n == len(chunk.t) or now - started >= self.max_seconds):
                    self._full.put(chunk)
                    chunk = None
                elif frame is None and chunk.n == 0:
                    # nothing arriving: hand over an empty chunk so the analysis notices
                    self._full.put(chunk)
                    chunk = None
        except Exception as e:
            log.exception("Acquisition thread failed")
            self.error = e
            self._full.put(None)
