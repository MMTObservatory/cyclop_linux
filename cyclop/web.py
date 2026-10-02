"""
Read-only web interface: a small HTTP server running in a thread next to the measurement loop.
It serves one page (cyclop/static/index.html) and JSON/PNG endpoints that read the Monitor's
live state:

    /api/status          state, Sun, camera, star, frame statistics, last result
    /api/frame.png       last camera frame, contrast-stretched (?stretch=sqrt|linear|raw)
    /api/zoom.png        unscaled cutout around the star (?r=half-width)
    /api/history         seeing results (?days=N): Seeing_Data.txt file(s) plus this session's results
    /api/motion          the last n_samples centroids (t, x, y, fwhm)
    /api/log             recent log records (?after=id)

Controls stay in the CLI; nothing here changes the monitor.
"""

import json
import logging
import struct
import threading
import time
import zlib
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from cyclop import seeing, sun

log = logging.getLogger(__name__)


class LogBuffer(logging.Handler):
    """Keeps the last `size` log records for the web log pane."""

    def __init__(self, size=1000):
        super().__init__()
        self.records = deque(maxlen=size)
        self.next_id = 0
        self._lock = threading.Lock()

    def emit(self, record):
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += "\n" + logging.Formatter().formatException(record.exc_info)
            with self._lock:
                self.records.append({'id': self.next_id, 't': record.created,
                                     'level': record.levelname, 'name': record.name, 'msg': msg})
                self.next_id += 1
        except Exception:
            self.handleError(record)

    def since(self, after=-1):
        with self._lock:
            return [r for r in self.records if r['id'] > after]


def png_gray(img, level=1):
    """Encode a 2D uint8 array as a grayscale PNG."""
    h, w = img.shape
    raw = np.empty((h, w + 1), dtype=np.uint8)
    raw[:, 0] = 0                      # filter type "none" for every row
    raw[:, 1:] = img

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
            + chunk(b"IEND", b""))


def bin_max(img, factor):
    """Max-pool by `factor` so faint stars stay visible in a reduced full frame."""
    if factor <= 1:
        return img
    h, w = img.shape
    h, w = h - h % factor, w - w % factor
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def stretch(img, mode="sqrt"):
    if mode == "raw":
        return img
    hi = float(img.max())
    lo = min(float(np.median(img[::4, ::4])), hi - 8.0)   # flat (e.g. saturated) frames stay bright
    v = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    if mode == "sqrt":
        v = np.sqrt(v)
    return (v * 255.0 + 0.5).astype(np.uint8)


def read_history(path, days=31, now=None):
    """(unix time, zenith seeing, flux, r0) rows of Seeing_Data.txt from the last `days` days."""
    path = Path(path)
    if not path.exists():
        return []
    now = now or time.time()
    t_min = now - days * 86400.0
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        tail = days * 1500 * 100       # ~1 result/min, <100 bytes/line
        f.seek(max(0, size - tail))
        text = f.read().decode(errors="replace")
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        try:
            jd, flux, zen, r0 = (float(p) for p in parts[2:6])
        except ValueError:
            continue
        t = (jd - 2440587.5) * 86400.0
        if t >= t_min:
            rows.append((t, zen, flux, r0))
    return rows


def _num(v, nd=None):
    if v is None:
        return None
    v = float(v)
    if not np.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


class WebState:
    """Builds the JSON/PNG responses from a Monitor; caches per-frame work."""

    def __init__(self, monitor, logbuf, history_files=(), history_days=31):
        self.mon = monitor
        self.logbuf = logbuf
        self.started = time.time()
        if isinstance(history_files, (str, Path)):
            history_files = [history_files]
        rows = set()
        for f in history_files:
            f = Path(f).expanduser()
            new = read_history(f, history_days, now=self.started)
            if f.exists():
                log.info(f"Web interface loaded {len(new)} results from {f}")
            else:
                log.info(f"Web interface: no history file {f}")
            rows.update(new)         # identical lines in several files count once
        self.history = sorted(r for r in rows if r[0] <= self.started)
        self._cache = {}
        self._cache_lock = threading.Lock()

    def _cached(self, key, frame, fn):
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] is frame:
                return hit[1]
        val = fn()
        with self._cache_lock:
            self._cache[key] = (frame, val)
        return val

    # --- JSON ----------------------------------------------------------------------------------

    def frame_stats(self, frame):
        t, img, region = frame
        sat = self.mon.cfg['star'].get('saturation', 255)

        def compute():
            iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
            return {
                't': t, 'region': list(region), 'width': img.shape[1], 'height': img.shape[0],
                'max': int(img[iy, ix]), 'max_xy': [int(ix) + region[0], int(iy) + region[1]],
                'n_saturated': int(np.count_nonzero(img >= sat)),
                'median': float(np.median(img[::4, ::4])),
                'histogram': np.bincount(img.ravel(), minlength=256).tolist(),
            }
        return self._cached('stats', img, compute)

    def status(self):
        mon, cfg = self.mon, self.mon.cfg
        site, cam_cfg, ms = cfg['site'], cfg['camera'], cfg['measurement']
        now = datetime.now(timezone.utc)
        with mon.lock:
            recent = list(mon.recent)[-300:]
            last = mon.results[-1] if mon.results else None
            last_ok = next((r for r in reversed(mon.results) if r.get('accepted', True)), None)
            cam_stats = list(mon.cam_stats)
        fps = None
        if len(recent) > 10:
            ts = [r[0] for r in recent]
            if now.timestamp() - ts[-1] < 5 and ts[-1] > ts[0]:
                fps = (len(ts) - 1) / (ts[-1] - ts[0])

        out = {
            'now': now.timestamp(),
            'started': self.started,
            'state': mon.state,
            'site': {'name': site.get('name', ''), 'timezone': site['timezone'],
                     'latitude': site['latitude'], 'longitude': site['longitude'],
                     'max_sun_alt': site['max_sun_alt']},
            'sun_alt': sun.sun_altitude(now, site['latitude'], site['longitude']),
            'lst_hours': sun.local_sidereal_time(now, site['longitude']),
            'camera': {
                'simulated': type(mon.camera).__name__ == 'SimCamera' if mon.camera else None,
                'open': mon.camera is not None,
                'exposure_us': cam_cfg['exposure_us'], 'gain': cam_cfg['gain'],
                'frame_rate': cam_cfg['frame_rate'],
                'sensor': list(mon.camera.sensor_size) if mon.camera else None,
                'region': list(mon.camera.region) if mon.camera else None,
            },
            'fps': fps,
            'frames': self.frame_counts(cam_stats, mon.clock()),
            'block': {'n': len(mon.samples['t']), 'target': ms['n_samples']},
            'n_results': mon.n_results,
            'max_zenith_seeing': ms['max_zenith_seeing'],
            'star': None, 'frame': None, 'last': None, 'last_accepted': None,
        }
        if mon.star:
            t, s = mon.star
            out['star'] = {'t': t, 'x': float(s.x), 'y': float(s.y), 'fwhm': _num(s.fwhm, 3), 'flux': _num(s.flux, 1),
                           'peak': _num(s.peak, 1), 'snr': _num(s.snr, 1), 'n_saturated': int(s.n_saturated)}
        if mon.frame is not None:
            out['frame'] = self.frame_stats(mon.frame)
        for key, r in (('last', last), ('last_accepted', last_ok)):
            if r is not None:
                out[key] = {
                    'time': r['time'].timestamp(), 'accepted': r.get('accepted', True),
                    'zenith': _num(r['zenith'], 3), 'local': _num(r['local'], 3),
                    'r0': _num(r['r0'], 1), 'r0_local': _num(seeing.r0_mm(r['local']), 1),
                    'sigma_x': _num(r['sigma_x'], 4), 'sigma_y': _num(r['sigma_y'], 4),
                    'flux': _num(r['flux'], 1), 'fwhm': _num(r['fwhm'], 3), 'rate': _num(r['rate'], 1),
                    'camera_fps': _num(r.get('camera_fps'), 1), 'drop_fraction': _num(r.get('drop_fraction'), 4),
                    'proc_ms': _num(r.get('proc_ms'), 3),
                }
        pub = mon.publisher
        if pub is not None:
            out['redis'] = {'ok': getattr(pub, 'last_ok', None), 'error': getattr(pub, 'last_error', None)}
        return out

    @staticmethod
    def frame_counts(cam_stats, clock_now):
        """Camera frame rate and dropped fraction over the last ~10 s, plus totals since opening."""
        if not cam_stats:
            return None
        t1, last = cam_stats[-1]
        out = {'totals': last, 'camera_fps': None, 'drop_fraction': None}
        if len(cam_stats) > 1 and clock_now - t1 < 5:
            t0, first = cam_stats[0]
            delivered = last['delivered'] - first['delivered']
            dropped = last['dropped'] - first['dropped']
            if delivered + dropped > 0 and t1 > t0:
                out['camera_fps'] = (delivered + dropped) / (t1 - t0)
                out['drop_fraction'] = dropped / (delivered + dropped)
        return out

    def history_rows(self, days=31):
        t_min = time.time() - days * 86400.0
        with self.mon.lock:
            session = [(r['time'].timestamp(), r['zenith'], r['flux'], r['r0'])
                       for r in self.mon.results if r.get('accepted', True)]
        rows = [r for r in self.history if r[0] >= t_min]
        rows += [r for r in session if r[0] > self.started and r[0] >= t_min]   # files: before startup
        return {'t': [round(r[0], 1) for r in rows], 'seeing': [round(r[1], 3) for r in rows],
                'flux': [round(r[2], 1) for r in rows], 'r0': [round(r[3], 1) for r in rows]}

    def motion(self):
        with self.mon.lock:
            a = np.array(self.mon.recent, dtype=float)
        if len(a) == 0:
            return {'t0': None, 't': [], 'x': [], 'y': [], 'fwhm': []}
        t0 = a[0, 0]
        return {'t0': t0, 't': np.round(a[:, 0] - t0, 4).tolist(), 'x': np.round(a[:, 1], 3).tolist(),
                'y': np.round(a[:, 2], 3).tolist(), 'fwhm': np.round(a[:, 3], 3).tolist()}

    # --- images --------------------------------------------------------------------------------

    def frame_png(self, mode="sqrt"):
        frame = self.mon.frame
        if frame is None:
            return None
        img = frame[1]

        def compute():
            factor = int(np.ceil(img.shape[1] / 1300))
            return png_gray(stretch(bin_max(img, factor), mode)), factor
        return self._cached(('png', mode), img, compute)

    def zoom_png(self, r=16):
        frame, st = self.mon.frame, self.mon.star
        if frame is None:
            return None
        t, img, (x0, y0, w, h) = frame
        if st is not None and abs(st[0] - t) < 5:
            cx, cy = st[1].x - x0, st[1].y - y0
        else:
            cy, cx = np.unravel_index(int(np.argmax(img)), img.shape)
        cx = int(np.clip(round(cx), r, w - r - 1))
        cy = int(np.clip(round(cy), r, h - r - 1))
        cut = img[cy - r:cy + r + 1, cx - r:cx + r + 1]
        return self._cached(('zoom', r), img, lambda: (png_gray(np.ascontiguousarray(cut)),
                                                       (cx - r + x0, cy - r + y0)))


def _handler(state):
    page = resources.files('cyclop').joinpath('static/index.html')

    class Handler(BaseHTTPRequestHandler):
        server_version = "cyclop"

        def log_message(self, fmt, *args):
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, body, ctype, code=200, headers=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj):
            self._send(json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                match url.path:
                    case "/" | "/index.html":
                        self._send(page.read_bytes(), "text/html; charset=utf-8")
                    case "/api/status":
                        self._json(state.status())
                    case "/api/history":
                        self._json(state.history_rows(float(q.get('days', 31))))
                    case "/api/motion":
                        self._json(state.motion())
                    case "/api/log":
                        self._json(state.logbuf.since(int(q.get('after', -1))))
                    case "/api/frame.png":
                        res = state.frame_png(q.get('stretch', 'sqrt'))
                        if res is None:
                            self._send(b"no frame", "text/plain", 404)
                        else:
                            self._send(res[0], "image/png", headers=[("X-Bin", str(res[1]))])
                    case "/api/zoom.png":
                        res = state.zoom_png(int(np.clip(int(q.get('r', 16)), 4, 64)))
                        if res is None:
                            self._send(b"no frame", "text/plain", 404)
                        else:
                            self._send(res[0], "image/png",
                                       headers=[("X-Origin", f"{res[1][0]},{res[1][1]}")])
                    case _:
                        self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log.exception(f"Web request {self.path} failed")
                try:
                    self._send(str(e).encode(), "text/plain", 500)
                except Exception:
                    pass

    return Handler


class WebServer:
    def __init__(self, state, host="0.0.0.0", port=8080):
        self.httpd = ThreadingHTTPServer((host, port), _handler(state))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="web", daemon=True)

    @property
    def url(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self):
        self.thread.start()
        log.info(f"Web interface at {self.url}")
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
