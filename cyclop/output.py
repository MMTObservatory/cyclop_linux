"""
Result outputs: text files in the same format as the Windows software, optional per-sample
motion files, and redis keys matching what minicyclop's tcs_logger published.
"""

import logging
import os
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from cyclop import seeing
from cyclop.sun import julian_date

log = logging.getLogger(__name__)

REDIS_PREFIX = "seeing_monitor_"


def _windows_time(t):
    """'10/1/2026 5:57:41 AM' as written by the Windows software."""
    return f"{t.month}/{t.day}/{t.year} {t.strftime('%I').lstrip('0') or '12'}:{t:%M:%S %p}"


def _windows_float(v):
    """' 1.56566504814043E+0000' / '-1.95402344100330E-0001', as in the Windows motion files."""
    m, e = f"{v:.14E}".split("E")
    return f"{m}E{e[0]}{int(e[1:]):04d}".rjust(23)


MOTION_HEADER = ("Date (sec)               PosX  (pixel)            PosY (Pixel)             "
                 "<FWHM>=(FWHMX+FWHMY)/2")


def format_motion(t, x, y, fwhm):
    """Body of a Windows-style *_Motion.txt file: one row per sample, header line last, CRLF."""
    rows = ["  ".join(_windows_float(v) for v in row) for row in zip(t, x, y, fwhm)]
    return "\r\n".join(rows + [MOTION_HEADER]) + "\r\n"


def format_results(result):
    """Body of a Windows-style *_results.txt file."""
    sx, sy = result['sigma_x'], result['sigma_y']
    lines = [
        f"Rms X motion (pixels) : {sx:.3f}",
        f"Rms Y motion (pixels) : {sy:.3f}",
        f"X seeing (arcsec)     : {seeing.axis_seeing(sx):.3f}",
        f"Y seeing (arcsec)     : {seeing.axis_seeing(sy):.3f}",
        f"Total seeing (arcsec) : {result['local']:.3f}",
        f"Zenith seeing (arcsec): {result['zenith']:.3f}",
        f"Zenith R0 (mm)        : {result['r0']:.1f}",
    ]
    return "\r\n".join(lines) + "\r\n"


def format_line(t_utc, flux, zenith_seeing, r0, tz):
    """One Seeing_Data.txt line: UT | local | JD | flux | zenith seeing | r0."""
    return (f"{_windows_time(t_utc)} | {_windows_time(t_utc.astimezone(tz))} | "
            f"{julian_date(t_utc):.7f} | {flux:.1f} | {zenith_seeing:.2f} | {r0:.1f}")


class FileWriter:
    def __init__(self, data_dir, tz="America/Phoenix", save_motion=False, detrend=1):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tz = ZoneInfo(tz)
        self.save_motion = save_motion
        self.detrend = detrend

    def write(self, t_utc, flux, result, samples=None):
        line = format_line(t_utc, flux, result['zenith'], result['r0'], self.tz)
        with open(self.data_dir / "Seeing_Data.txt", "a") as f:
            f.write(line + "\n")
        last = self.data_dir / "Last_Seeing_Data.txt"
        tmp = last.with_suffix(".tmp")
        tmp.write_text(line + "\n")
        os.replace(tmp, last)

        if self.save_motion and samples is not None:
            # Like the Windows software: named by the block's end time in UT, times relative to the
            # first sample, positions with the drift removed (so their rms is the reported sigma).
            t = np.asarray(samples['t'], dtype=float)
            stem = self.data_dir / f"{t_utc.astimezone(timezone.utc):%Y-%m-%d-%Hh%Mm%Ss}"
            x = seeing.remove_drift(t, samples['x'], self.detrend)
            y = seeing.remove_drift(t, samples['y'], self.detrend)
            with open(f"{stem}_Motion.txt", "w", newline="") as f:
                f.write(format_motion(t - t[0], x, y, samples['fwhm']))
            with open(f"{stem}_results.txt", "w", newline="") as f:
                f.write(format_results(result))


class RedisPublisher:
    """
    Sets and publishes seeing_monitor_{seeing,flux,r0,measurement_timestamp}, as tcs_logger did.
    Connection problems are logged and otherwise ignored so they never stop measurements.
    """

    def __init__(self, host=None, port=None, password=None):
        import redis
        host = host or os.environ.get('REDISHOST', 'redis.mmto.arizona.edu')
        port = int(port or os.environ.get('REDISPORT', 6379))
        password = password or os.environ.get('REDISPW')
        self.server = redis.StrictRedis(host=host, port=port, password=password, db=0,
                                        socket_timeout=5, socket_connect_timeout=5)
        log.info(f"Publishing to redis at {host}:{port}")
        self.last_ok = None           # time of the last successful update, for the web interface
        self.last_error = None

    def publish(self, t_utc, flux, result):
        values = {
            'seeing': float(round(result['zenith'], 2)),
            'flux': float(round(flux, 1)),
            'r0': float(round(result['r0'], 1)),
            'measurement_timestamp': t_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
        }
        try:
            for k, v in values.items():
                key = REDIS_PREFIX + k
                self.server.set(key, v)
                self.server.publish(key, v)
            self.last_ok = t_utc.timestamp()
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            log.warning(f"Problem updating seeing values in redis: {e}")
