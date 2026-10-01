"""
Result outputs: text files in the same format as the Windows software, optional per-sample
motion files, and redis keys matching what minicyclop's tcs_logger published.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from cyclop.sun import julian_date

log = logging.getLogger(__name__)

REDIS_PREFIX = "seeing_monitor_"


def _windows_time(t):
    """'10/1/2026 5:57:41 AM' as written by the Windows software."""
    return f"{t.month}/{t.day}/{t.year} {t.strftime('%I').lstrip('0') or '12'}:{t:%M:%S %p}"


def format_line(t_utc, flux, zenith_seeing, r0, tz):
    """One Seeing_Data.txt line: UT | local | JD | flux | zenith seeing | r0."""
    return (f"{_windows_time(t_utc)} | {_windows_time(t_utc.astimezone(tz))} | "
            f"{julian_date(t_utc):.7f} | {flux:.1f} | {zenith_seeing:.2f} | {r0:.1f}")


class FileWriter:
    def __init__(self, data_dir, tz="America/Phoenix", save_motion=False):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tz = ZoneInfo(tz)
        self.save_motion = save_motion

    def write(self, t_utc, flux, result, samples=None):
        line = format_line(t_utc, flux, result['zenith'], result['r0'], self.tz)
        with open(self.data_dir / "Seeing_Data.txt", "a") as f:
            f.write(line + "\n")
        last = self.data_dir / "Last_Seeing_Data.txt"
        tmp = last.with_suffix(".tmp")
        tmp.write_text(line + "\n")
        os.replace(tmp, last)

        if self.save_motion and samples is not None:
            t0 = samples['t'][0]
            start = datetime.fromtimestamp(t0, timezone.utc).astimezone(self.tz)
            name = self.data_dir / f"{start:%Y-%m-%d-%Hh%Mm%Ss}_motion.txt"
            with open(name, "w") as f:
                for t, x, y, fw in zip(samples['t'], samples['x'], samples['y'], samples['fwhm']):
                    f.write(f"{t - t0:.4f}\t{x:.4f}\t{y:.4f}\t{fw:.3f}\n")


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
        except Exception as e:
            log.warning(f"Problem updating seeing values in redis: {e}")
