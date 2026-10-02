from datetime import datetime, timezone

import numpy as np
import pytest

from cyclop import seeing
from cyclop.cli import read_motion
from cyclop.output import FileWriter, format_motion

# first row of 2026-10-02-02h05m25s_Motion.txt from the Windows software
WINDOWS_ROW = (" 0.00000000000000E+0000  -1.95402344100330E-0001   2.26029033354337E-0001"
               "   1.56566504814043E+0000")
WINDOWS_RESULTS = """\
Rms X motion (pixels) : 0.231
Rms Y motion (pixels) : 0.257
X seeing (arcsec)     : 2.336
Y seeing (arcsec)     : 2.658
Total seeing (arcsec) : 2.497
Zenith seeing (arcsec): 1.697
Zenith R0 (mm)        : 66.9
"""


def test_motion_row_format():
    body = format_motion([0.0], [-0.195402344100330], [0.226029033354337], [1.56566504814043])
    first, header, end = body.split("\r\n")
    assert first == WINDOWS_ROW and header.startswith("Date (sec)") and end == ""


def test_writer_motion_files(tmp_path):
    rng = np.random.default_rng(5)
    n = 3000
    t = 1.79e9 + np.arange(n) / 58.0
    # drift plus jitter scaled to the Windows block's rms
    jx, jy = (seeing.remove_drift(t, rng.normal(0, 1, n)) for _ in range(2))
    x = 1100 + 0.4 * (t - t[0]) + 0.231 * jx / jx.std()
    y = 800 - 0.2 * (t - t[0]) + 0.257 * jy / jy.std()
    r = seeing.compute(t, x, y, 31.68)
    end = datetime(2026, 10, 2, 2, 5, 25, tzinfo=timezone.utc)
    FileWriter(tmp_path, save_motion=True).write(end, 1674.0, r, samples={
        't': t, 'x': x, 'y': y, 'fwhm': np.full(n, 1.9), 'flux': np.full(n, 1674.0)})

    motion = tmp_path / "2026-10-02-02h05m25s_Motion.txt"     # end time, in UT
    raw = motion.read_bytes()
    assert raw.count(b"\r\n") == n + 1 and b"\n" not in raw.replace(b"\r\n", b"")
    a = read_motion(motion)
    assert a.shape == (n, 4) and a[0, 0] == 0
    # positions are saved with the drift removed: their plain rms is the reported sigma
    assert a[:, 1].std() == pytest.approx(r['sigma_x'], rel=1e-9)
    assert abs(np.polyfit(a[:, 0], a[:, 1], 1)[0]) < 1e-9
    results = (tmp_path / "2026-10-02-02h05m25s_results.txt").read_bytes().decode()
    assert results.replace("\r\n", "\n").splitlines()[:2] == WINDOWS_RESULTS.splitlines()[:2]
    assert results.endswith("\r\n") and len(results.splitlines()) == 7
