from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from cyclop import config, seeing, sun
from cyclop.output import format_line

LAT = config.DEFAULTS['site']['latitude']

# Records from the Windows software's Data.bin:
# local TDateTime, UTC TDateTime, JD, aux seeing, local seeing, flux, sigma_x, sigma_y
DATABIN = [
    (45704.93019748843, 45705.22186415509, 2460723.721864155, 0.8958278891899969, 1.621753963051285,
     512.722585727347, 0.1634270750281891, 0.176869902913846),
    (45971.24183403935, 45971.53350070602, 2460990.033500706, 0.9174694025862775, 1.6609324822069906,
     1841.9765796880656, 0.13503299487469708, 0.21049828337057244),
    (46296.2483934375, 46296.54006010417, 2461315.0400601043, 0.621791092662614, 1.1256539128597194,
     1221.6157318446913, 0.11187939753551929, 0.13888833402435882),
]

# The matching lines that the Windows software wrote to Seeing_Data.txt
SEEING_DATA = {
    0: "2/17/2025 5:19:29 AM | 2/16/2025 10:19:29 PM | 2460723.7218642 | 512.7 | 1.10 | 102.9",
    2: "10/1/2026 12:57:41 PM | 10/1/2026 5:57:41 AM | 2461315.0400601 | 1221.6 | 0.76 | 148.3",
}


def tdatetime(days):
    return datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=days)


@pytest.mark.parametrize("rec", DATABIN)
def test_local_seeing_matches_windows(rec):
    assert seeing.local_seeing(rec[6], rec[7]) == pytest.approx(rec[4], rel=1e-8)
    assert rec[4] / rec[3] == pytest.approx(seeing.AUX_RATIO, rel=1e-8)


@pytest.mark.parametrize("i", sorted(SEEING_DATA))
def test_seeing_data_line_matches_windows(i):
    rec = DATABIN[i]
    t = tdatetime(rec[1])
    zen = rec[4] * seeing.zenith_factor(LAT)
    line = format_line(t, rec[5], zen, seeing.r0_mm(zen), ZoneInfo("America/Phoenix"))
    assert line == SEEING_DATA[i]
    assert sun.julian_date(t) == pytest.approx(rec[2], abs=1e-8)


def test_motion_sigma_removes_drift():
    rng = np.random.default_rng(1)
    t = np.arange(3000) / 60.0
    x = 100 + 0.15 * t + rng.normal(0, 0.2, t.size)
    y = 200 - 0.07 * t + rng.normal(0, 0.1, t.size)
    sx, sy = seeing.motion_sigma(t, x, y, detrend=1)
    assert sx == pytest.approx(0.2, rel=0.05)
    assert sy == pytest.approx(0.1, rel=0.05)


def test_sun_altitude():
    # 2026-06-21 noon MST at the MMTO: Sun about 81.8 deg high (90 - 31.69 + 23.44)
    t = datetime(2026, 6, 21, 19, 30, tzinfo=timezone.utc)
    alt = sun.sun_altitude(t, LAT, config.DEFAULTS['site']['longitude'])
    assert alt == pytest.approx(81.7, abs=0.5)
    # local midnight is dark
    t = datetime(2026, 6, 22, 7, 30, tzinfo=timezone.utc)
    assert sun.sun_altitude(t, LAT, config.DEFAULTS['site']['longitude']) < -30
