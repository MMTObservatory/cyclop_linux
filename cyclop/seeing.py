"""
Seeing computation from Polaris image motion.

The constants here were recovered from the Windows SeeingMonitor_Cyclop (v2.3.65) history file
Data.bin, which stores per-measurement centroid sigmas alongside the seeing it reported. Over
300000 measurements (Feb 2025 - Oct 2026) these relations reproduce its output to ~1e-10:

    local seeing ["]  = K * (sigma_x**1.2 + sigma_y**1.2) / 2      (sigmas in pixels)
    zenith seeing ["] = local * cos(90deg - latitude)**0.6
    r0 [mm]           = 550 nm / zenith seeing

The 6/5 power is the standard tip-tilt relation (sigma**2 ~ r0**-5/3, seeing ~ 1/r0); K absorbs
the plate scale (~5"/pixel) and the entrance aperture. The zenith correction uses the latitude
only (Polaris' zenith distance), not the instantaneous altitude of Polaris.
"""

import math

import numpy as np

K_LOCAL = 13.58031732652
WAVELENGTH_M = 550e-9
ARCSEC_PER_RAD = 206264.80624709636

# Data.bin also stores a second "seeing" value that is always local / 1.81034101.
# Its meaning is not documented; it is not exported anywhere so we only keep it for reference.
AUX_RATIO = 1.81034101


def motion_sigma(t, x, y, detrend=1):
    """
    RMS image motion in each axis after removing slow drift of Polaris around the pole.

    Parameters
    ----------
    t, x, y : array-like
        Sample times (s) and full-frame centroid positions (pixels).
    detrend : int or None
        Degree of the polynomial in time removed from each axis; None for no detrending.

    Returns
    -------
    (float, float)
        sigma_x, sigma_y in pixels.
    """
    return tuple(float(np.std(remove_drift(t, v, detrend))) for v in (x, y))


def remove_drift(t, v, detrend=1):
    """Residuals of v(t) about a polynomial fit of degree `detrend` (None: about the mean)."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if detrend is None:
        return v - v.mean()
    tc = t - t.mean()
    return v - np.polyval(np.polyfit(tc, v, detrend), tc)


def inliers(t, x, y, detrend=1, clip=5.0):
    """
    Samples whose drift-removed position lies within `clip` robust standard deviations (1.4826 *
    MAD) of the median on both axes; all True when `clip` is None. A frame where the centroid
    landed on noise or a cosmic ray is far out in the tails and would dominate the variance.
    """
    keep = np.ones(len(t), dtype=bool)
    if clip is None:
        return keep
    for v in (x, y):
        res = remove_drift(t, v, detrend)
        dev = np.abs(res - np.median(res))
        mad = 1.4826 * np.median(dev)
        if mad > 0:
            keep &= dev <= clip * mad
    return keep


def axis_seeing(sigma):
    """Seeing from the image motion along one axis, in arcsec."""
    return K_LOCAL * sigma ** 1.2


def local_seeing(sigma_x, sigma_y):
    """Seeing along the line of sight to Polaris, in arcsec."""
    return (axis_seeing(sigma_x) + axis_seeing(sigma_y)) / 2


def zenith_factor(latitude_deg):
    """Factor converting seeing at Polaris' zenith distance to zenith seeing."""
    return math.cos(math.radians(90.0 - latitude_deg)) ** 0.6


def r0_mm(zenith_seeing):
    """Fried parameter in mm at 550 nm, matching the Windows software's convention."""
    return WAVELENGTH_M / (zenith_seeing / ARCSEC_PER_RAD) * 1e3


def compute(t, x, y, latitude_deg, detrend=1):
    """
    Full seeing reduction for one block of samples.

    Returns
    -------
    dict with sigma_x, sigma_y, local, zenith, r0
    """
    sx, sy = motion_sigma(t, x, y, detrend=detrend)
    local = local_seeing(sx, sy)
    zen = local * zenith_factor(latitude_deg)
    return {
        'sigma_x': sx,
        'sigma_y': sy,
        'local': local,
        'zenith': zen,
        'r0': r0_mm(zen),
    }
