"""
Low-precision solar altitude (Astronomical Almanac approximation, good to ~0.01 deg),
which is plenty for deciding whether it is dark enough to measure.
"""

import math
from datetime import datetime, timezone


def julian_date(t):
    """Julian date of an aware datetime."""
    return t.astimezone(timezone.utc).timestamp() / 86400.0 + 2440587.5


def local_sidereal_time(t, longitude_deg):
    """Local mean sidereal time in hours; longitude is east-positive."""
    n = julian_date(t) - 2451545.0
    return ((280.46061837 + 360.98564736629 * n + longitude_deg) % 360.0) / 15.0


def sun_altitude(t, latitude_deg, longitude_deg):
    """
    Altitude of the Sun in degrees.

    Parameters
    ----------
    t : datetime
        Aware datetime.
    latitude_deg, longitude_deg : float
        Site coordinates; longitude is east-positive.
    """
    n = julian_date(t) - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (280.46061837 + 360.98564736629 * n) % 360.0
    ha = math.radians(gmst + longitude_deg) - ra
    lat = math.radians(latitude_deg)
    alt = math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha))
    return math.degrees(alt)


def is_night(latitude_deg, longitude_deg, max_sun_alt_deg, t=None):
    t = t or datetime.now(timezone.utc)
    return sun_altitude(t, latitude_deg, longitude_deg) <= max_sun_alt_deg
