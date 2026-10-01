"""
Star detection and centroiding.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Star:
    x: float          # centroid in frame pixel coordinates
    y: float
    flux: float       # background-subtracted sum in ADU
    peak: float       # background-subtracted peak in ADU
    fwhm: float       # from second moments, pixels
    n_saturated: int
    snr: float        # peak / background noise


def _box3(img):
    """3x3 box sum, same shape, used to suppress single hot pixels when finding the peak."""
    p = np.pad(img, 1, mode='edge')
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return c[3:, 3:] - c[:-3, 3:] - c[3:, :-3] + c[:-3, :-3]


def measure(img, box=10, min_snr=10.0, saturation=255, guess=None, search=None):
    """
    Find the brightest star in `img` and measure it.

    Parameters
    ----------
    img : 2D ndarray
    box : int
        Half-width of the centroiding box around the peak.
    min_snr : float
        Minimum peak/noise to accept a detection.
    saturation : int
        Pixel value counted as saturated.
    guess : (x, y) or None
        If given with `search`, only look for the peak within `search` pixels of it.
    search : int or None

    Returns
    -------
    Star or None
    """
    a = img.astype(np.float32)
    h, w = a.shape

    y0 = x0 = 0
    region = a
    if guess is not None and search is not None:
        gx, gy = int(round(guess[0])), int(round(guess[1]))
        x0, x1 = max(gx - search, 0), min(gx + search + 1, w)
        y0, y1 = max(gy - search, 0), min(gy + search + 1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        region = a[y0:y1, x0:x1]

    # coarse background and noise from a subsample to keep this cheap on full frames
    step = max(1, int(np.sqrt(a.size / 40000)))
    sample = a[::step, ::step]
    bg = float(np.median(sample))
    noise = float(1.4826 * np.median(np.abs(sample - bg)))
    noise = max(noise, 0.5)

    sm = _box3(region)
    py, px = np.unravel_index(int(np.argmax(sm)), sm.shape)
    py += y0
    px += x0

    xa, xb = max(px - box, 0), min(px + box + 1, w)
    ya, yb = max(py - box, 0), min(py + box + 1, h)
    cut = a[ya:yb, xa:xb]

    # local background from the cutout border
    border = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
    lbg = float(np.median(border))
    s = cut - lbg
    peak = float(s.max())
    snr = peak / noise
    if snr < min_snr:
        return None

    # threshold at a fraction of the peak to keep noise out of the moments
    thr = max(3 * noise, 0.1 * peak)
    wgt = np.where(s > thr, s, 0.0)
    tot = float(wgt.sum())
    if tot <= 0:
        return None
    yy, xx = np.mgrid[ya:yb, xa:xb]
    cx = float((wgt * xx).sum() / tot)
    cy = float((wgt * yy).sum() / tot)
    vx = float((wgt * (xx - cx) ** 2).sum() / tot)
    vy = float((wgt * (yy - cy) ** 2).sum() / tot)
    # Moments of a Gaussian clipped at a fraction f of its peak are biased low: with
    # u = r^2 / 2 sigma^2 ~ Exp(1) truncated at U = ln(1/f), E[u] = 1 - U f / (1 - f).
    f = min(thr / peak, 0.9)
    U = -np.log(f)
    fwhm = 2.3548 * np.sqrt(max((vx + vy) / 2, 0.0) / (1 - U * f / (1 - f)))

    return Star(
        x=cx, y=cy,
        flux=float(s.sum()),
        peak=peak,
        fwhm=float(fwhm),
        n_saturated=int((cut >= saturation).sum()),
        snr=float(snr),
    )
