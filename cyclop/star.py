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


def _box3_cube(cube):
    """`_box3` applied to every frame of an (n, h, w) uint8 stack, in exact integer arithmetic."""
    p = np.pad(cube.astype(np.int16), ((0, 0), (1, 1), (1, 1)), mode='edge')
    r = p[:, :-2] + p[:, 1:-1] + p[:, 2:]
    return r[:, :, :-2] + r[:, :, 1:-1] + r[:, :, 2:]


def _row_median(v):
    """Median of each row of an (n, m) array of integers 0..255 with m odd, via histograms."""
    n, m = v.shape
    offset = 256 * np.arange(n, dtype=np.int32)[:, None]
    hist = np.bincount((v + offset).ravel(), minlength=256 * n).reshape(n, 256)
    return (hist.cumsum(axis=1) > m // 2).argmax(axis=1)


def measure_cube(cube, box=10, min_snr=10.0, saturation=255):
    """
    `measure` vectorized over a stack of cutouts, each searched in full for its brightest star.

    Parameters
    ----------
    cube : (n, h, w) uint8 ndarray
        Cutouts around the star, h * w odd and larger than the centroiding box (2 * box + 1).

    Returns
    -------
    dict of length-n arrays: x, y (cutout pixel coordinates), flux, peak, fwhm, n_saturated, snr,
    and `ok`, False where `measure` would have returned None. The centroiding box is kept inside
    the cutout rather than truncated at its edge; otherwise the arithmetic is the same.
    """
    n, h, w = cube.shape
    b = 2 * box + 1
    if cube.dtype != np.uint8 or h * w % 2 == 0 or h < b or w < b:
        raise ValueError(f"need uint8 cutouts with an odd number of pixels, at least {b}x{b}")
    a = cube.astype(np.float32)
    # same median / MAD as `measure`, exact on integers (the median of an odd count is a pixel value)
    flat = cube.reshape(n, -1).astype(np.int16)
    bg = _row_median(flat)
    mad = _row_median(np.abs(flat - bg[:, None].astype(np.int16)))
    noise = np.maximum(1.4826 * mad, 0.5).astype(np.float32)

    peak_at = _box3_cube(cube).reshape(n, -1).argmax(axis=1)
    py, px = np.divmod(peak_at, w)
    ya = np.clip(py - box, 0, h - b)
    xa = np.clip(px - box, 0, w - b)
    k = np.arange(b)
    rows = (ya[:, None] + k)[:, :, None]
    cols = (xa[:, None] + k)[:, None, :]
    cut = a[np.arange(n)[:, None, None], rows, cols]                # (n, b, b)

    border = np.concatenate([cut[:, 0], cut[:, -1], cut[:, 1:-1, 0], cut[:, 1:-1, -1]], axis=1)
    s = cut - np.median(border, axis=1)[:, None, None]
    peak = s.reshape(n, -1).max(axis=1)
    snr = peak / noise
    thr = np.maximum(3 * noise, 0.1 * peak)
    wgt = np.where(s > thr[:, None, None], s, 0.0)
    tot = wgt.sum(axis=(1, 2))
    ok = (snr >= min_snr) & (tot > 0)
    tot_safe = np.where(tot > 0, tot, 1.0)
    yy, xx = rows.astype(float), cols.astype(float)
    cx = (wgt * xx).sum(axis=(1, 2)) / tot_safe
    cy = (wgt * yy).sum(axis=(1, 2)) / tot_safe
    vx = (wgt * (xx - cx[:, None, None]) ** 2).sum(axis=(1, 2)) / tot_safe
    vy = (wgt * (yy - cy[:, None, None]) ** 2).sum(axis=(1, 2)) / tot_safe
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.clip(thr / np.where(peak > 0, peak, np.inf), 1e-6, 0.9)
        U = -np.log(f)
        fwhm = 2.3548 * np.sqrt(np.maximum((vx + vy) / 2, 0.0) / (1 - U * f / (1 - f)))
    return {
        'x': cx, 'y': cy,
        'flux': s.sum(axis=(1, 2)).astype(float),
        'peak': peak.astype(float),
        'fwhm': fwhm,
        'n_saturated': (cut >= saturation).sum(axis=(1, 2)),
        'snr': snr.astype(float),
        'ok': ok,
    }


def _gaussian_kernel(sigma):
    r = int(np.ceil(3 * sigma))
    return np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)


def _log_parabola(lo, mid, hi):
    """Sub-pixel offset of a peak from three samples, fitting a Gaussian (a parabola in log)."""
    lo, mid, hi = (np.log(np.maximum(v, 1e-6)) for v in (lo, mid, hi))
    d = lo - 2 * mid + hi
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.where(d < 0, 0.5 * (lo - hi) / d, 0.0)
    return np.clip(off, -0.5, 0.5)


def correlate_cube(cube, gx, gy, sigma=1.0, reach=10):
    """
    Star positions in a stack of cutouts by cross-correlation with a Gaussian (a matched filter).

    Each frame, background subtracted, is correlated with a Gaussian of width `sigma` over a window
    `reach` pixels around the guess (gx, gy); the position is the correlation peak, refined to
    sub-pixel by fitting a Gaussian through it and its neighbours along each axis. Every pixel
    contributes in proportion to the expected signal there, so read noise in the wings, which
    dominates thresholded or aperture centroids of a ~2 px star, is weighted down, and there is no
    threshold or aperture edge for the star to cross. Searching only near the guess keeps a faint
    frame from locking onto noise elsewhere in the cutout.

    Parameters
    ----------
    cube : (n, h, w) uint8 ndarray
        Cutouts around the star, h * w odd.
    gx, gy : float or length-n arrays
        Expected position in each cutout, pixels.
    sigma : float
        Gaussian sigma, pixels; close to the star's (FWHM / 2.355) is best, the result is not
        sensitive to it.
    reach : int
        Search half-width around the guess, pixels.

    Returns
    -------
    dict of length-n arrays: x, y (cutout pixel coordinates) and snr, the correlation peak over
    its noise (the S/N of a matched-filter detection).
    """
    n, h, w = cube.shape
    if cube.dtype != np.uint8 or h * w % 2 == 0:
        raise ValueError("need uint8 cutouts with an odd number of pixels")
    g = _gaussian_kernel(sigma)
    r = len(g) // 2
    m = 2 * (reach + r) + 1
    if h < m or w < m:
        raise ValueError(f"cutouts must be at least {m}x{m} for reach={reach}, sigma={sigma}")

    flat = cube.reshape(n, -1).astype(np.int16)
    bg = _row_median(flat)
    noise = np.maximum(1.4826 * _row_median(np.abs(flat - bg[:, None].astype(np.int16))), 0.5)

    gx = np.broadcast_to(np.rint(gx).astype(int), (n,))
    gy = np.broadcast_to(np.rint(gy).astype(int), (n,))
    ya = np.clip(gy - reach - r, 0, h - m)
    xa = np.clip(gx - reach - r, 0, w - m)
    k = np.arange(m)
    win = cube[np.arange(n)[:, None, None], (ya[:, None] + k)[:, :, None], (xa[:, None] + k)[:, None, :]]
    win = win.astype(np.float32) - bg[:, None, None].astype(np.float32)

    # separable correlation over the window ('valid' part: one value per candidate position)
    q = m - 2 * r
    c = sum(g[j] * win[:, j:j + q, :] for j in range(len(g)))
    c = sum(g[j] * c[:, :, j:j + q] for j in range(len(g)))

    at = c[:, 1:-1, 1:-1].reshape(n, -1).argmax(axis=1)     # keep a neighbour on every side
    py, px = np.divmod(at, q - 2)
    py += 1
    px += 1
    i = np.arange(n)
    peak = c[i, py, px]
    fx = px + _log_parabola(c[i, py, px - 1], peak, c[i, py, px + 1])
    fy = py + _log_parabola(c[i, py - 1, px], peak, c[i, py + 1, px])
    return {
        'x': xa + r + fx,
        'y': ya + r + fy,
        # noise in the correlation is noise * sqrt(sum of squared 2-d weights) = noise * sum(g^2)
        'snr': (peak / (noise * float((g ** 2).sum()))).astype(float),
    }


def column_flux_factor(x, terms):
    """
    Relative sensitivity at star column position(s) `x` (full-frame pixels), for dividing out of flux.

    The DMK 33GP031's sensor reads columns in a pattern that repeats every 4 columns, so a star's
    flux varies by ~+-9% as it drifts across them. `terms` = [a1, b1, a2, b2, ...] gives the
    factor 1 + sum_k (a_k cos(k pi x / 2) + b_k sin(k pi x / 2)); empty means no correction.
    """
    x = np.asarray(x, dtype=float)
    f = np.ones_like(x)
    for k, (a, b) in enumerate(zip(terms[::2], terms[1::2]), start=1):
        f += a * np.cos(k * np.pi * x / 2) + b * np.sin(k * np.pi * x / 2)
    return f
