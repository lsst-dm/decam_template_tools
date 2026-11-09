"""Robust, mask-aware statistics on numpy arrays.

Every function takes an optional boolean `bad` array (True = ignore) and also ignores
non-finite pixels, so callers never pre-filter.
"""

import warnings

import numpy as np

MAD_TO_SIGMA = 1.482602218505602  # 1 / Phi^-1(3/4)


def good_values(x, bad=None):
    """1-D array of the finite values of `x` not flagged in `bad`."""
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x)
    if bad is not None:
        keep &= ~bad
    return x[keep]


def mad_sigma(x, bad=None):
    """Gaussian-equivalent sigma from the median absolute deviation."""
    v = good_values(x, bad)
    if v.size == 0:
        return np.nan
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def clipped_sigma(x, bad=None, nsigma=4.0, iters=10):
    """Standard deviation after iterative nsigma clipping, corrected for the truncation.

    Use this for noise amplitudes: unlike mad_sigma it is not quantized on integer data
    (raw ADU, where the MAD is a whole number). Starts from median/MAD-sigma so gross
    outliers never enter the first estimate.
    """
    from scipy.stats import norm

    v = good_values(x, bad)
    if v.size < 2:
        return np.nan
    center = np.median(v)
    sigma = MAD_TO_SIGMA * np.median(np.abs(v - center))
    if not sigma > 0:
        sigma = v.std()
    keep = np.abs(v - center) <= nsigma * sigma
    for _ in range(iters):
        sel = v[keep]
        if sel.size < 2:
            return np.nan
        center, sigma = sel.mean(), sel.std()
        new = np.abs(v - center) <= nsigma * sigma
        if np.array_equal(new, keep):
            break
        keep = new
    k = nsigma
    factor = np.sqrt(1.0 - 2.0 * k * norm.pdf(k) / (2.0 * norm.cdf(k) - 1.0))
    return float(v[keep].std() / factor)


def robust_median(x, bad=None):
    v = good_values(x, bad)
    return float(np.median(v)) if v.size else np.nan


def clip_mask(x, bad=None, nsigma=5.0):
    """`bad` plus non-finite pixels plus pixels beyond nsigma (MAD) of the median."""
    x = np.asarray(x, dtype=float)
    out = ~np.isfinite(x)
    if bad is not None:
        out |= bad
    v = x[~out]
    if v.size == 0:
        return out
    med = np.median(v)
    sig = MAD_TO_SIGMA * np.median(np.abs(v - med))
    if sig > 0:
        with np.errstate(invalid="ignore"):
            out |= np.abs(x - med) > nsigma * sig
    return out


def masked_to_nan(x, bad=None):
    x = np.array(x, dtype=float)
    if bad is not None:
        x[bad] = np.nan
    return x


def clipped_profile(x, bad=None, axis=0, nsigma=5.0):
    """Mean along `axis` after dropping masked and >nsigma pixels.

    axis=0 gives one value per column, axis=1 one value per row.
    """
    y = masked_to_nan(x, clip_mask(x, bad, nsigma))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(y, axis=axis)


def bin_image(x, bad=None, factor=32):
    """Median-bin by `factor` (trailing rows/columns that don't fill a bin are dropped).

    Bins with no good pixel are NaN.
    """
    y = masked_to_nan(x, bad)
    ny, nx = (s // factor for s in y.shape)
    y = y[:ny * factor, :nx * factor].reshape(ny, factor, nx, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(y, axis=(1, 3))


def smooth_model(x, bad=None, box=32):
    """Full-resolution large-scale model: median-binned by `box`, then block-replicated.

    Edge pixels that don't fill a bin get the nearest bin's value.
    """
    binned = bin_image(x, bad, box)
    if binned.size == 0:
        return np.full(np.shape(x), np.nan)
    model = np.repeat(np.repeat(binned, box, axis=0), box, axis=1)
    ny, nx = np.shape(x)
    pad_y, pad_x = max(0, ny - model.shape[0]), max(0, nx - model.shape[1])
    model = np.pad(model, ((0, pad_y), (0, pad_x)), mode="edge")
    return model[:ny, :nx]


def _legendre_grid(ny, nx):
    return np.linspace(-1.0, 1.0, ny), np.linspace(-1.0, 1.0, nx)


def fit_poly2d(x, bad=None, order=2):
    """Least-squares 2D Legendre fit (total degree in each axis <= order).

    Returns (model, coeffs) with model on the same grid as `x`. Fit is over good pixels.
    """
    from numpy.polynomial import legendre

    x = np.asarray(x, dtype=float)
    ny, nx = x.shape
    yy, xx = _legendre_grid(ny, nx)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    vander = legendre.legvander2d(gy.ravel(), gx.ravel(), [order, order])
    keep = np.isfinite(x).ravel()
    if bad is not None:
        keep &= ~bad.ravel()
    if keep.sum() < vander.shape[1]:
        return np.full_like(x, np.nan), None
    coeffs, *_ = np.linalg.lstsq(vander[keep], x.ravel()[keep], rcond=None)
    return (vander @ coeffs).reshape(ny, nx), coeffs.reshape(order + 1, order + 1)


def row_power_spectrum(x, bad=None):
    """Mean power spectrum of rows (readout direction), DC removed.

    Each row has its median subtracted; masked pixels are set to zero after that so they
    add no power. Returns (freq in cycles/pixel, mean power), excluding f=0.
    """
    y = masked_to_nan(x, bad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = y - np.nanmedian(y, axis=1, keepdims=True)
    y = np.nan_to_num(y, nan=0.0)
    power = np.abs(np.fft.rfft(y, axis=1)) ** 2
    freq = np.fft.rfftfreq(y.shape[1])
    return freq[1:], power[:, 1:].mean(axis=0)


def separable_model(x, bad=None, box=32):
    """Column profile + row profile + smooth residual: the structure of a bias-like frame.

    Column and row medians follow sharp 1-D features (edge ramps, column offsets) that a
    block model smears out; the smooth term takes what is left on scales > `box`.
    x - separable_model(x) is then pixel-scale noise plus isolated defects.
    """
    y = masked_to_nan(x, bad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col = np.nan_to_num(np.nanmedian(y, axis=0))
        row = np.nan_to_num(np.nanmedian(y - col, axis=1))
    rest = np.asarray(x, dtype=float) - col[np.newaxis, :] - row[:, np.newaxis]
    return col[np.newaxis, :] + row[:, np.newaxis] + np.nan_to_num(smooth_model(rest, bad, box))


def line_ratio(power, size=51):
    """power / running-median continuum: ~1 for any smooth (white or red) spectrum,
    >> 1 at a narrow line."""
    from scipy.ndimage import median_filter

    continuum = median_filter(np.asarray(power, dtype=float), size=size, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(continuum > 0, power / continuum, np.nan)


def interp_smooth(x, bad=None, box=64):
    """Large-scale model: median-binned by `box`, then bilinearly interpolated back to
    full resolution (bin centres at (i + 0.5) * box). Unlike smooth_model it has no
    block steps, so x / interp_smooth(x) - 1 is clean at the 0.1% level. Empty bins are
    filled from the nearest good bin; pixels outside the outermost bin centres take the
    edge value.
    """
    from scipy.ndimage import distance_transform_edt, map_coordinates

    binned = bin_image(x, bad, box)
    if binned.size == 0 or not np.isfinite(binned).any():
        return np.full(np.shape(x), np.nan)
    holes = ~np.isfinite(binned)
    if holes.any():
        idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
        binned = binned[tuple(idx)]
    ny, nx = np.shape(x)
    yy = np.clip((np.arange(ny) + 0.5) / box - 0.5, 0, binned.shape[0] - 1)
    xx = np.clip((np.arange(nx) + 0.5) / box - 0.5, 0, binned.shape[1] - 1)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(binned, [gy, gx], order=1, mode="nearest")


def bin_mean(x, bad=None, factor=8, return_counts=False):
    """Mean-bin by `factor` over good pixels (NaN where a bin has none).

    Use instead of bin_image when the binned noise must be known: a median of factor^2
    pixels is ~1.25x noisier than their mean. With return_counts, also returns the number
    of good pixels per bin (a bin's white noise is sigma / sqrt(count)).
    """
    y = masked_to_nan(x, bad)
    ny, nx = (s // factor for s in y.shape)
    y = y[:ny * factor, :nx * factor].reshape(ny, factor, nx, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(y, axis=(1, 3))
    if return_counts:
        return mean, np.isfinite(y).sum(axis=(1, 3))
    return mean
