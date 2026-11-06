"""Bias metrics: pure functions of CalibFrame/AmpGeom and numpy arrays, no I/O.

Callable from notebooks: load frames with common.calio, then call compute_bias_metrics()
or any single metric below. Metric names ending in _e are in electrons (ADU x the amp's
gain from the detector/headers); the rest are in ADU or are counts/ratios.

Inputs
  master        the combined bias (assembled, overscan-subtracted)
  frames        a few single ISR'd biases from the master's inputs (cpBiasIsrExp); raws
                are overscan-corrected here first
  input_meta    [(CalibInfo, metadata dict)] for every input; overscan levels come from
                ISR's 'LSST ISR OVERSCAN SERIAL MEDIAN <amp>', or from the raw overscan
  defects       curated bad-pixel mask (bool, master shape) or None

Single frames are compared against the master (frame - master), which removes the fixed
pattern so that read noise, pickup and inter-amp correlation see only temporal noise.
interamp_row_corr is the correlation of amp-vs-amp row-mean residuals: a proxy for
correlated pickup between amps, not a crosstalk coefficient.
"""

import re
import warnings
from dataclasses import replace

import numpy as np
from scipy.ndimage import median_filter

from common.stats import (bin_image, clipped_profile, clipped_sigma, fit_poly2d, line_ratio,
                          mad_sigma, robust_median, row_power_spectrum, separable_model)

OVERSCAN_KEY = "LSST ISR OVERSCAN SERIAL MEDIAN {amp}"
OVERSCAN_RESIDUAL_KEY = "LSST ISR OVERSCAN RESIDUAL SERIAL STDEV {amp}"

UNITS = {
    "overscan_level_adu": "adu",
    "overscan_stability_adu": "adu",
    "n_outlier_inputs": "exposure",
    "bias_median_adu": "adu",
    "read_noise_e": "electron",
    "read_noise_overscan_e": "electron",
    "single_bias_sigma_e": "electron",
    "col_profile_rms_adu": "adu",
    "row_profile_rms_adu": "adu",
    "poly2d_ptp_adu": "adu",
    "pickup_peak_ratio": "",
    "pickup_peak_freq": "cycle/pixel",
    "interamp_row_corr": "",
    "n_hot_pix": "pixel",
    "n_bad_cols": "column",
    "n_new_defect_pix": "pixel",
    "master_noise_ratio": "",
}

DEFAULT_PARAMS = dict(
    n_noise_frames=6,
    poly_order=2,
    poly_bin=32,
    smooth_box=32,
    profile_nsigma=5.0,
    pickup_fmin=0.01,
    pickup_continuum=51,
    outlier_nsigma=10.0,
    outlier_min_adu=20.0,
    outlier_noise_ratio=1.5,
    hot_nsigma=7.0,
    hot_min_adu=3.0,
    col_nsigma=7.0,
    col_min_adu=1.0,
    col_filter=31,
)


# ----------------------------------------------------------------------- per frame


def overscan_corrected(frame):
    """Raw frame with each amp's overscan median subtracted; other frames unchanged."""
    if not frame.is_raw:
        return frame
    image = frame.image.copy()
    for amp in frame.amps:
        if amp.overscan is not None:
            sl = amp.overscan_slices
            image[amp.slices] -= robust_median(frame.image[sl], frame.bad[sl])
    return replace(frame, image=image)


def overscan_level(frame, amp, metadata=None):
    """Overscan median of one input frame, ADU: ISR metadata, else the raw overscan."""
    value = (metadata or {}).get(OVERSCAN_KEY.format(amp=amp.name))
    if value is not None:
        return float(value)
    if frame is not None and frame.is_raw and amp.overscan is not None:
        sl = amp.overscan_slices
        return robust_median(frame.image[sl], frame.bad[sl])
    return np.nan


def outlier_inputs(levels, noise, nsigma=10.0, min_adu=20.0, noise_ratio=1.5):
    """Inputs whose overscan level is off the sequence median by max(nsigma*MAD-sigma,
    min_adu), or whose overscan noise exceeds noise_ratio x the median. Returns bools.

    overscan_stability_adu is robust by design and hides a single bad frame; this
    catches it (e.g. a mis-typed or light-struck 'zero' left in the combine).
    """
    levels = np.asarray(levels, dtype=float)
    noise = np.asarray(noise, dtype=float)
    out = np.zeros(levels.shape, dtype=bool)
    if levels.size == 0:
        return out
    cut = max(nsigma * mad_sigma(levels), min_adu)
    with np.errstate(invalid="ignore"):
        out |= np.abs(levels - robust_median(levels)) > cut
        med_noise = robust_median(noise)
        if np.isfinite(med_noise) and med_noise > 0:
            out |= noise > noise_ratio * med_noise
    return out


def amp_median(frame, amp):
    return robust_median(*frame.amp_pixels(amp))


def residual(frame, master, amp):
    """(frame - master, bad) for one amp."""
    img, bad = frame.amp_pixels(amp.name)
    mimg, mbad = master.amp_pixels(amp.name)
    return img.astype(float) - mimg, bad | mbad


def frame_sigma_adu(frame, master, amp):
    """Clipped sigma of frame - master: per-frame temporal noise (read noise plus anything
    the frame has that the master does not), ADU."""
    return clipped_sigma(*residual(frame, master, amp))


def pair_read_noise_adu(frame1, frame2, amp):
    """Clipped sigma of the difference of two frames / sqrt(2), ADU.

    Clipped std rather than MAD: raw frames are integer ADU, and the MAD of integer
    differences is quantized (it overestimated DECam read noise by ~30%).
    """
    img1, bad1 = frame1.amp_pixels(amp.name)
    img2, bad2 = frame2.amp_pixels(amp.name)
    return clipped_sigma(img1.astype(float) - img2, bad1 | bad2) / np.sqrt(2.0)


def read_noise_adu(frames, amp):
    """Median over disjoint pairs (0,1), (2,3), ...; NaN with fewer than two frames."""
    values = [pair_read_noise_adu(frames[i], frames[i + 1], amp)
              for i in range(0, len(frames) - 1, 2)]
    return float(np.median(values)) if values else np.nan


def single_frame_sigma_adu(frames, amp):
    """Median over frames of the clipped sigma of one frame (read noise + fixed pattern)."""
    values = [clipped_sigma(*f.amp_pixels(amp.name)) for f in frames]
    return float(np.median(values)) if values else np.nan


def pickup(frames, master, amp, fmin=0.01, continuum_size=51):
    """Strongest narrow line in the row-wise (serial) power spectrum of frame - master.

    The spectrum is averaged over rows and frames, then divided by its running-median
    continuum so a red (1/f) spectrum still gives ~1. Returns (peak ratio, peak
    frequency in cycles/pixel) over f >= fmin.
    """
    spectra = []
    for f in frames:
        res, bad = residual(f, master, amp)
        freq, power = row_power_spectrum(res, bad)
        spectra.append(power)
    if not spectra:
        return np.nan, np.nan
    ratio = line_ratio(np.mean(spectra, axis=0), continuum_size)
    keep = (freq >= fmin) & np.isfinite(ratio)
    if not keep.any():
        return np.nan, np.nan
    i = int(np.argmax(np.where(keep, ratio, -np.inf)))
    return float(ratio[i]), float(freq[i])


def interamp_row_corr(frames, master, amps, nsigma=5.0):
    """{amp: mean Pearson r of its row-mean residual profile with each other amp}.

    Only amps sharing rows are paired; amps with no partner get NaN.
    """
    out = {a.name: [] for a in amps}
    for f in frames:
        profiles = {a.name: clipped_profile(*residual(f, master, a), axis=1, nsigma=nsigma)
                    for a in amps}
        for i, a in enumerate(amps):
            for b in amps[i + 1:]:
                y0, y1 = max(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1])
                if y1 - y0 < 3:
                    continue
                pa = profiles[a.name][y0 - a.bbox[0]:y1 - a.bbox[0]]
                pb = profiles[b.name][y0 - b.bbox[0]:y1 - b.bbox[0]]
                ok = np.isfinite(pa) & np.isfinite(pb)
                if ok.sum() < 3 or np.std(pa[ok]) == 0 or np.std(pb[ok]) == 0:
                    continue
                r = float(np.corrcoef(pa[ok], pb[ok])[0, 1])
                out[a.name].append(r)
                out[b.name].append(r)
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in out.items()}


# ---------------------------------------------------------------------- per master


def profile_rms(frame, amp, axis, nsigma=5.0):
    """MAD-sigma of the column (axis=0) or row (axis=1) clipped-mean profile, ADU."""
    return mad_sigma(clipped_profile(*frame.amp_pixels(amp), axis=axis, nsigma=nsigma))


def poly2d_ptp(frame, amp, order=2, bin_factor=32):
    """Peak-to-peak of a 2D Legendre fit to the median-binned amp, ADU."""
    img, bad = frame.amp_pixels(amp)
    binned = bin_image(img, bad, bin_factor)
    model, _ = fit_poly2d(binned, None, order)
    return float(np.nanmax(model) - np.nanmin(model)) if np.isfinite(model).any() else np.nan


def hot_pixel_map(frame, amp, nsigma=7.0, min_adu=3.0, box=32):
    """Pixels above the local structure (stats.separable_model) by max(nsigma*sigma, min_adu).

    Column/row structure such as edge ramps is part of the model, so this finds isolated
    pixels; whole columns are bad_column_mask's job. Detection uses every finite pixel,
    so already-known defects are found too; sigma comes from the good pixels only.
    """
    img, bad = frame.amp_pixels(amp)
    hp = img - separable_model(img, bad, box)
    cut = max(nsigma * mad_sigma(hp, bad), min_adu)
    with np.errstate(invalid="ignore"):
        return np.isfinite(hp) & (hp > cut)


def bad_column_mask(frame, amp, nsigma=7.0, min_adu=1.0, filter_size=31):
    """Columns whose median departs from a running median of neighbours by
    max(nsigma*sigma, min_adu). Returns a bool per column."""
    img, bad = frame.amp_pixels(amp)
    y = np.where(bad, np.nan, img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmedian(y, axis=0)
    filled = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    dev = prof - median_filter(filled, size=filter_size, mode="nearest")
    cut = max(nsigma * mad_sigma(dev), min_adu)
    with np.errstate(invalid="ignore"):
        return np.isfinite(dev) & (np.abs(dev) > cut)


def new_defect_count(flagged, known):
    """Pixels flagged now but absent from the known defects (curated BPM + mask planes)."""
    return int((flagged & ~known).sum())


def count_inputs(metadata, input_type="cpBiasIsrExp"):
    """Number of frames combined into a master, from its butler provenance headers."""
    n = sum(1 for k, v in metadata.items()
            if re.fullmatch(r"LSST BUTLER INPUT \d+ DATASETTYPE", k) and v == input_type)
    if n:
        return n
    ncombine = metadata.get("NCOMBINE")
    return int(ncombine) if ncombine is not None else np.nan


def master_noise_ratio(master, amp, read_noise_adu, n_inputs, box=32):
    """Measured high-pass master sigma / (single-frame read noise / sqrt(N)).

    ~1 when the master's pixel noise is just averaged-down read noise; >1 means fixed
    pattern or a combine that did not average all inputs.
    """
    if not (np.isfinite(read_noise_adu) and np.isfinite(n_inputs) and n_inputs > 0):
        return np.nan
    img, bad = master.amp_pixels(amp)
    measured = clipped_sigma(img - separable_model(img, bad, box), bad)
    return float(measured / (read_noise_adu / np.sqrt(n_inputs)))


# ------------------------------------------------------------------------- driver


def compute_bias_metrics(master, frames, input_meta, defects=None, params=None,
                         input_type="cpBiasIsrExp"):
    """All bias metrics for one master.

    Returns (results, trend): results is a list of dict(amp, metric, value, unit, error),
    trend one dict per (input exposure, amp) with its overscan level, plus
    frame_sigma_e for the frames read in full (a bad input shows up there).
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    raw_levels = {a.name: [overscan_level(f, f.amp(a.name)) for f in frames if f.is_raw]
                  for a in master.amps}
    frames = [overscan_corrected(f) for f in frames]
    results, trend = [], []

    def add(amp, metric, fn):
        try:
            value, error = fn(), ""
        except Exception as exc:
            value, error = np.nan, f"{type(exc).__name__}: {exc}"[:300]
        if not error and not np.isfinite(value):
            error = "no finite result (empty, fully masked or constant data)"
        results.append(dict(amp=amp.name, metric=metric, value=value,
                            unit=UNITS[metric], error=error))
        return value

    corr = {}
    if frames:
        try:
            corr = interamp_row_corr(frames, master, master.amps, p["profile_nsigma"])
        except Exception:
            corr = {}
    n_inputs = count_inputs(master.metadata, input_type)
    if not np.isfinite(n_inputs) and input_meta:
        n_inputs = len(input_meta)

    for amp in master.amps:
        gain = amp.gain

        amp_trend, levels, resid = [], [], []
        for info, md in input_meta:
            level = overscan_level(None, amp, md)
            sig = md.get(OVERSCAN_RESIDUAL_KEY.format(amp=amp.name))
            levels.append(level)
            resid.append(np.nan if sig is None else float(sig))
            amp_trend.append(dict(detector=master.info.detector, amp=amp.name,
                                  calib_id=master.info.calib_id, exposure=info.exposure,
                                  mjd=info.mjd, overscan_adu=level,
                                  overscan_residual_sigma_adu=resid[-1]))
        n_meta = len(levels)
        levels = np.asarray(levels + raw_levels[amp.name], dtype=float)  # raws: no ISR metadata
        noise = np.concatenate([resid, np.full(levels.size - n_meta, np.nan)])
        outliers = outlier_inputs(levels, noise, p["outlier_nsigma"], p["outlier_min_adu"],
                                  p["outlier_noise_ratio"])
        for row, is_outlier in zip(amp_trend, outliers):
            row["outlier"] = bool(is_outlier)
        by_exposure = {t["exposure"]: t for t in amp_trend}
        for f in frames:
            try:
                sigma_e = frame_sigma_adu(f, master, amp) * gain
            except Exception:
                sigma_e = np.nan
            row = by_exposure.get(f.info.exposure)
            if row is None:
                row = dict(detector=master.info.detector, amp=amp.name,
                           calib_id=master.info.calib_id, exposure=f.info.exposure,
                           mjd=f.info.mjd, overscan_adu=np.nan,
                           overscan_residual_sigma_adu=np.nan, outlier=False)
                amp_trend.append(row)
            row["frame_sigma_e"] = sigma_e
        trend.extend(amp_trend)

        add(amp, "overscan_level_adu", lambda: robust_median(levels))
        add(amp, "overscan_stability_adu", lambda: mad_sigma(levels))
        add(amp, "n_outlier_inputs", lambda: int(outliers.sum()))
        add(amp, "bias_median_adu", lambda: amp_median(master, amp))
        rn_e = add(amp, "read_noise_e", lambda: read_noise_adu(frames, amp) * gain)
        add(amp, "read_noise_overscan_e", lambda: robust_median(np.asarray(resid)) * gain)
        add(amp, "single_bias_sigma_e", lambda: single_frame_sigma_adu(frames, amp) * gain)
        add(amp, "col_profile_rms_adu", lambda: profile_rms(master, amp, 0, p["profile_nsigma"]))
        add(amp, "row_profile_rms_adu", lambda: profile_rms(master, amp, 1, p["profile_nsigma"]))
        add(amp, "poly2d_ptp_adu", lambda: poly2d_ptp(master, amp, p["poly_order"], p["poly_bin"]))

        peak = {}
        if frames:
            add(amp, "pickup_peak_ratio", lambda: peak.setdefault(
                "v", pickup(frames, master, amp, p["pickup_fmin"], p["pickup_continuum"]))[0])
        else:
            add(amp, "pickup_peak_ratio", lambda: np.nan)
        add(amp, "pickup_peak_freq", lambda: peak.get("v", (np.nan, np.nan))[1])
        add(amp, "interamp_row_corr", lambda: corr.get(amp.name, np.nan))

        maps = {}
        add(amp, "n_hot_pix", lambda: int(maps.setdefault("hot", hot_pixel_map(
            master, amp, p["hot_nsigma"], p["hot_min_adu"], p["smooth_box"])).sum()))
        add(amp, "n_bad_cols", lambda: int(maps.setdefault("cols", bad_column_mask(
            master, amp, p["col_nsigma"], p["col_min_adu"], p["col_filter"])).sum()))

        def new_defects():
            flagged = maps["hot"] | maps["cols"][np.newaxis, :]
            known = master.bad if defects is None else (master.bad | defects)
            _, known = replace(master, bad=known).amp_pixels(amp)
            return new_defect_count(flagged, known)

        add(amp, "n_new_defect_pix", new_defects)
        add(amp, "master_noise_ratio",
            lambda: master_noise_ratio(master, amp, rn_e / gain, n_inputs, p["smooth_box"]))
    return results, trend
