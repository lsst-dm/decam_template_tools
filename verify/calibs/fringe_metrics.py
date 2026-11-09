"""Fringe metrics: pure functions of CalibFrame/AmpGeom and numpy arrays, no I/O.

Callable from notebooks like bias_metrics. All amplitudes are fractions of the sky level.

Inputs
  master        the fringe template: per-pixel fringe / sky (cp_pipe combines
                background-subtracted, sky-normalized frames, cpFringeMeasuredIsrExp)
  frames        science exposures to fit, independent of the template: sky-normalized
                frames (cpFringeMeasuredIsrExp, as is) or ISR'd exposures, which are
                normalized here (normalized_science); their sky level is BGMEAN (ADU)
  frame_meta    [(CalibInfo, metadata dict)] for every input of the template (trend only)
  reference     another period's template (independent inputs), or None

DECam z fringes are weak (~1e-4 of sky rms) next to a template's per-pixel noise
(~5e-4), so amplitudes and fits use `bin` x `bin` binned images and subtract the white
noise expected from neighbouring-pixel differences. Every image is high-passed per amp
(minus a `smooth_box` interpolated median model) so sky gradients and amp offsets don't
count as fringes; the amp offset of the template is reported separately (amp_step_frac).
Pixels within `edge_margin` of the CCD border are ignored.

Do not fit a template to its own inputs: on DECam a single input carries ~5% of the
template (far more than 1/N, through the clipped combine and structure shared between
frames), which inflates the scale several-fold, and a 1/N leave-one-out correction
overshoots into a negative scale. run_fringe_qa.py fits another period's inputs or
held-out science frames instead. With a noisy template the fitted scale is attenuated by
s^2 / (s^2 + n^2) (signal and noise of the template per bin); template_snr says how much.

The fitted scale is the fringe fraction of a frame relative to the template: it falls as
the continuum sky rises (on DECam ~0.9 in dark sky, ~0 at 8x the sky level), so scale
outliers and time jumps use residuals from a scale = a + b / sky fit, in units of each
exposure's fit error plus an intrinsic fractional variability (scale_sky_z).

Coherence between templates from independent inputs is measured against the reference
period (adjacent validity periods use disjoint exposures); a within-period split would
need every input read in full.
"""

import re

import numpy as np

from common.stats import bin_mean, clipped_sigma, interp_smooth, robust_median

SKY_KEY = "BGMEAN"

UNITS = {
    "fringe_rms_frac": "",
    "fringe_ptp_frac": "",
    "template_noise_frac": "",
    "template_snr": "",
    "n_template_inputs": "exposure",
    "amp_step_frac": "",
    "ref_template_corr": "",
    "ref_template_coherence": "",
    "n_science": "exposure",
    "scale_median": "",
    "scale_snr_median": "",
    "residual_rms_frac": "",
    "residual_excess_frac": "",
    "fringe_removed_frac": "",
    "scale_sky_corr": "",
    "n_scale_outliers": "exposure",
    "scale_time_jump_sigma": "",
}

DEFAULT_PARAMS = dict(
    edge_margin=30,
    smooth_box=256,
    bin=8,
    coherence_bin=32,
    boundary_width=20,
    boundary_gap=4,
    outlier_nsigma=5.0,
    scale_intrinsic_frac=0.1,
)


# ---------------------------------------------------------------------- helpers


def edge_mask(shape, margin):
    m = np.zeros(shape, dtype=bool)
    if margin > 0:
        m[:margin] = m[-margin:] = True
        m[:, :margin] = m[:, -margin:] = True
    return m


def highpass(frame, margin=30, box=256):
    """(image minus a per-amp interpolated median model, bad incl. the edge margin), full
    frame, in assembled orientation."""
    bad = frame.bad | edge_mask(frame.image.shape, margin) | ~np.isfinite(frame.image)
    out = np.full(frame.image.shape, np.nan)
    for amp in frame.amps:
        x = frame.image[amp.slices].astype(float)
        b = bad[amp.slices]
        out[amp.slices] = x - interp_smooth(x, b, box)
    out[bad] = np.nan
    return out, bad


def pixel_noise(image, bad):
    """White-noise sigma per pixel from neighbouring-pixel differences / sqrt(2)."""
    d = np.diff(image, axis=1)
    ok = ~(bad[:, 1:] | bad[:, :-1]) & np.isfinite(d)
    return clipped_sigma(d[ok]) / np.sqrt(2.0)


def binned(image, bad, factor):
    """(mean-binned image, good pixels per bin). A bin's white noise is
    pixel_noise / sqrt(count); a median would be ~1.25x noisier and show up as spurious
    fringe 'excess', and so would ignoring partly masked bins."""
    return bin_mean(image, bad | ~np.isfinite(image), factor, return_counts=True)


def binned_noise(pix_noise, counts):
    """rms white noise of the binned values with these good-pixel counts."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    return float(pix_noise * np.sqrt(np.mean(1.0 / c))) if c.size else np.nan


def signal_rms(values, noise):
    """sqrt(max(clipped variance - noise^2, 0)) of binned values: the fringe part."""
    var = clipped_sigma(values) ** 2
    return float(np.sqrt(max(var - noise ** 2, 0.0))) if np.isfinite(var) else np.nan


def count_inputs(metadata, input_type="cpFringeMeasuredIsrExp"):
    n = sum(1 for k, v in metadata.items()
            if re.fullmatch(r"LSST BUTLER INPUT \d+ DATASETTYPE", k) and v == input_type)
    return n if n else np.nan


def normalized_science(frame, margin=30, box=256, nsigma=4.0):
    """Sky-normalized science frame: (x - sky) / sky with sky the per-amp smooth model.

    Frames that are already normalized (|median| << 1, like cpFringeMeasuredIsrExp) are
    returned as is. Bright pixels (> nsigma above the local sky noise) are masked as
    sources when the frame has no DETECTED plane of its own.
    """
    x = frame.image.astype(float)
    if abs(robust_median(x, frame.bad)) < 1.0:
        return frame
    out = np.full(x.shape, np.nan)
    bad = frame.bad | edge_mask(x.shape, margin)
    for amp in frame.amps:
        xa, ba = x[amp.slices], bad[amp.slices]
        sky = interp_smooth(xa, ba, box)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[amp.slices] = xa / sky - 1.0
    sig = clipped_sigma(out, bad)
    with np.errstate(invalid="ignore"):
        bad = bad | (out > nsigma * sig)
    out[bad] = np.nan
    return type(frame)(image=out, bad=bad, amps=frame.amps, metadata=frame.metadata,
                       info=frame.info)


# ---------------------------------------------------------------------- template


def template_amplitude(master, amp, margin=30, box=256, bin_factor=8):
    """(fringe rms, robust p1-p99 peak-to-peak, per-pixel noise, S/N per bin) of one amp,
    all as fractions of sky."""
    hp, bad = highpass(master, margin, box)
    x, b = hp[amp.slices], bad[amp.slices]
    noise = pixel_noise(x, b)
    vb, counts = binned(x, b, bin_factor)
    ok = np.isfinite(vb)
    nb = binned_noise(noise, counts[ok])
    rms = signal_rms(vb[ok], nb)
    ptp = float(np.subtract(*np.percentile(vb[ok], [99, 1]))) if ok.any() else np.nan
    snr = rms / nb if nb > 0 else np.nan
    return rms, ptp, noise, snr


def amp_step(master, a, b, margin=30, width=20, gap=4):
    """median(template) in a strip of amp a minus that of amp b at their shared boundary."""
    left, right = sorted((a, b), key=lambda m: m.bbox[2])
    if left.bbox[3] != right.bbox[2]:
        return np.nan
    bad = master.bad | edge_mask(master.image.shape, margin)
    y0, y1 = max(left.bbox[0], right.bbox[0]), min(left.bbox[1], right.bbox[1])
    x = left.bbox[3]
    strips = {left.name: np.s_[y0:y1, x - gap - width:x - gap],
              right.name: np.s_[y0:y1, x + gap:x + gap + width]}
    med = {k: robust_median(master.image[sl], bad[sl]) for k, sl in strips.items()}
    return med[a.name] - med[b.name]


def template_coherence(master, reference, margin=30, box=256, bin_factor=32):
    """(correlation of the two binned templates, correlation / the value expected for an
    identical pattern given each template's noise). ~1 when the pattern is stable."""
    hm, bm = highpass(master, margin, box)
    hr, br = highpass(reference, margin, box)
    (vm, cm), (vr, cr) = binned(hm, bm | br, bin_factor), binned(hr, bm | br, bin_factor)
    ok = np.isfinite(vm) & np.isfinite(vr)
    if ok.sum() < 10:
        return np.nan, np.nan
    corr = float(np.corrcoef(vm[ok], vr[ok])[0, 1])
    s1 = signal_rms(vm[ok], binned_noise(pixel_noise(hm, bm), cm[ok]))
    s2 = signal_rms(vr[ok], binned_noise(pixel_noise(hr, br), cr[ok]))
    t1, t2 = clipped_sigma(vm[ok]), clipped_sigma(vr[ok])
    expected = (s1 * s2) / (t1 * t2) if t1 > 0 and t2 > 0 else np.nan
    return corr, (corr / expected if expected and expected > 0 else np.nan)


# ---------------------------------------------------------------------- science


def fit_scale(frame, template_hp, template_bad, margin=30, box=256, bin_factor=8):
    """Least-squares fringe scale for one sky-normalized frame on binned images.

    Returns dict(scale, scale_err, pre, post, noise, pre_excess, post_excess): pre/post
    are the clipped rms of the binned frame before/after subtracting scale x template,
    noise the binned white noise, *_excess the fringe-like part above it.
    """
    hp, bad = highpass(frame, margin, box)
    bad = bad | template_bad
    (xb, counts), (tb, _) = binned(hp, bad, bin_factor), binned(template_hp, bad, bin_factor)
    ok = np.isfinite(xb) & np.isfinite(tb)
    x, t, counts = xb[ok], tb[ok], counts[ok]
    keep = np.abs(x - np.median(x)) < 5 * clipped_sigma(x)  # residual sources, edges
    x, t = x[keep], t[keep]
    noise = binned_noise(pixel_noise(hp, bad), counts[keep])
    tt = float(np.sum(t * t))
    if t.size < 10 or tt == 0:
        return None
    scale = float(np.sum(x * t) / tt)
    resid = x - scale * t
    pre, post = clipped_sigma(x), clipped_sigma(resid)
    return dict(scale=scale, scale_err=post / np.sqrt(tt), pre=pre, post=post, noise=noise,
                pre_var=pre ** 2 - noise ** 2, post_var=post ** 2 - noise ** 2,
                pre_excess=float(np.sqrt(max(pre ** 2 - noise ** 2, 0.0))),
                post_excess=float(np.sqrt(max(post ** 2 - noise ** 2, 0.0))))


def scale_sky_z(scales, errors, skies, frac=0.1, nsigma=5.0):
    """Normalized residuals of the scales from a fit of scale = a + b / sky.

    Fringes are OH emission, so their fraction of the sky falls as the continuum rises.
    Each residual is divided by sqrt(error^2 + (frac x model)^2): the fit error plus an
    intrinsic fractional variability `frac` (OH varies from night to night), so dark-sky
    frames with large scales are not judged by the scatter of bright-sky frames near 0.
    The fit is repeated once without |z| > nsigma. With fewer than 4 usable frames the
    model is the median scale.
    """
    s, e = np.asarray(scales, dtype=float), np.asarray(errors, dtype=float)
    k = np.asarray(skies, dtype=float)
    if not np.isfinite(s).any():
        return s

    def z_for(model):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (s - model) / np.hypot(e, frac * model)

    ok = np.isfinite(s) & np.isfinite(e) & np.isfinite(k) & (k > 0)
    if ok.sum() < 4 or np.std(1.0 / k[ok]) == 0:
        return z_for(np.full(s.shape, np.nanmedian(s)))
    x = np.where(ok, 1.0 / np.where(ok, k, 1.0), np.nan)
    z = z_for(np.polyval(np.polyfit(x[ok], s[ok], 1), x))
    with np.errstate(invalid="ignore"):
        keep = ok & (np.abs(z) <= nsigma)
    if 4 <= keep.sum() < ok.sum():
        z = z_for(np.polyval(np.polyfit(x[keep], s[keep], 1), x))
    return z


# ----------------------------------------------------------------------- driver


def compute_fringe_metrics(master, frames, frame_meta, reference=None, params=None,
                           input_type="cpFringeMeasuredIsrExp"):
    """All fringe metrics for one fringe template.

    Returns (results, trend): results is a list of dict(amp, metric, value, unit, error)
    with amp names for per-amp template metrics, '' for whole-CCD metrics and 'A/B' for
    the amp step; trend has one dict per input exposure (sky level from metadata) plus
    the fitted scale and residuals for the frames read in full.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    m, box, nb = p["edge_margin"], p["smooth_box"], p["bin"]
    results, trend = [], []

    def add(amp_name, metric, fn, note=""):
        try:
            value, error = fn(), note
        except Exception as exc:
            value, error = np.nan, f"{type(exc).__name__}: {exc}"[:300]
        if not error and not np.isfinite(value):
            error = "no finite result (empty, fully masked or constant data)"
        results.append(dict(amp=amp_name, metric=metric, value=value, unit=UNITS[metric],
                            error=error))
        return value

    for amp in master.amps:
        amp_vals = {}

        def tmpl(i, amp=amp, amp_vals=amp_vals):
            if not amp_vals:
                amp_vals["v"] = template_amplitude(master, amp, m, box, nb)
            return amp_vals["v"][i]

        add(amp.name, "fringe_rms_frac", lambda: tmpl(0))
        add(amp.name, "fringe_ptp_frac", lambda: tmpl(1))
        add(amp.name, "template_noise_frac", lambda: tmpl(2))
        add(amp.name, "template_snr", lambda: tmpl(3))

    amps = sorted(master.amps, key=lambda a: a.name)
    for i, a in enumerate(amps):
        for b in amps[i + 1:]:
            if a.bbox[3] == b.bbox[2] or b.bbox[3] == a.bbox[2]:
                add(f"{a.name}/{b.name}", "amp_step_frac",
                    lambda a=a, b=b: amp_step(master, a, b, m, p["boundary_width"],
                                              p["boundary_gap"]))

    add("", "n_template_inputs", lambda: count_inputs(master.metadata, input_type))
    no_ref = "" if reference is not None else "no reference template (first period or none given)"
    coh = {}
    if reference is not None:
        try:
            coh["v"] = template_coherence(master, reference, m, box, p["coherence_bin"])
        except Exception as exc:
            no_ref = f"{type(exc).__name__}: {exc}"[:300]
    add("", "ref_template_corr", lambda: coh["v"][0] if coh else np.nan, no_ref)
    add("", "ref_template_coherence", lambda: coh["v"][1] if coh else np.nan, no_ref)

    sky = {info.exposure: float(md.get(SKY_KEY, np.nan)) for info, md in frame_meta}
    for info, md in frame_meta:
        trend.append(dict(detector=master.info.detector, amp="", calib_id=master.info.calib_id,
                          exposure=info.exposure, mjd=info.mjd, sky_adu=sky[info.exposure]))

    fits = []
    if frames:
        t_hp, t_bad = highpass(master, m, box)
        for f in frames:
            fit = fit_scale(normalized_science(f, m, box), t_hp, t_bad, m, box, nb)
            if fit is None:
                continue
            fit.update(exposure=f.info.exposure, mjd=f.info.mjd, calib_id=f.info.calib_id,
                       sky=float(f.metadata.get(SKY_KEY, sky.get(f.info.exposure, np.nan))))
            fits.append(fit)
    for fit in fits:  # fitted frames are another set of exposures: rows of their own
        trend.append(dict(detector=master.info.detector, amp="", calib_id=master.info.calib_id,
                          exposure=fit["exposure"], mjd=fit["mjd"], sky_adu=fit["sky"],
                          scale=fit["scale"], scale_err=fit["scale_err"],
                          pre_rms_frac=fit["pre"], post_rms_frac=fit["post"]))

    no_frames = "" if fits else "no science frames fitted"
    scales = np.array([f["scale"] for f in fits])
    order = np.argsort([f["mjd"] if np.isfinite(f["mjd"]) else f["exposure"] for f in fits])
    add("", "n_science", lambda: len(fits))
    def over_fits(values):
        return float(np.median(values)) if fits else np.nan

    add("", "scale_median", lambda: over_fits(scales), no_frames)
    add("", "scale_snr_median",
        lambda: over_fits([f["scale"] / f["scale_err"] for f in fits]), no_frames)
    add("", "residual_rms_frac", lambda: over_fits([f["post"] for f in fits]), no_frames)
    add("", "residual_excess_frac", lambda: over_fits([f["post_excess"] for f in fits]),
        no_frames)

    def removed():
        # sums of (variance - noise^2), not clipped per frame, so noise averages out
        pre = np.sum([f["pre_var"] for f in fits])
        post = np.sum([f["post_var"] for f in fits])
        return float(1.0 - post / pre) if pre > 0 else np.nan

    add("", "fringe_removed_frac", removed, no_frames)

    def sky_corr():
        s = np.array([f["sky"] for f in fits])
        ok = np.isfinite(s) & np.isfinite(scales)
        if ok.sum() < 3 or np.std(s[ok]) == 0 or np.std(scales[ok]) == 0:
            return np.nan
        return float(np.corrcoef(scales[ok], s[ok])[0, 1])

    add("", "scale_sky_corr", sky_corr, no_frames)
    z = scale_sky_z(scales, [f["scale_err"] for f in fits], [f["sky"] for f in fits],
                    p["scale_intrinsic_frac"], p["outlier_nsigma"])
    add("", "n_scale_outliers",
        lambda: int(np.sum(np.abs(z[np.isfinite(z)]) > p["outlier_nsigma"])), no_frames)

    def time_jump():
        # ~0.95 for pure noise: the median |difference| of two unit normals
        return float(np.nanmedian(np.abs(np.diff(z[order])))) if len(fits) > 1 else np.nan

    add("", "scale_time_jump_sigma", time_jump, no_frames)
    return results, trend
