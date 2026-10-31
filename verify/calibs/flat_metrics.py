"""Flat metrics: pure functions of CalibFrame/AmpGeom and numpy arrays, no I/O.

Callable from notebooks like bias_metrics. Fractions are relative to the local flat level
(1 = 100%); levels of the input frames are in ADU.

Inputs
  master        the combined flat (assembled; normalized but still in ADU space, so amps
                differ by their gain ratio)
  frames        a few single ISR'd flats from the master's inputs (cpFlatIsrExp)
  input_meta    [(CalibInfo, metadata dict)] for every input: levels and saturated-pixel
                counts come from ISR's 'LSST ISR FINAL MEDIAN <amp>' / 'LSST ISR MASK SAT <amp>'
  reference     an earlier/known-good master flat for the same detector and filter, or None

DECam CCD edges are bright (the outer ~10 px read 1.3-2x the interior, the outermost
columns are dead), so every pixel-level metric ignores `edge_margin` pixels at the CCD
border; the edges have their own metric, edge_feature_frac. High-pass images are
master / interp_smooth(master) - 1 per amp, with dead/hot pixels kept out of the model.

treering_proxy_frac is the RMS of the high-pass image smoothed on a few pixels: on DECam
it is dominated by tree rings, but it is a proxy (no ring centre is fitted).
Photometric closure (zero-point scatter vs focal-plane position) needs source catalogs
and is not implemented here; see photometric_closure().
"""

import warnings
from dataclasses import replace

import numpy as np
from scipy import ndimage

from common.stats import (bin_image, clipped_sigma, fit_poly2d, interp_smooth, mad_sigma,
                          robust_median)

LEVEL_KEY = "LSST ISR FINAL MEDIAN {amp}"
SAT_KEY = "LSST ISR MASK SAT {amp}"

UNITS = {
    "input_level_adu": "adu",
    "input_level_scatter_frac": "",
    "n_low_count_inputs": "exposure",
    "max_level_sat_frac": "",
    "max_sat_pix_frac": "",
    "master_median": "",
    "pix_rms_frac": "",
    "master_poisson_frac": "",
    "pix_rms_poisson_ratio": "",
    "input_noise_ratio": "",
    "low_qe_frac": "",
    "n_dust_spots": "spot",
    "treering_proxy_frac": "",
    "edge_feature_frac": "",
    "amp_ratio": "",
    "amp_ratio_vs_gain": "",
    "gradient_ptp_frac": "",
    "ref_amp_ratio": "",
    "ref_ratio_rms": "",
    "ref_ratio_frac_out": "",
    "n_new_dust_spots": "spot",
}

DEFAULT_PARAMS = dict(
    edge_margin=30,
    model_box=128,
    pix_box=32,
    low_count_adu=5000.0,
    low_qe_level=0.9,
    treering_sigma=3.0,
    spot_sigma=2.0,
    spot_nsigma=5.0,
    spot_min_area=12,
    spot_max_area=40000,
    spot_max_elong=3.0,
    spot_min_depth=0.005,
    edge_width=10,
    edge_ref=(20, 60),
    boundary_width=20,
    boundary_gap=4,
    gradient_order=2,
    gradient_bin=64,
    ref_tol=0.01,
)


# --------------------------------------------------------------------- helpers


def edge_mask(shape, margin):
    """True within `margin` pixels of the image border."""
    m = np.zeros(shape, dtype=bool)
    if margin > 0:
        m[:margin] = m[-margin:] = True
        m[:, :margin] = m[:, -margin:] = True
    return m


def core_frame(frame, margin):
    """Copy of `frame` whose bad mask also covers the CCD edge margin."""
    return replace(frame, bad=frame.bad | edge_mask(frame.image.shape, margin))


def amp_highpass(frame, amp, box=128):
    """(x / interp_smooth(x) - 1, bad) for one amp; pixels below 0.5x or above 2x the amp
    median (dead, hot) are kept out of the smooth model but stay in the result."""
    x, bad = frame.amp_pixels(amp)
    x = x.astype(float)
    med = robust_median(x, bad)
    with np.errstate(invalid="ignore"):
        model_bad = bad | ~(np.abs(x / med - 1) < 1.0) | (x < 0.5 * med)
    model = interp_smooth(x, model_bad, box)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / model - 1.0, bad


def detect_spots(hp, bad, sigma=2.0, nsigma=5.0, min_area=12, max_area=40000, max_elong=3.0,
                 min_depth=0.0):
    """Compact depressions in a high-pass image: dust specks/donuts, not tree-ring arcs.

    hp is smoothed with a Gaussian of `sigma` px and thresholded at -nsigma x its clipped
    sigma. Connected regions are kept if min_area <= area <= max_area, their bounding
    box is no more than max_elong times longer than wide, and their deepest smoothed
    pixel is below -min_depth. Returns [(y, x, area, depth)]. On DECam the six tape
    bumps per CCD (corners and mid long edges) are found too; they are stable.
    """
    s = ndimage.gaussian_filter(np.where(bad | ~np.isfinite(hp), 0.0, hp), sigma)
    sig = clipped_sigma(s, bad)
    if not np.isfinite(sig):
        return []
    labels, n = ndimage.label((s < -max(nsigma * sig, 1e-9)) & ~bad)  # noiseless: any dip
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum_labels(np.ones_like(s), labels, idx)
    depths = ndimage.minimum(s, labels, idx)
    centres = ndimage.center_of_mass(np.ones_like(s), labels, idx)
    spots = []
    for k, sl in enumerate(ndimage.find_objects(labels)):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (min_area <= areas[k] <= max_area):
            continue
        if max(h, w) > max_elong * min(h, w) or depths[k] > -min_depth:
            continue
        spots.append((float(centres[k][0]), float(centres[k][1]), int(areas[k]),
                      float(depths[k])))
    return spots


def boundary_strips(frame, a, b, width=20, gap=4):
    """Pixels of amps a and b in `width`-column strips either side of their shared
    vertical boundary (skipping `gap` columns). None if they don't share one."""
    left, right = sorted((a, b), key=lambda m: m.bbox[2])
    if left.bbox[3] != right.bbox[2]:
        return None
    y0, y1 = max(left.bbox[0], right.bbox[0]), min(left.bbox[1], right.bbox[1])
    if y1 <= y0:
        return None
    x = left.bbox[3]
    sl_l = np.s_[y0:y1, x - gap - width:x - gap]
    sl_r = np.s_[y0:y1, x + gap:x + gap + width]
    strips = {left.name: (frame.image[sl_l], frame.bad[sl_l]),
              right.name: (frame.image[sl_r], frame.bad[sl_r])}
    return strips[a.name], strips[b.name]


def amp_pairs(amps):
    """Pairs of amps sharing a vertical boundary, each ordered by name ('A', 'B')."""
    out = []
    for i, a in enumerate(amps):
        for b in amps[i + 1:]:
            if a.bbox[3] == b.bbox[2] or b.bbox[3] == a.bbox[2]:
                out.append(tuple(sorted((a, b), key=lambda m: m.name)))
    return out


# ------------------------------------------------------------------ input frames


def well_exposed(input_meta, low_count_adu, max_sat_frac=0.8):
    """Inputs with every amp at or above low_count_adu and below max_sat_frac of its
    SATURAT<amp> header (when present): the ones worth reading as noise frames."""
    prefix = LEVEL_KEY.format(amp="")
    out = []
    for info, md in input_meta:
        levels = {k[len(prefix):]: float(v) for k, v in md.items() if k.startswith(prefix)}
        if not levels or min(levels.values()) < low_count_adu:
            continue
        if any(v > max_sat_frac * float(md.get(f"SATURAT{a}", np.inf))
               for a, v in levels.items()):
            continue
        out.append(info)
    return out


def input_levels(input_meta, amp):
    """(levels ADU, saturated pixel counts) of every input from ISR metadata."""
    levels = np.array([float(md.get(LEVEL_KEY.format(amp=amp.name), np.nan))
                       for _, md in input_meta])
    sat = np.array([float(md.get(SAT_KEY.format(amp=amp.name), np.nan))
                    for _, md in input_meta])
    return levels, sat


def input_noise_ratio(frames, master, amp, margin=30, n_inputs=None):
    """Measured / Poisson-expected pixel noise of single flats divided by the master.

    frame / (scale x master) removes the flat pattern (PRNU, tree rings, dust), leaving
    shot + read noise; expected sigma = sqrt(S g + rn^2) / (S g) for a level S ADU and
    gain g, times sqrt(1 - 1/n_inputs) when the frames are among the master's n_inputs
    (the frame's own noise is partly in the master). ~1 for a correct gain;
    brighter-fatter pulls it below 1.
    """
    values = []
    for f in frames:
        x, bad = core_frame(f, margin).amp_pixels(amp.name)
        m, mbad = master.amp_pixels(amp.name)
        bad = bad | mbad
        level = robust_median(x, bad)
        scale = level / robust_median(m, bad)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = x / (scale * m) - 1.0
        measured = clipped_sigma(r, bad)
        electrons = level * amp.gain
        rn = amp.read_noise if np.isfinite(amp.read_noise) else 0.0
        expected = np.sqrt(electrons + rn ** 2) / electrons
        if n_inputs and n_inputs > 1:
            expected *= np.sqrt(1.0 - 1.0 / n_inputs)
        values.append(measured / expected)
    return float(np.median(values)) if values else np.nan


# ------------------------------------------------------------------------ master


def stitched(frame, width=20, gap=4):
    """Image with each amp scaled so adjacent amps agree at their shared boundary
    (chained from the left-most amp): removes the gain step, keeps the illumination."""
    image = frame.image.astype(float).copy()
    amps = sorted(frame.amps, key=lambda m: m.bbox[2])
    for left, right in zip(amps, amps[1:]):
        strips = boundary_strips(replace(frame, image=image), left, right, width, gap)
        if strips is None:
            continue
        (xl, bl), (xr, br) = strips
        image[right.slices] *= robust_median(xl, bl) / robust_median(xr, br)
    return image


def gradient_ptp(frame, order=2, bin_factor=64, width=20, gap=4):
    """Peak-to-peak / median of an order-N Legendre fit to the binned, stitched CCD."""
    binned = bin_image(stitched(frame, width, gap), frame.bad, bin_factor)
    binned = binned / np.nanmedian(binned)
    model, _ = fit_poly2d(binned, ~np.isfinite(binned), order)
    return float(np.nanmax(model) - np.nanmin(model)) if np.isfinite(model).any() else np.nan


def edge_feature(frame, amp, width=10, ref=(20, 60)):
    """Signed largest (outer `width` px / band `ref` px in from the edge) - 1 over the
    amp's sides that lie on the CCD border. >0 bright edges, <0 roll-off."""
    ny, nx = frame.image.shape
    y0, y1, x0, x1 = amp.bbox
    lo, hi = ref
    sides = []
    if x0 == 0:
        sides.append((np.s_[y0:y1, 0:width], np.s_[y0:y1, lo:hi]))
    if x1 == nx:
        sides.append((np.s_[y0:y1, nx - width:nx], np.s_[y0:y1, nx - hi:nx - lo]))
    if y0 == 0:
        sides.append((np.s_[0:width, x0:x1], np.s_[lo:hi, x0:x1]))
    if y1 == ny:
        sides.append((np.s_[ny - width:ny, x0:x1], np.s_[ny - hi:ny - lo, x0:x1]))
    devs = [robust_median(frame.image[o], frame.bad[o]) / robust_median(frame.image[r], frame.bad[r])
            - 1.0 for o, r in sides]
    devs = [d for d in devs if np.isfinite(d)]
    return float(max(devs, key=abs)) if devs else np.nan


def reference_ratio(core, reference, amp, tol=0.01):
    """master / reference for one amp: (amp ratio relative to the CCD, clipped rms of the
    amp-normalised ratio, fraction of pixels off by more than `tol`, relative image)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = core.image.astype(float) / reference.image
    bad = core.bad | reference.bad | ~np.isfinite(ratio)
    ccd = robust_median(ratio, bad)
    full = replace(core, image=ratio, bad=bad)
    r, rb = full.amp_pixels(amp)
    med = robust_median(r, rb)
    rel = r / med - 1.0
    good = ~rb & np.isfinite(rel)
    frac_out = float(np.mean(np.abs(rel[good]) > tol)) if good.any() else np.nan
    return med / ccd, clipped_sigma(rel, rb), frac_out, rel, rb


def photometric_closure(*args, **kwargs):
    """Zero-point scatter vs focal-plane position from flat-fielded science catalogs.

    Not implemented: needs source catalogs and a reference catalog, outside the scope of
    this QA so far.
    """
    raise NotImplementedError("photometric closure test is not implemented yet")


# ------------------------------------------------------------------------ driver


def compute_flat_metrics(master, frames, input_meta, reference=None, params=None):
    """All flat metrics for one master flat.

    Returns (results, trend): results is a list of dict(amp, metric, value, unit, error),
    where amp is an amp name, '' for whole-CCD metrics or 'A/B' for amp pairs; trend has
    one dict per (input exposure, amp) with its level and saturated-pixel fraction.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    core = core_frame(master, p["edge_margin"])
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

    ccd_rel = None
    if reference is not None:
        ref_core = core_frame(reference, p["edge_margin"])
        ccd_rel = {}
        for amp in master.amps:
            try:
                ccd_rel[amp.name] = reference_ratio(core, ref_core, amp, p["ref_tol"])
            except Exception:
                pass
    no_ref = "" if reference is not None else "no reference flat (first period or none given)"

    for amp in master.amps:
        npix = (amp.bbox[1] - amp.bbox[0]) * (amp.bbox[3] - amp.bbox[2])
        levels, sat = input_levels(input_meta, amp)
        for (info, _), level, n_sat in zip(input_meta, levels, sat):
            trend.append(dict(detector=master.info.detector, amp=amp.name,
                              calib_id=master.info.calib_id, exposure=info.exposure,
                              mjd=info.mjd, level_adu=level, sat_pix_frac=n_sat / npix))
        add(amp.name, "input_level_adu", lambda: robust_median(levels))
        add(amp.name, "input_level_scatter_frac",
            lambda: mad_sigma(levels) / robust_median(levels))
        add(amp.name, "n_low_count_inputs",
            lambda: int(np.sum(levels < p["low_count_adu"])) if levels.size else np.nan)
        add(amp.name, "max_level_sat_frac", lambda: np.nanmax(levels) / amp.saturation)
        add(amp.name, "max_sat_pix_frac", lambda: np.nanmax(sat) / npix)

        hp = {}

        def highpass(box):
            if box not in hp:
                hp[box] = amp_highpass(core, amp, box)
            return hp[box]

        add(amp.name, "master_median", lambda: robust_median(*master.amp_pixels(amp)))
        rms = add(amp.name, "pix_rms_frac", lambda: clipped_sigma(*highpass(p["pix_box"])))
        poisson = add(amp.name, "master_poisson_frac",
                      lambda: (1.0 / np.sqrt(np.nansum(levels) * amp.gain)
                               if np.nansum(levels) > 0 else np.nan))
        add(amp.name, "pix_rms_poisson_ratio", lambda: rms / poisson)
        add(amp.name, "input_noise_ratio",
            lambda: input_noise_ratio(frames, master, amp, p["edge_margin"], len(input_meta)),
            "" if frames else "no single input frames read")

        def low_qe():
            x, bad = highpass(p["model_box"])
            good = ~bad & np.isfinite(x)
            return float(np.mean(x[good] < p["low_qe_level"] - 1.0))

        add(amp.name, "low_qe_frac", low_qe)
        add(amp.name, "n_dust_spots", lambda: len(detect_spots(
            *highpass(p["model_box"]), p["spot_sigma"], p["spot_nsigma"], p["spot_min_area"],
            p["spot_max_area"], p["spot_max_elong"], p["spot_min_depth"])))

        def treering():
            x, bad = highpass(p["model_box"])
            s = ndimage.gaussian_filter(np.where(bad | ~np.isfinite(x), 0.0, x),
                                        p["treering_sigma"])
            return clipped_sigma(s, bad)

        add(amp.name, "treering_proxy_frac", treering)
        add(amp.name, "edge_feature_frac",
            lambda: edge_feature(master, amp, p["edge_width"], p["edge_ref"]))

        rel = (ccd_rel or {}).get(amp.name)
        add(amp.name, "ref_amp_ratio", lambda: rel[0] if rel else np.nan, no_ref)
        add(amp.name, "ref_ratio_rms", lambda: rel[1] if rel else np.nan, no_ref)
        add(amp.name, "ref_ratio_frac_out", lambda: rel[2] if rel else np.nan, no_ref)
        add(amp.name, "n_new_dust_spots", lambda: len(detect_spots(
            rel[3], rel[4], p["spot_sigma"], p["spot_nsigma"], p["spot_min_area"],
            p["spot_max_area"], p["spot_max_elong"], p["spot_min_depth"]))
            if rel else np.nan, no_ref)

    for a, b in amp_pairs(master.amps):
        name = f"{a.name}/{b.name}"

        def ratio(a=a, b=b):
            (xa, ba), (xb, bb) = boundary_strips(core, a, b, p["boundary_width"],
                                                 p["boundary_gap"])
            return robust_median(xa, ba) / robust_median(xb, bb)

        r = add(name, "amp_ratio", ratio)
        # a flat in ADU scales as 1 / gain, so A/B should equal gain_B / gain_A
        add(name, "amp_ratio_vs_gain", lambda a=a, b=b: r / (b.gain / a.gain))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        add("", "gradient_ptp_frac", lambda: gradient_ptp(
            core, p["gradient_order"], p["gradient_bin"], p["boundary_width"],
            p["boundary_gap"]))
    return results, trend
