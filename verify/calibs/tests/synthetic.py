"""Synthetic two-amp detectors with known truth, in memory or written as FITS.

Geometry is generic (two amps side by side, B on the left like DECam's assembled
frames) and deliberately small so the tests are fast.
"""

import numpy as np
import yaml

from bias_metrics import OVERSCAN_KEY, OVERSCAN_RESIDUAL_KEY
from common.calio import AmpGeom, CalibFrame, CalibInfo

NY, NX = 512, 256
GAINS = {"A": 4.0, "B": 3.5}  # e-/ADU
READ_NOISE_E = {"A": 6.0, "B": 7.0}
OVERSCAN_ADU = {"A": 2300.0, "B": 1600.0}


def make_amps(ny=NY, nx=NX, gains=GAINS, read_noise=READ_NOISE_E):
    half = nx // 2
    return [AmpGeom("A", (0, ny, half, nx), gains["A"], read_noise["A"], 60000.0),
            AmpGeom("B", (0, ny, 0, half), gains["B"], read_noise["B"], 60000.0)]


def make_info(dataset_type="bias", detector=1, **kw):
    kw.setdefault("calib_id", f"{dataset_type}-{detector}")
    return CalibInfo(collection="synthetic", dataset_type=dataset_type, detector=detector, **kw)


def noise_frame(rng, amps, ny=NY, nx=NX, pickup=None, common_rows=0.0):
    """Read noise per amp (ADU = e- / gain), plus optional pickup and common row noise.

    pickup=(amplitude ADU, cycles/pixel): a row-wise sinusoid with a random phase per
    row, as from interference during serial readout. common_rows: sigma (ADU) of a row
    offset shared by all amps.
    """
    image = np.zeros((ny, nx))
    for a in amps:
        y0, y1, x0, x1 = a.bbox
        image[a.slices] = rng.normal(0.0, a.read_noise / a.gain, size=(y1 - y0, x1 - x0))
    if pickup:
        amplitude, freq = pickup
        phase = rng.uniform(0, 2 * np.pi, size=(ny, 1))
        image += amplitude * np.sin(2 * np.pi * freq * np.arange(nx)[np.newaxis, :] + phase)
    if common_rows:
        image += rng.normal(0.0, common_rows, size=(ny, 1))
    return image


def fixed_pattern(ny=NY, nx=NX, offset=1.0, gradient_ptp=0.0, hot=(), bad_cols=(),
                  col_sigma=0.0, rng=None):
    """Bias structure common to every frame: offset, a gradient along y, hot pixels
    [(y, x, adu)], bad columns [(x, adu)] and random column offsets of col_sigma."""
    pattern = np.full((ny, nx), float(offset))
    pattern += gradient_ptp * np.arange(ny)[:, np.newaxis] / (ny - 1)
    if col_sigma:
        pattern += rng.normal(0.0, col_sigma, size=(1, nx))
    for x, adu in bad_cols:
        pattern[:, x] += adu
    for y, x, adu in hot:
        pattern[y, x] += adu
    return pattern


def bias_set(seed=0, n_inputs=16, n_frames=6, pattern=None, pickup=None, common_rows=0.0,
             overscan_sigma=1.0, outliers=(), detector=1, amps=None):
    """(master, frames, input_meta) like one cp_pipe bias RUN.

    The master is the mean of n_inputs frames, frames are the first n_frames of those,
    and input_meta carries ISR-style overscan metadata (outliers: indices whose overscan
    level is pushed up by 16000 ADU).
    """
    rng = np.random.default_rng(seed)
    amps = amps or make_amps()
    pattern = fixed_pattern() if pattern is None else pattern
    stack = [pattern + noise_frame(rng, amps, pickup=pickup, common_rows=common_rows)
             for _ in range(n_inputs)]
    master = CalibFrame(
        image=np.mean(stack, axis=0).astype(np.float32),
        bad=np.zeros((NY, NX), dtype=bool), amps=amps,
        metadata={f"LSST BUTLER INPUT {i} DATASETTYPE": "cpBiasIsrExp" for i in range(n_inputs)},
        info=make_info("bias", detector))
    infos = [make_info("cpBiasIsrExp", detector, exposure=1000 + i, mjd=57000.0 + i / 100,
                       calib_id=f"input-{detector}-{i}") for i in range(n_inputs)]
    frames = [CalibFrame(image=stack[i].astype(np.float32), bad=np.zeros((NY, NX), dtype=bool),
                         amps=amps, metadata={}, info=infos[i]) for i in range(n_frames)]
    meta = []
    for i, info in enumerate(infos):
        md = {}
        for a in amps:
            level = OVERSCAN_ADU[a.name] + rng.normal(0.0, overscan_sigma)
            md[OVERSCAN_KEY.format(amp=a.name)] = level + (16000.0 if i in outliers else 0.0)
            md[OVERSCAN_RESIDUAL_KEY.format(amp=a.name)] = a.read_noise / a.gain
        meta.append((info, md))
    return master, frames, meta


# -------------------------------------------------------------------- FITS writing


def layout_entry(detector, amps=None):
    amps = amps or make_amps()
    x0 = 70.0 * (detector - 1)
    return dict(name=f"D{detector}", type="SCIENCE", shape=[NY, NX],
                fp_corners=[[x0, 0.0], [x0 + 30.0, 0.0], [x0 + 30.0, 60.0], [x0, 60.0]],
                amps=[dict(name=a.name, bbox=list(a.bbox), gain=a.gain,
                           read_noise=a.read_noise, saturation=a.saturation) for a in amps])


def write_amp_geometry(root, detectors):
    with open(root / "amp_geometry.yaml", "w") as f:
        yaml.safe_dump({"detectors": {d: layout_entry(d) for d in detectors}}, f)


def write_exposure_fits(path, image, header=None, mask=None, mask_planes=None):
    """Stack-style Exposure FITS: empty primary + IMAGE + MASK (MP_<plane> bit numbers)."""
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    for k, v in (header or {}).items():
        primary.header[k if len(k) <= 8 else f"HIERARCH {k}"] = v
    hdus = [primary, fits.ImageHDU(np.asarray(image, dtype=np.float32), name="IMAGE")]
    mhdu = fits.ImageHDU(np.zeros(np.shape(image), dtype=np.int32) if mask is None
                         else mask.astype(np.int32), name="MASK")
    for plane, bit in (mask_planes or {"BAD": 0, "SAT": 1, "NO_DATA": 8}).items():
        key = f"MP_{plane}"
        mhdu.header[key if len(key) <= 8 else f"HIERARCH {key}"] = bit
    hdus.append(mhdu)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def write_bias_dir(root, detectors=(1, 2), seed=0, defects=None, **kw):
    """FITS layout for FitsSource: bias/, bias_inputs/, defects/, amp_geometry.yaml.

    defects: {detector: [(x0, y0, width, height)]} written as a defects table.
    Returns {detector: (master, frames, meta)} for the truth.
    """
    from astropy.io import fits

    write_amp_geometry(root, detectors)
    truth = {}
    for det in detectors:
        master, frames, meta = bias_set(seed=seed + det, detector=det, **kw)
        truth[det] = (master, frames, meta)
        write_exposure_fits(root / "bias" / f"bias_{det:02d}.fits", master.image,
                            dict(CCDNUM=det, **master.metadata))
        frame_by_exp = {f.info.exposure: f for f in frames}
        for info, md in meta:
            if info.exposure not in frame_by_exp:
                continue  # only the frames read in full are written as files
            write_exposure_fits(root / "bias_inputs" / f"in_{det:02d}_{info.exposure}.fits",
                                frame_by_exp[info.exposure].image,
                                dict(CCDNUM=det, EXPNUM=info.exposure, **{"MJD-OBS": info.mjd},
                                     **md))
        boxes = np.array((defects or {}).get(det, []), dtype=int).reshape(-1, 4)
        if len(boxes):
            cols = [fits.Column(name=n, format="J", array=boxes[:, i])
                    for i, n in enumerate(("x0", "y0", "width", "height"))]
            hdr = fits.Header({"CCDNUM": det})
            (root / "defects").mkdir(exist_ok=True)
            fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols)]
                         ).writeto(root / "defects" / f"defects_{det:02d}.fits", overwrite=True)
    return truth


# ------------------------------------------------------------------------- flats

FLAT_NY, FLAT_NX = 1024, 512
FLAT_GAINS = {"A": 4.0, "B": 3.8}
FLAT_PARAMS = dict(edge_margin=12, model_box=64, pix_box=16, gradient_bin=32,
                   spot_min_depth=0.005)


def flat_amps():
    return make_amps(FLAT_NY, FLAT_NX, FLAT_GAINS, {"A": 6.0, "B": 6.0})


def flat_response(seed=0, gradient=0.0, prnu=0.005, dust=(), low_qe=(), edge=0.0,
                  rings=0.0, ny=FLAT_NY, nx=FLAT_NX):
    """Relative response (illumination x QE), mean ~1.

    gradient: total linear change along y; dust: [(y, x, radius, depth)] donuts (rings
    of width 2 px); low_qe: [(y0, y1, x0, x1, factor)]; edge: fractional excess of the
    outer 8 px on all four sides; rings: amplitude of concentric sinusoidal 'tree rings'
    (period 24 px) centred off-chip.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    resp = 1.0 + gradient * (yy / (ny - 1) - 0.5)
    resp *= 1.0 + rng.normal(0.0, prnu, (ny, nx))
    for y, x, radius, depth in dust:
        r = np.hypot(yy - y, xx - x)
        resp *= 1.0 - depth * (np.abs(r - radius) < 1.0)
    for y0, y1, x0, x1, factor in low_qe:
        resp[y0:y1, x0:x1] *= factor
    if edge:
        e = np.zeros((ny, nx), dtype=bool)
        e[:8] = e[-8:] = True
        e[:, :8] = e[:, -8:] = True
        resp[e] *= 1.0 + edge
    if rings:
        r = np.hypot(yy + 1500, xx + 900)
        resp *= 1.0 + rings * np.sin(2 * np.pi * r / 24.0)
    return resp


def flat_frame(rng, resp, amps, level_e):
    """One ISR'd flat in ADU: Poisson (Gaussian approx.) + read noise, divided by gain."""
    e = level_e * resp
    image = e + rng.normal(0.0, 1.0, e.shape) * np.sqrt(e + 36.0)
    for a in amps:
        image[a.slices] /= a.gain
    return image


def flat_set(seed=0, n_inputs=12, n_frames=2, level_e=60000.0, levels_e=None, resp=None,
             detector=1, **resp_kw):
    """(master, frames, input_meta) like one cp_pipe flat RUN.

    The master is the mean of the inputs, each divided by its own median (so amps keep
    their 1/gain ratio). levels_e overrides the per-input electron level.
    """
    rng = np.random.default_rng(seed + 1000)
    amps = flat_amps()
    resp = flat_response(seed, **resp_kw) if resp is None else resp
    levels_e = list(levels_e) if levels_e is not None else [level_e] * n_inputs
    stack = [flat_frame(rng, resp, amps, lev) for lev in levels_e]
    master = np.mean([f / np.median(f) for f in stack], axis=0)
    shape = (FLAT_NY, FLAT_NX)
    master = CalibFrame(master.astype(np.float32), np.zeros(shape, dtype=bool), amps, {},
                        make_info("flat", detector, filter="r"))
    infos = [make_info("cpFlatIsrExp", detector, exposure=2000 + i, mjd=57100.0 + i / 50,
                       filter="r", calib_id=f"flat-in-{detector}-{i}")
             for i in range(len(stack))]
    frames = [CalibFrame(stack[i].astype(np.float32), np.zeros(shape, dtype=bool), amps, {},
                         infos[i]) for i in range(min(n_frames, len(stack)))]
    meta = []
    for info, img in zip(infos, stack):
        md = {}
        for a in amps:
            md[f"LSST ISR FINAL MEDIAN {a.name}"] = float(np.median(img[a.slices]))
            md[f"LSST ISR MASK SAT {a.name}"] = 0
            md[f"SATURAT{a.name}"] = a.saturation
        meta.append((info, md))
    return master, frames, meta


def write_flat_dir(root, periods, detector=1, seed=0):
    """FITS layout with one master per period: periods = [(validity_begin, resp_kw)].

    Inputs go to flat_inputs/ only for the first period (enough for the CLI test).
    """
    amps = flat_amps()
    with open(root / "amp_geometry.yaml", "w") as f:
        entry = layout_entry(detector, amps)
        entry["shape"] = [FLAT_NY, FLAT_NX]
        yaml.safe_dump({"detectors": {detector: entry}}, f)
    truth = {}
    for k, (begin, resp_kw) in enumerate(periods):
        master, frames, meta = flat_set(seed=seed, detector=detector, **resp_kw)
        truth[begin] = (master, frames, meta)
        write_exposure_fits(root / "flat" / f"flat_{detector:02d}_{k}.fits", master.image,
                            {"CCDNUM": detector, "FILTER": "r DECam SDSS c0002 6415.0 1480.0",
                             "CALIB_VALID_BEGIN": begin})
        if k == 0:
            frame_by_exp = {f.info.exposure: f for f in frames}
            for info, md in meta:
                if info.exposure in frame_by_exp:
                    write_exposure_fits(
                        root / "flat_inputs" / f"in_{info.exposure}.fits",
                        frame_by_exp[info.exposure].image,
                        dict(CCDNUM=detector, EXPNUM=info.exposure, FILTER="r DECam",
                             **{"MJD-OBS": info.mjd}, **md))
    return truth


# ------------------------------------------------------------------------ fringes

FRINGE_PARAMS = dict(edge_margin=12, smooth_box=128, bin=8, coherence_bin=16)


def fringe_pattern(seed=0, amplitude=2e-4, period=60.0, ny=FLAT_NY, nx=FLAT_NX):
    """Interference-like fringes: two crossed sinusoids with random phases, rms
    `amplitude` (fraction of sky)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    pattern = np.zeros((ny, nx))
    for _ in range(2):
        theta, phase = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        pattern += np.sin(2 * np.pi * (xx * np.cos(theta) + yy * np.sin(theta)) / period + phase)
    return amplitude * pattern / pattern.std()


def fringe_template(pattern, noise=2e-4, seed=0, amp_step=0.0, n_inputs=60, detector=1):
    """Template CalibFrame: pattern + white noise; amp_step is added to amp A."""
    rng = np.random.default_rng(seed)
    image = pattern + rng.normal(0.0, noise, pattern.shape)
    amps = flat_amps()
    image[amps[0].slices] += amp_step
    md = {f"LSST BUTLER INPUT {i} DATASETTYPE": "cpFringeMeasuredIsrExp"
          for i in range(n_inputs)}
    return CalibFrame(image.astype(np.float32), np.zeros(pattern.shape, dtype=bool), amps, md,
                      make_info("fringe", detector, filter="z"))


def fringe_frames(pattern, scales, skies, noise=0.004, seed=0, normalized=True, stars=20,
                  detector=1):
    """Science frames with fringes `scale` x pattern and sky level `sky` (BGMEAN).

    normalized=True gives cpFringeMeasuredIsrExp-like frames ((x - sky) / sky, stars
    masked as bad); False gives ISR'd frames in ADU (sky + sky x fringes + stars).
    """
    rng = np.random.default_rng(seed)
    amps = flat_amps()
    frames = []
    for i, (scale, sky) in enumerate(zip(scales, skies)):
        rel = scale * pattern + rng.normal(0.0, noise, pattern.shape)
        bad = np.zeros(pattern.shape, dtype=bool)
        for _ in range(stars):
            y, x = rng.integers(20, pattern.shape[0] - 20), rng.integers(20, pattern.shape[1] - 20)
            rel[y - 2:y + 3, x - 2:x + 3] += 0.5
            bad[y - 3:y + 4, x - 3:x + 4] = normalized
        image = rel if normalized else sky * (1.0 + rel)
        info = make_info("cpFringeMeasuredIsrExp" if normalized else "post_isr_image", detector,
                         exposure=3000 + i, mjd=57200.0 + i / 10, filter="z",
                         calib_id=f"sci-{detector}-{i}")
        frames.append(CalibFrame(image.astype(np.float32), bad, amps, {"BGMEAN": sky}, info))
    return frames


def write_fringe_dir(root, periods, detector=1, n_frames=6):
    """FITS layout: fringe/ templates per period, fringe_inputs/ normalized frames.

    periods = [(validity_begin, template_kw)]; every template shares one fringe pattern.
    """
    with open(root / "amp_geometry.yaml", "w") as f:
        entry = layout_entry(detector, flat_amps())
        entry["shape"] = [FLAT_NY, FLAT_NX]
        yaml.safe_dump({"detectors": {detector: entry}}, f)
    pattern = fringe_pattern()
    for k, (begin, kw) in enumerate(periods):
        t = fringe_template(pattern, seed=k + 1, detector=detector, **kw)
        write_exposure_fits(root / "fringe" / f"fringe_{detector:02d}_{k}.fits", t.image,
                            {"CCDNUM": detector, "FILTER": "z DECam SDSS c0004 9260.0 1520.0",
                             "CALIB_VALID_BEGIN": begin, **t.metadata})
    skies = np.linspace(2000.0, 16000.0, n_frames)
    for f in fringe_frames(pattern, 1800.0 / skies, skies, seed=9, detector=detector):
        mask = f.bad.astype(np.int32) << 5  # DETECTED
        write_exposure_fits(root / "fringe_inputs" / f"in_{f.info.exposure}.fits", f.image,
                            {"CCDNUM": detector, "EXPNUM": f.info.exposure, "FILTER": "z DECam",
                             "MJD-OBS": f.info.mjd, "BGMEAN": f.metadata["BGMEAN"]},
                            mask=mask, mask_planes={"BAD": 0, "SAT": 1, "DETECTED": 5})
    return pattern
