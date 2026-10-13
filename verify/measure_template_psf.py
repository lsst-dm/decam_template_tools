#!/usr/bin/env python
"""Measure template_coadd PSF quality per (tract, patch, band), in parallel.

Parallel counterpart of the measurement loop in evaluate_template_psf_quality.ipynb,
for sky areas too large to walk serially. Writes the same two CSVs the notebook
writes (template_psf_per_patch.csv, template_psf_per_tract.csv); point the
notebook's PRECOMPUTED_CSV at the first to skip its slow loop and go straight to
plotting.

Each worker is a freshly spawned process with its own Butler (never a forked copy
of a parent's open Postgres connection), reads only the mask/psf/wcs/coaddInputs
components of each coadd, and keeps its own cache of preliminary_visit_summary
tables. Patches are dispatched in contiguous (tract, band, patch)-sorted chunks so
neighbouring patches, which share most input visits, tend to land on the same
worker and hit its cache. Note each worker holds one Postgres connection.

Needs the LSST stack set up (see run-measure-template-psf.sh).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

GAUSSIAN_SIGMA_TO_FWHM = 2.3548200450309493  # 2 * sqrt(2 * ln(2))

_worker = {}


def _init_worker(repo, collection, visit_summary_dataset_type, n_samples, seed):
    from lsst.daf.butler import Butler

    _worker["butler"] = Butler(repo, collections=[collection])
    _worker["collection"] = collection
    _worker["visit_summary_dataset_type"] = visit_summary_dataset_type
    _worker["n_samples"] = n_samples
    _worker["seed"] = seed
    _worker["visit_cache"] = {}


def _get_visit_summary(visit):
    cache = _worker["visit_cache"]
    if visit not in cache:
        butler = _worker["butler"]
        vrefs = list(butler.registry.queryDatasets(
            _worker["visit_summary_dataset_type"],
            collections=[_worker["collection"]], visit=visit,
        ))
        cache[visit] = butler.get(vrefs[0]) if vrefs else None
    return cache[visit]


def _measure_template_psf(ref):
    import lsst.geom as geom

    butler = _worker["butler"]
    empty = dict(template_psf_fwhm_median=np.nan, template_psf_fwhm_std=np.nan,
                 n_psf_samples=0, coverage_frac=0.0)

    mask = butler.get(ref.makeComponentRef("mask"))
    good = (mask.array & mask.getPlaneBitMask("NO_DATA")) == 0
    good_flat = np.flatnonzero(good)
    if good_flat.size == 0:
        return empty
    coverage = good_flat.size / good.size
    empty["coverage_frac"] = coverage

    # Seeded per patch, so results don't depend on which worker gets which patch.
    data_id = ref.dataId
    rng = np.random.default_rng([
        _worker["seed"], int(data_id["tract"]), int(data_id["patch"]),
        zlib.crc32(str(data_id["band"]).encode()),
    ])
    n = min(_worker["n_samples"], good_flat.size)
    ys, xs = np.unravel_index(rng.choice(good_flat, size=n, replace=False), good.shape)
    bbox = mask.getBBox()

    psf = butler.get(ref.makeComponentRef("psf"))
    wcs = butler.get(ref.makeComponentRef("wcs"))

    fwhms = []
    for x, y in zip(xs + bbox.getMinX(), ys + bbox.getMinY()):
        point = geom.Point2D(float(x), float(y))
        try:
            shape = psf.computeShape(point)
        except Exception:
            continue  # CoaddPsf has no input coverage at this exact point
        pixel_scale = wcs.getPixelScale(point).asArcseconds()
        fwhms.append(shape.getTraceRadius() * GAUSSIAN_SIGMA_TO_FWHM * pixel_scale)

    if not fwhms:
        return empty
    return dict(
        template_psf_fwhm_median=float(np.median(fwhms)),
        template_psf_fwhm_std=float(np.std(fwhms)),
        n_psf_samples=len(fwhms),
        coverage_frac=coverage,
    )


def _measure_input_psf(ref):
    butler = _worker["butler"]
    empty = dict(n_inputs=0, input_psf_fwhm_weighted_median=np.nan,
                 input_psf_fwhm_min=np.nan, input_psf_fwhm_max=np.nan)
    try:
        coadd_inputs = butler.get(ref.makeComponentRef("coaddInputs"))
    except Exception:
        return empty

    ccds = coadd_inputs.ccds.asAstropy()
    fwhms, weights = [], []
    for row in ccds:
        summary = _get_visit_summary(int(row["visit"]))
        if summary is None:
            continue
        match = np.asarray(summary["id"]) == int(row["ccd"])  # id == detector number
        if not match.any():
            continue
        sigma = np.asarray(summary["psfSigma"])[match][0]
        pixel_scale = np.asarray(summary["pixelScale"])[match][0]
        fwhms.append(sigma * GAUSSIAN_SIGMA_TO_FWHM * pixel_scale)
        weights.append(float(row["weight"]))

    result = dict(empty)
    result["n_inputs"] = len(ccds)
    if not fwhms:
        return result

    fwhms = np.asarray(fwhms)
    weights = np.asarray(weights)
    order = np.argsort(fwhms)
    cum_weight = np.cumsum(weights[order])
    median_fwhm = fwhms[order][np.searchsorted(cum_weight, cum_weight[-1] / 2.0)]
    result.update(
        input_psf_fwhm_weighted_median=float(median_fwhm),
        input_psf_fwhm_min=float(fwhms.min()),
        input_psf_fwhm_max=float(fwhms.max()),
    )
    return result


def process_patch(ref):
    """One row for the patch; a failure is recorded in `error`, never raised."""
    row = dict(ref.dataId.mapping)
    try:
        row.update(_measure_template_psf(ref))
        row.update(_measure_input_psf(ref))
        row["error"] = ""
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def _default_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _csv_ints(text):
    return [int(t) for t in text.split(",")] if text else None


def _csv_strs(text):
    return [t for t in text.split(",")] if text else None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=os.environ.get("REPO"),
                   help="Butler repo (default: $REPO)")
    p.add_argument("--collection", required=True,
                   help="template collection, e.g. DECam/templates/S250830bp")
    p.add_argument("--coadd-dataset-type", default="template_coadd")
    p.add_argument("--visit-summary-dataset-type", default="preliminary_visit_summary")
    p.add_argument("--tracts", type=_csv_ints, default=None, help="comma-separated tract ids")
    p.add_argument("--bands", type=_csv_strs, default=None, help="comma-separated bands")
    p.add_argument("--max-patches", type=int, default=None,
                   help="process only the first N patches (smoke test)")
    p.add_argument("--n-sample-points", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=_default_workers(),
                   help="worker processes (default: $SLURM_CPUS_PER_TASK or all cores)")
    p.add_argument("--chunksize", type=int, default=8,
                   help="contiguous patches handed to a worker at a time")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()
    if not args.repo:
        p.error("--repo not given and $REPO is not set")
    return args


def main():
    args = parse_args()

    # One BLAS/OpenMP thread per worker, or N workers x M threads oversubscribes the node.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from lsst.daf.butler import Butler

    butler = Butler(args.repo, collections=[args.collection])
    refs = list(butler.registry.queryDatasets(
        args.coadd_dataset_type, collections=[args.collection]))
    del butler

    if args.tracts is not None:
        refs = [r for r in refs if r.dataId["tract"] in args.tracts]
    if args.bands is not None:
        refs = [r for r in refs if r.dataId["band"] in args.bands]
    refs.sort(key=lambda r: (r.dataId["tract"], r.dataId["band"], r.dataId["patch"]))
    if args.max_patches is not None:
        refs = refs[:args.max_patches]
    if not refs:
        sys.exit("No matching datasets found.")

    workers = max(1, min(args.workers, len(refs)))
    print(f"{len(refs)} {args.coadd_dataset_type} patches, {workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    t_start = time.time()
    rows = []
    with ctx.Pool(
        processes=workers, initializer=_init_worker,
        initargs=(args.repo, args.collection, args.visit_summary_dataset_type,
                  args.n_sample_points, args.seed),
    ) as pool:
        for i, row in enumerate(pool.imap(process_patch, refs, chunksize=args.chunksize), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(refs):
                elapsed = time.time() - t_start
                print(f"...{i}/{len(refs)} patches, {elapsed / 60:.1f} min elapsed, "
                      f"ETA {elapsed / i * (len(refs) - i) / 60:.1f} min", flush=True)

    print(f"Done: {len(rows)} patches in {(time.time() - t_start) / 60:.1f} min")

    df = pd.DataFrame.from_records(rows)
    n_err = int((df["error"] != "").sum())
    if n_err:
        print(f"WARNING: {n_err} patches failed; see the 'error' column", file=sys.stderr)
    df["template_vs_input_diff"] = (
        df["template_psf_fwhm_median"] - df["input_psf_fwhm_weighted_median"])

    tract_summary = (
        df.groupby(["tract", "band"])["template_psf_fwhm_median"]
        .agg(fwhm_median="median", fwhm_mean="mean", fwhm_std="std",
             fwhm_min="min", fwhm_max="max", n_patches="count")
        .reset_index().sort_values(["band", "tract"])
    )

    os.makedirs(args.output_dir, exist_ok=True)
    patch_csv = os.path.join(args.output_dir, "template_psf_per_patch.csv")
    tract_csv = os.path.join(args.output_dir, "template_psf_per_tract.csv")
    df.to_csv(patch_csv, index=False)
    tract_summary.to_csv(tract_csv, index=False)
    print(f"Wrote {patch_csv} ({len(df)} rows)")
    print(f"Wrote {tract_csv} ({len(tract_summary)} rows)")


if __name__ == "__main__":
    main()
