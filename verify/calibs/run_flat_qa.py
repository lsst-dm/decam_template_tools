#!/usr/bin/env python
"""Flat QA: per-amp metrics of combined flats, flagged pass/warn/fail.

For every `flat` in --collection (default DECam/calib/flat, i.e. every certified band and
validity period), per science detector and amp: input levels, saturation and low-count
frames (from ISR metadata of the master's cpFlatIsrExp inputs), pixel-to-pixel RMS and
its Poisson expectation, single-frame noise vs Poisson, low-QE fraction, dust spots, a
tree-ring proxy, CCD edge features, the amp A/B ratio vs the camera gains, the CCD-wide
gradient, and stability against a reference flat. flat_metrics.py defines each metric;
README.md describes the outputs.

The reference for each flat is the previous validity period of the same detector and
filter in --collection, or the matching flat in --reference-collection. The photometric
closure test (--closure) is not implemented yet.

Needs the LSST stack for --repo; --fits-dir needs only numpy/scipy/astropy/pandas.
"""

import sys

import matplotlib

matplotlib.use("Agg")

from common import calio, qa  # noqa: E402
from common.stats import bin_image  # noqa: E402
from flat_metrics import DEFAULT_PARAMS, compute_flat_metrics, well_exposed  # noqa: E402

DATASET_TYPE = "flat"
DEFAULT_COLLECTION = "DECam/calib/flat"
KEY_METRICS = ["pix_rms_frac", "input_noise_ratio", "gradient_ptp_frac", "amp_ratio_vs_gain",
               "n_dust_spots", "treering_proxy_frac", "edge_feature_frac", "ref_ratio_rms",
               "n_new_dust_spots"]


def flat_params(config):
    p = dict(DEFAULT_PARAMS, input_dataset_type="cpFlatIsrExp", n_noise_frames=2,
             thumb_bin=32)
    p.update(qa.params(config, DATASET_TYPE))
    return p


def measure_flat(source, info, ref_info, p):
    """Metrics for one master flat. Returns (rows, trend rows, binned master)."""
    master = source.load(info)
    inputs = source.inputs(info, p["input_dataset_type"])
    input_meta = [(i, source.load_metadata(i)) for i in inputs]
    # sample noise frames among properly exposed inputs (dark or ramp frames say nothing
    # about the master's noise; they are counted by n_low_count_inputs instead)
    usable = well_exposed(input_meta, p["low_count_adu"]) or inputs
    frames = [source.load(i) for i in qa.spread(usable, p["n_noise_frames"])]
    reference = source.load(ref_info) if ref_info is not None else None

    results, trend = compute_flat_metrics(master, frames, input_meta, reference, p)
    rows = []
    for r in results:
        error = r["error"]
        if not inputs and r["metric"].startswith(("input_", "n_low", "max_", "master_poisson")):
            error = f"no {p['input_dataset_type']} inputs found"
        rows.append(qa.make_row(info, r["amp"], r["metric"], r["value"], r["unit"], error))
    return rows, trend, bin_image(master.image, master.bad, p["thumb_bin"])


def process_flat(task):
    """Worker entry point: (flat, reference or None). Never raises."""
    info, ref_info = task
    p = flat_params(qa.worker_config())
    try:
        rows, trend, binned = measure_flat(qa.worker_source(), info, ref_info, p)
        return qa.task_result(info, rows, trend, binned)
    except Exception as exc:
        return qa.task_result(info, error=f"{type(exc).__name__}: {exc}"[:300])


def parse_args(argv=None):
    p = qa.new_parser(__doc__)
    qa.add_common_args(p, DEFAULT_COLLECTION)
    p.add_argument("--reference-collection",
                   help="compare against flats from this collection (or FITS directory) "
                        "instead of the previous period in --collection")
    p.add_argument("--closure", action="store_true",
                   help="photometric closure test (not implemented yet; ignored with a note)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, spec, source, layout, detectors, infos = qa.prepare_run(args, DATASET_TYPE)
    if args.closure:
        print("NOTE: --closure (photometric closure test) is not implemented yet; skipped.",
              file=sys.stderr)

    if args.reference_collection:
        ref_spec = (dict(spec, fits_dir=args.reference_collection) if spec["kind"] == "fits"
                    else dict(spec, collection=args.reference_collection))
        candidates = calio.open_source(ref_spec).list_calibs(DATASET_TYPE, detectors,
                                                             args.filters)
        refs = qa.pick_references(infos, candidates, same_collection=False)
    else:
        refs = qa.pick_references(infos, infos)
    tasks = [(info, refs[qa.entry_key(info)]) for info in infos]

    results = qa.run_tasks(process_flat, tasks, spec, config, args.workers, source=source,
                           label=f"{DATASET_TYPE} calibs")
    qa.finish_run(args, DATASET_TYPE, config, spec, source, layout, infos, results,
                  flat_params(config), KEY_METRICS)


if __name__ == "__main__":
    main()
