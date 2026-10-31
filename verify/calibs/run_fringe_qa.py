#!/usr/bin/env python
"""Fringe QA: metrics of fringe templates, flagged pass/warn/fail.

For every `fringe` in --collection (default DECam/calib/fringe, i.e. every certified
validity period; z only for DECam), per science detector: fringe amplitude and template
noise per amp, the template's amp-boundary step, coherence with the previous period's
template (independent inputs), and a fit of the template to science exposures: scale
per exposure, residual RMS, fraction of fringe power removed, scale vs sky level, scale
outliers and smoothness in time. fringe_metrics.py defines each metric; README.md
describes the outputs.

The template is fitted to `n_science` exposures it was not built from: by default the
inputs (cpFringeMeasuredIsrExp) of the nearest other validity period of the same
detector, or, with --science-collection, held-out exposures of --science-dataset-type
inside the validity range. A template's own inputs would bias the fit (see
fringe_metrics.py).

Needs the LSST stack for --repo; --fits-dir needs only numpy/scipy/astropy/pandas.
"""

import matplotlib

matplotlib.use("Agg")

from common import calio, qa  # noqa: E402
from common.stats import bin_image  # noqa: E402
from fringe_metrics import DEFAULT_PARAMS, compute_fringe_metrics  # noqa: E402

DATASET_TYPE = "fringe"
DEFAULT_COLLECTION = "DECam/calib/fringe"
KEY_METRICS = ["fringe_rms_frac", "template_snr", "amp_step_frac", "ref_template_coherence",
               "scale_median", "fringe_removed_frac", "residual_excess_frac"]
FRAME_MASK_PLANES = ["BAD", "SAT", "NO_DATA", "SUSPECT", "UNMASKEDNAN", "DETECTED", "INTRP",
                     "EDGE"]


def fringe_params(config):
    p = dict(DEFAULT_PARAMS, input_dataset_type="cpFringeMeasuredIsrExp", n_science=12,
             frame_mask_planes=FRAME_MASK_PLANES, thumb_bin=32)
    p.update(qa.params(config, DATASET_TYPE))
    return p


def measure_fringe(source, info, ref_info, p, science=None, fit_info=None):
    """Metrics for one fringe template. science = (collection, dataset_type) for held-out
    frames; otherwise the inputs of fit_info (another period) are fitted.
    Returns (rows, trend rows, binned template)."""
    master = source.load(info)
    inputs = source.inputs(info, p["input_dataset_type"])
    frame_meta = [(i, source.load_metadata(i)) for i in inputs]
    if science:
        candidates = source.frames_in(info, science[1], science[0])
    elif fit_info is not None:
        candidates = source.inputs(fit_info, p["input_dataset_type"])
    else:
        candidates = []
    frames = [source.load(i, mask_planes=p["frame_mask_planes"])
              for i in qa.spread(candidates, p["n_science"])]
    reference = source.load(ref_info) if ref_info is not None else None

    results, trend = compute_fringe_metrics(master, frames, frame_meta, reference, p,
                                            input_type=p["input_dataset_type"])
    rows = []
    for r in results:
        error = r["error"]
        if not candidates and r["metric"] in ("scale_median", "residual_rms_frac"):
            what = (f"{science[1]} in {science[0]}" if science
                    else f"{p['input_dataset_type']} of another period")
            error = f"no {what} frames found"
        rows.append(qa.make_row(info, r["amp"], r["metric"], r["value"], r["unit"], error))
    return rows, trend, bin_image(master.image, master.bad, p["thumb_bin"])


def process_fringe(task):
    """Worker entry point: (template, reference, science, period to fit). Never raises."""
    info, ref_info, science, fit_info = task
    p = fringe_params(qa.worker_config())
    try:
        rows, trend, binned = measure_fringe(qa.worker_source(), info, ref_info, p, science,
                                             fit_info)
        return qa.task_result(info, rows, trend, binned)
    except Exception as exc:
        return qa.task_result(info, error=f"{type(exc).__name__}: {exc}"[:300])


def parse_args(argv=None):
    p = qa.new_parser(__doc__)
    qa.add_common_args(p, DEFAULT_COLLECTION)
    p.add_argument("--reference-collection",
                   help="compare against templates from this collection (or FITS directory) "
                        "instead of the previous period in --collection")
    p.add_argument("--science-collection",
                   help="fit held-out science exposures from this collection (or FITS "
                        "directory) instead of the template's own inputs")
    p.add_argument("--science-dataset-type", default="post_isr_image",
                   help="dataset type of --science-collection frames (default: post_isr_image)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, spec, source, layout, detectors, infos = qa.prepare_run(args, DATASET_TYPE)

    if args.reference_collection:
        ref_spec = (dict(spec, fits_dir=args.reference_collection) if spec["kind"] == "fits"
                    else dict(spec, collection=args.reference_collection))
        candidates = calio.open_source(ref_spec).list_calibs(DATASET_TYPE, detectors,
                                                             args.filters)
        refs = qa.pick_references(infos, candidates, same_collection=False)
    else:
        refs = qa.pick_references(infos, infos)
    science = ((args.science_collection, args.science_dataset_type)
               if args.science_collection else None)
    # nearest other period (earlier, else the next one): its inputs are disjoint exposures
    others = qa.pick_references(infos, infos, same_collection=False)
    tasks = [(info, refs[qa.entry_key(info)], science, others[qa.entry_key(info)])
             for info in infos]

    results = qa.run_tasks(process_fringe, tasks, spec, config, args.workers, source=source,
                           label=f"{DATASET_TYPE} calibs")
    qa.finish_run(args, DATASET_TYPE, config, spec, source, layout, infos, results,
                  fringe_params(config), KEY_METRICS)


if __name__ == "__main__":
    main()
