#!/usr/bin/env python
"""Bias QA: per-amp metrics of combined biases, flagged pass/warn/fail.

For every `bias` in --collection (default DECam/calib/bias, i.e. every certified validity
period), per science detector and amp: overscan level and its stability over the inputs,
master level, read noise from pairs of single biases, residual structure (row/column
profiles, 2D polynomial), row-wise pickup, inter-amp row correlation, defects against
the curated bad pixel mask, and master noise vs 1/sqrt(N). bias_metrics.py defines each
metric; README.md describes the outputs.

Single frames come from the master's own RUN (cpBiasIsrExp and its ISR metadata). When
that RUN has none, raw zeros inside the validity range are used instead.

Needs the LSST stack for --repo; --fits-dir needs only numpy/scipy/astropy/pandas.
"""

import matplotlib

matplotlib.use("Agg")

from bias_metrics import DEFAULT_PARAMS, compute_bias_metrics  # noqa: E402
from common import qa  # noqa: E402
from common.stats import bin_image  # noqa: E402

DATASET_TYPE = "bias"
DEFAULT_COLLECTION = "DECam/calib/bias"
# Metrics that need single frames read in full.
FRAME_METRICS = ("read_noise_e", "single_bias_sigma_e", "pickup_peak_ratio", "pickup_peak_freq",
                 "interamp_row_corr")
KEY_METRICS = ["read_noise_e", "bias_median_adu", "overscan_stability_adu",
               "pickup_peak_ratio", "n_new_defect_pix", "master_noise_ratio"]


def bias_params(config):
    p = dict(DEFAULT_PARAMS, input_dataset_type="cpBiasIsrExp", raw_observation_type="zero",
             thumb_bin=32)
    p.update(qa.params(config, DATASET_TYPE))
    return p


def measure_bias(source, info, p):
    """Metrics for one master bias. Returns (rows, trend rows, binned master)."""
    master = source.load(info)
    inputs = source.inputs(info, p["input_dataset_type"])
    if inputs:
        input_meta = [(i, source.load_metadata(i)) for i in inputs]
    else:
        inputs = source.raw_inputs(info, p["raw_observation_type"])
        input_meta = []
    frames = [source.load(i) for i in qa.spread(inputs, p["n_noise_frames"])]
    defects = source.defects_mask(info, master.image.shape)

    results, trend = compute_bias_metrics(master, frames, input_meta, defects, p,
                                          input_type=p["input_dataset_type"])
    rows = []
    for r in results:
        error = r["error"]
        if r["metric"] == "n_new_defect_pix" and defects is None and not error:
            error = "no curated defects found; compared against mask planes only"
        if not frames and r["metric"] in FRAME_METRICS:
            error = f"no {p['input_dataset_type']} or raw inputs found"
        rows.append(qa.make_row(info, r["amp"], r["metric"], r["value"], r["unit"], error))
    return rows, trend, bin_image(master.image, master.bad, p["thumb_bin"])


def process_bias(info):
    """Worker entry point: one master bias. Never raises; failures go to `error`."""
    p = bias_params(qa.worker_config())
    try:
        rows, trend, binned = measure_bias(qa.worker_source(), info, p)
        return qa.task_result(info, rows, trend, binned)
    except Exception as exc:
        return qa.task_result(info, error=f"{type(exc).__name__}: {exc}"[:300])


def parse_args(argv=None):
    p = qa.new_parser(__doc__)
    qa.add_common_args(p, DEFAULT_COLLECTION)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, spec, source, layout, _, infos = qa.prepare_run(args, DATASET_TYPE)
    results = qa.run_tasks(process_bias, infos, spec, config, args.workers, source=source,
                           label=f"{DATASET_TYPE} calibs")
    qa.finish_run(args, DATASET_TYPE, config, spec, source, layout, infos, results,
                  bias_params(config), KEY_METRICS)


if __name__ == "__main__":
    main()
