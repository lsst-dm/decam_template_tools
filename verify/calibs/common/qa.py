"""Config, thresholds, result tables and the worker pool shared by the run_*_qa scripts.

Result table: one row per (calib, detector, amp, metric), columns ROW_COLUMNS.
Flags: pass / warn / fail from the config intervals, `info` for metrics with no
thresholds, `nan` when the value could not be measured (see `error`), and `skip` for
amps listed under `known_bad`.

A metric's thresholds come from <dataset_type>.metrics.<metric> in the config, then
<dataset_type>.per_amp.<detector>.<amp>.<metric> overrides it key by key. `warn` and
`fail` are [lo, hi] intervals (null = unbounded); `mode` says how to read them:
absolute (the values themselves), offset (reference + lo/hi) or fraction
(reference * (1 + lo/hi)).
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import calio
from .stats import mad_sigma

CALIBS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = CALIBS_DIR / "calqa_config.yaml"

ROW_COLUMNS = [
    "collection", "dataset_type", "filter", "calib_id", "validity_begin", "validity_end",
    "detector", "amp", "metric", "value", "unit", "reference", "threshold", "flag", "error",
]
FLAGS = ("pass", "warn", "fail", "nan", "info", "skip")


# ------------------------------------------------------------------------- config


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def params(config, dataset_type):
    return dict((config.get(dataset_type) or {}).get("params") or {})


def metric_spec(config, dataset_type, metric, detector, amp):
    section = config.get(dataset_type) or {}
    spec = dict((section.get("metrics") or {}).get(metric) or {})
    per_amp = section.get("per_amp") or {}
    det = per_amp.get(str(detector)) or per_amp.get(detector) or {}
    override = (det.get(amp) or {}).get(metric)
    if override:
        spec.update(override)
    return spec


def known_bad(config, dataset_type, detector, amp):
    """Reason string if (detector, amp) is listed under known_bad, else None."""
    for entry in config.get("known_bad") or []:
        if int(entry["detector"]) != int(detector):
            continue
        if entry.get("amp") not in (None, amp):
            continue
        if entry.get("dataset_types") and dataset_type not in entry["dataset_types"]:
            continue
        return entry.get("reason", "known bad")
    return None


# ----------------------------------------------------------------------- thresholds


def _bounds(interval, spec):
    if interval is None:
        return None
    lo, hi = interval
    mode = spec.get("mode", "absolute")
    if mode == "absolute":
        return lo, hi
    ref = spec.get("reference")
    if ref is None:
        return None
    if mode == "offset":
        return (None if lo is None else ref + lo), (None if hi is None else ref + hi)
    if mode == "fraction":
        return (None if lo is None else ref * (1 + lo)), (None if hi is None else ref * (1 + hi))
    raise ValueError(f"unknown threshold mode {mode!r}")


def _outside(value, bounds):
    lo, hi = bounds
    return (lo is not None and value < lo) or (hi is not None and value > hi)


def _fmt_bounds(bounds):
    lo, hi = bounds
    return f"[{'-inf' if lo is None else f'{lo:.4g}'},{'inf' if hi is None else f'{hi:.4g}'}]"


def evaluate(value, spec):
    """(flag, reference, threshold string) for one value under one metric spec."""
    warn = _bounds(spec.get("warn"), spec)
    fail = _bounds(spec.get("fail"), spec)
    ref = spec.get("reference")
    threshold = " ".join(f"{name}:{_fmt_bounds(b)}"
                         for name, b in (("warn", warn), ("fail", fail)) if b)
    if value is None or not np.isfinite(value):
        flag = "nan"
    elif warn is None and fail is None:
        flag = "info"
    elif fail is not None and _outside(value, fail):
        flag = "fail"
    elif warn is not None and _outside(value, warn):
        flag = "warn"
    else:
        flag = "pass"
    return flag, (np.nan if ref is None else float(ref)), threshold


def make_row(info, amp, metric, value, unit, error=""):
    return dict(
        collection=info.collection, dataset_type=info.dataset_type, filter=info.filter,
        calib_id=info.calib_id, validity_begin=info.validity_begin,
        validity_end=info.validity_end, detector=info.detector, amp=amp, metric=metric,
        value=np.nan if value is None else float(value), unit=unit, error=error,
    )


def apply_thresholds(df, config):
    """Add reference / threshold / flag columns and return df in ROW_COLUMNS order."""
    refs, thresholds, flags = [], [], []
    for row in df.itertuples(index=False):
        spec = metric_spec(config, row.dataset_type, row.metric, row.detector, row.amp)
        flag, ref, threshold = evaluate(row.value, spec)
        if known_bad(config, row.dataset_type, row.detector, row.amp):
            flag = "skip"
        refs.append(ref)
        thresholds.append(threshold)
        flags.append(flag)
    df = df.assign(reference=refs, threshold=thresholds, flag=flags)
    return df[ROW_COLUMNS]


# -------------------------------------------------------------------------- outputs


NO_EPOCH = "no validity range"


def _flag_counts(df):
    """(totals {flag: n}, per metric {metric: {flag: n}}) over all FLAGS."""
    counts = df.groupby(["metric", "flag"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(FLAGS), fill_value=0)
    totals = {f: int((df["flag"] == f).sum()) for f in FLAGS}
    return totals, {m: {f: int(v) for f, v in row.items()} for m, row in counts.iterrows()}


def summarize(df, run_info=None):
    """Flag counts overall and per calibration epoch (keyed by validity start date)."""
    totals, metrics = _flag_counts(df)
    epochs = {}
    begin = df["validity_begin"].fillna("").str[:10]
    for start, g in df.groupby(begin, sort=True):
        e_totals, e_metrics = _flag_counts(g)
        end = g["validity_end"].fillna("").str[:10]
        epochs[start or NO_EPOCH] = dict(
            validity_end=end.iloc[0] if end.nunique() == 1 else sorted(end.unique()),
            n_rows=int(len(g)), totals=e_totals, metrics=e_metrics)
    fails = df[df["flag"] == "fail"].sort_values(["metric", "detector", "amp"])
    return dict(
        run=run_info or {},
        n_rows=int(len(df)),
        n_errors=int((df["error"].fillna("") != "").sum()),
        totals=totals,
        metrics=metrics,
        epochs=epochs,
        failures=fails[["calib_id", "validity_begin", "detector", "amp", "metric", "value",
                        "threshold"]].head(200).to_dict(orient="records"),
    )


def write_outputs(out_dir, prefix, df, trend=None, summary=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{prefix}_qa.parquet", out / f"{prefix}_qa.csv"]
    df.to_parquet(paths[0], index=False)
    df.to_csv(paths[1], index=False)
    if trend is not None and len(trend):
        paths.append(out / f"{prefix}_qa_trend.parquet")
        trend.to_parquet(paths[-1], index=False)
    if summary is not None:
        paths.append(out / f"{prefix}_qa_summary.json")
        with open(paths[-1], "w") as f:
            json.dump(summary, f, indent=1, default=_json_default)
    for p in paths:
        print(f"Wrote {p}")
    return paths


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


# ------------------------------------------------------------------------- baseline


def build_baseline(df, config, dataset_type, collection):
    """Copy of `config` with per_amp references/thresholds derived from `df`.

    df must hold the metrics of a known-good collection. reference is the median over
    its calibs; warn/fail are reference +- max(k * MAD-sigma, min_frac * |reference|,
    baseline_min_abs). Metrics without thresholds in `config` stay info-only.
    """
    b = config.get("baseline") or {}
    k_warn, k_fail = b.get("k_warn", 3.0), b.get("k_fail", 5.0)
    frac_warn, frac_fail = b.get("min_frac_warn", 0.1), b.get("min_frac_fail", 0.2)

    per_amp = {}
    good = df[(df["flag"] != "skip") & np.isfinite(df["value"])]
    for (det, amp, metric), g in good.groupby(["detector", "amp", "metric"]):
        spec = metric_spec(config, dataset_type, metric, det, amp)
        if spec.get("warn") is None and spec.get("fail") is None:
            continue
        values = g["value"].to_numpy()
        ref = float(np.median(values))
        sig = mad_sigma(values) if values.size > 1 else 0.0
        min_abs = float(spec.get("baseline_min_abs", 0.0))
        hw = max(k_warn * sig, frac_warn * abs(ref), min_abs)
        hf = max(k_fail * sig, frac_fail * abs(ref), 2 * min_abs, hw)
        side = spec.get("baseline_side", "both")
        warn = [None if side == "upper" else -hw, None if side == "lower" else hw]
        fail = [None if side == "upper" else -hf, None if side == "lower" else hf]
        per_amp.setdefault(str(det), {}).setdefault(str(amp), {})[metric] = dict(
            reference=ref, warn=warn, fail=fail, mode="offset")

    new = copy.deepcopy(config)
    new["placeholder"] = False
    new.setdefault("baseline_from", {})[dataset_type] = collection
    new.setdefault(dataset_type, {})["per_amp"] = per_amp
    return new


def write_baseline(new_config, path):
    with open(path, "w") as f:
        f.write("# Generated by --build-baseline; per_amp values are measured, not placeholders.\n")
        yaml.safe_dump(new_config, f, sort_keys=False)
    print(f"Wrote baseline config {path}")


# ----------------------------------------------------------------------------- CLI


def _default_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def int_list(text):
    """'1-62,64' -> [1, ..., 62, 64]."""
    out = []
    for tok in text.split(","):
        if "-" in tok:
            lo, hi = tok.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif tok:
            out.append(int(tok))
    return out


def str_list(text):
    return [t for t in text.split(",") if t]


def add_common_args(p, default_collection):
    p.add_argument("--repo", default=os.environ.get("REPO"),
                   help="Butler repo (default: $REPO)")
    p.add_argument("--collection", default=default_collection,
                   help=f"calibration collection (default: {default_collection})")
    p.add_argument("--fits-dir", help="read FITS files instead of the Butler (see calio.py)")
    p.add_argument("--detectors", type=int_list,
                   help="e.g. 1-62 or 10,11 (default: science CCDs)")
    p.add_argument("--filters", type=str_list, help="bands or physical filters, comma-separated")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="thresholds YAML (default: calqa_config.yaml)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--build-baseline", metavar="OUT_YAML",
                   help="treat --collection as known-good; write a config with per-amp "
                        "references and thresholds derived from it")
    p.add_argument("--include-wavefront", action="store_true",
                   help="also process wavefront/guide/focus detectors")
    p.add_argument("--max-calibs", type=int, help="process only the first N (smoke test)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--workers", type=int, default=_default_workers(),
                   help="worker processes (default: $SLURM_CPUS_PER_TASK or all cores)")


def new_parser(doc):
    return argparse.ArgumentParser(description=doc,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)


# ---------------------------------------------------------------------- worker pool

_worker = {}


def _init_worker(spec, config):
    _worker["source"] = calio.open_source(spec)
    _worker["config"] = config


def worker_source():
    return _worker["source"]


def worker_config():
    return _worker["config"]


def limit_threads():
    """One BLAS/OpenMP thread per worker, or N workers x M threads oversubscribes the node."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def run_tasks(fn, tasks, spec, config, workers, source=None, label="calibs"):
    """Map `fn` over `tasks` in spawned workers (each opens its own source from `spec`).

    With one worker it runs in this process, reusing `source` if given.
    """
    workers = max(1, min(workers, len(tasks)))
    print(f"{len(tasks)} {label}, {workers} workers", flush=True)
    t_start = time.time()
    every = max(1, len(tasks) // 20)

    def progress(i):
        if i % every == 0 or i == len(tasks):
            elapsed = time.time() - t_start
            print(f"...{i}/{len(tasks)} {label}, {elapsed / 60:.1f} min elapsed, "
                  f"ETA {elapsed / i * (len(tasks) - i) / 60:.1f} min", flush=True)

    results = []
    if workers == 1:
        if source is not None:
            _worker.update(source=source, config=config)
        else:
            _init_worker(spec, config)
        for i, task in enumerate(tasks, 1):
            results.append(fn(task))
            progress(i)
        return results

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(spec, config)) as pool:
        for i, result in enumerate(pool.imap(fn, tasks, chunksize=1), 1):
            results.append(result)
            progress(i)
        pool.close()
        pool.join()
    return results


# ------------------------------------------------------------------ run skeleton


def calib_label(info):
    """Short tag for one calib period: '<filter>_<validity start date>' or the run name."""
    when = info.validity_begin[:10] or Path(str(info.run)).name
    return f"{info.filter}_{when}" if info.filter else when


def spread(items, n):
    """n items evenly spaced through `items` (all of them if there are fewer)."""
    if len(items) <= n:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def _order(info):
    return (info.validity_begin, info.run)


def entry_key(info):
    """Identity of one QA entry: a dataset in one validity window (the same dataset
    certified in two windows gives two entries)."""
    return (info.calib_id, info.run, info.validity_begin)


def pick_references(infos, candidates, same_collection=True):
    """{entry_key: reference CalibInfo or None} for each calib.

    Within one collection the reference is the latest earlier period of the same
    detector and filter (none for the first); a dataset reused in a later window is
    compared with its own earlier window. From a separate reference collection it is
    the latest one starting no later than the calib, else the earliest available.
    """
    by_key = {}
    for c in candidates:
        by_key.setdefault((c.detector, c.filter), []).append(c)
    refs = {}
    for info in infos:
        options = sorted((c for c in by_key.get((info.detector, info.filter), [])
                          if entry_key(c) != entry_key(info)), key=_order)
        earlier = [c for c in options if _order(c) < _order(info)]
        if earlier:
            refs[entry_key(info)] = earlier[-1]
        elif not same_collection and options:
            refs[entry_key(info)] = options[0]
        else:
            refs[entry_key(info)] = None
    return refs


def task_result(info, rows=(), trend=(), binned=None, error=None):
    """What a worker returns for one calib; with `error`, a single 'load' row says why."""
    if error is not None:
        rows = [make_row(info, "", "load", np.nan, "", error)]
    return dict(label=calib_label(info), detector=info.detector, rows=list(rows),
                trend=list(trend), binned=binned)


def prepare_run(args, dataset_type):
    """Config, source spec, source, layout and the calibs to process for a run_*_qa main."""
    limit_threads()
    config = load_config(args.config)
    spec = calio.source_spec(args, config.get("mask_planes", calio.MASK_PLANES))
    source = calio.open_source(spec)
    layout = source.focal_plane_layout()
    detectors = calio.select_detectors(layout, args.detectors, args.include_wavefront)
    infos = source.list_calibs(dataset_type, detectors, args.filters)
    if args.max_calibs is not None:
        infos = infos[:args.max_calibs]
    if not infos:
        sys.exit(f"No matching {dataset_type} datasets in {source.collection}.")
    if config.get("placeholder"):
        print(f"NOTE: thresholds in {args.config} are placeholders; flags are indicative "
              "only until a --build-baseline config is used.", flush=True)
    return config, spec, source, layout, detectors, infos


def finish_run(args, dataset_type, config, spec, source, layout, infos, results, params,
               key_metrics):
    """Thresholds, outputs, optional baseline and figures from the worker results."""
    df = apply_thresholds(
        pd.DataFrame.from_records([r for res in results for r in res["rows"]]), config)
    trend = pd.DataFrame.from_records([t for res in results for t in res["trend"]])
    run_info = dict(dataset_type=dataset_type, collection=source.collection,
                    source=spec["kind"], config=str(args.config), n_calibs=len(infos),
                    placeholder_thresholds=bool(config.get("placeholder")), params=params)
    summary = summarize(df, run_info)
    write_outputs(args.out, dataset_type, df, trend, summary)

    if args.build_baseline:
        write_baseline(build_baseline(df, config, dataset_type, source.collection),
                       args.build_baseline)

    if not args.no_plots:
        from . import plots

        thumbs, binned = {}, {}
        for info in {calib_label(i): i for i in infos}.values():
            mosaic = source.thumbnail(info)
            if mosaic is not None:
                thumbs[calib_label(info)] = mosaic
        for res in results:
            if res["binned"] is not None and res["label"] not in thumbs:
                binned.setdefault(res["label"], {})[res["detector"]] = res["binned"]
        plots.standard_figures(df, trend, layout, args.out, dataset_type, key_metrics,
                               thumbs=thumbs, binned=binned)

    print("Flags: " + ", ".join(f"{k}={v}" for k, v in summary["totals"].items()))
    if summary["n_errors"]:
        print(f"WARNING: {summary['n_errors']} rows have errors; see the 'error' column",
              file=sys.stderr)
    return df, summary
