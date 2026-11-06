"""Thresholds, flags, summary, outputs and baselines."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import synthetic as syn
from common import qa


@pytest.mark.parametrize("value, flag", [
    (6.0, "pass"), (9.5, "warn"), (3.5, "warn"), (13.0, "fail"), (1.0, "fail"),
    (np.nan, "nan"), (None, "nan"),
])
def test_evaluate_absolute(value, flag):
    spec = {"reference": 6.5, "warn": [4, 9], "fail": [2, 12]}
    got, ref, threshold = qa.evaluate(value, spec)
    assert got == flag
    assert ref == 6.5
    assert threshold == "warn:[4,9] fail:[2,12]"


def test_evaluate_one_sided_and_info():
    assert qa.evaluate(1e6, {"warn": [None, 5], "fail": [None, 20]})[0] == "fail"
    assert qa.evaluate(-1e6, {"warn": [None, 5], "fail": [None, 20]})[0] == "pass"
    assert qa.evaluate(3.0, {})[0] == "info"
    assert qa.evaluate(3.0, {})[2] == ""


def test_evaluate_offset_and_fraction_modes():
    offset = {"reference": 10.0, "warn": [-1, 1], "fail": [-3, 3], "mode": "offset"}
    assert [qa.evaluate(v, offset)[0] for v in (10.5, 11.5, 13.5)] == ["pass", "warn", "fail"]
    frac = {"reference": 10.0, "warn": [-0.1, 0.1], "fail": [-0.5, 0.5], "mode": "fraction"}
    assert [qa.evaluate(v, frac)[0] for v in (10.5, 8.5, 4.0)] == ["pass", "warn", "fail"]
    # relative modes without a reference cannot be evaluated: info, not a false pass
    assert qa.evaluate(1.0, {"warn": [-1, 1], "mode": "offset"})[0] == "info"
    with pytest.raises(ValueError):
        qa.evaluate(1.0, {"reference": 1.0, "warn": [0, 1], "mode": "bogus"})


CONFIG = {
    "known_bad": [{"detector": 61}, {"detector": 31, "amp": "B", "dataset_types": ["flat"]}],
    "bias": {
        "metrics": {"read_noise_e": {"reference": 6.5, "warn": [4, 9], "fail": [2, 12]},
                    "pickup_peak_freq": {}},
        "per_amp": {"5": {"A": {"read_noise_e": {"warn": [4, 7]}}}},
    },
}


def test_metric_spec_per_amp_override_merges():
    spec = qa.metric_spec(CONFIG, "bias", "read_noise_e", 5, "A")
    assert spec == {"reference": 6.5, "warn": [4, 7], "fail": [2, 12]}
    assert qa.metric_spec(CONFIG, "bias", "read_noise_e", 5, "B")["warn"] == [4, 9]


def test_known_bad():
    assert qa.known_bad(CONFIG, "bias", 61, "A")
    assert not qa.known_bad(CONFIG, "bias", 31, "B")
    assert qa.known_bad(CONFIG, "flat", 31, "B")
    assert not qa.known_bad(CONFIG, "flat", 31, "A")


def table(values):
    info = syn.make_info("bias", 1)
    rows = []
    for det, amp, metric, value in values:
        info.detector = det
        rows.append(qa.make_row(info, amp, metric, value, "electron"))
    return pd.DataFrame.from_records(rows)


def test_apply_thresholds_and_summary():
    df = qa.apply_thresholds(table([
        (1, "A", "read_noise_e", 6.0), (1, "B", "read_noise_e", 10.0),
        (5, "A", "read_noise_e", 8.0), (2, "A", "read_noise_e", 20.0),
        (61, "A", "read_noise_e", 20.0), (1, "A", "pickup_peak_freq", 0.2),
        (1, "A", "read_noise_e", np.nan),
    ]), CONFIG)
    assert list(df.columns) == qa.ROW_COLUMNS
    assert df["flag"].tolist() == ["pass", "warn", "warn", "fail", "skip", "info", "nan"]
    s = qa.summarize(df, {"collection": "synthetic"})
    assert s["totals"] == {"pass": 1, "warn": 2, "fail": 1, "nan": 1, "info": 1, "skip": 1}
    assert s["metrics"]["read_noise_e"]["warn"] == 2
    assert [f["detector"] for f in s["failures"]] == [2]


def test_write_outputs(tmp_path):
    df = qa.apply_thresholds(table([(1, "A", "read_noise_e", 6.0)]), CONFIG)
    trend = pd.DataFrame({"detector": [1], "amp": ["A"], "exposure": [3], "overscan_adu": [2.0]})
    paths = qa.write_outputs(tmp_path / "out", "bias", df, trend, qa.summarize(df))
    assert {p.name for p in paths} == {"bias_qa.parquet", "bias_qa.csv",
                                       "bias_qa_trend.parquet", "bias_qa_summary.json"}
    pd.testing.assert_frame_equal(pd.read_parquet(paths[0]), df)
    assert json.loads((tmp_path / "out" / "bias_qa_summary.json").read_text())["n_rows"] == 1


def test_build_baseline_round_trip(tmp_path):
    cfg = dict(CONFIG, baseline={"k_warn": 3, "k_fail": 5, "min_frac_warn": 0.05,
                                 "min_frac_fail": 0.1})
    rng = np.random.default_rng(0)
    rows = [(d, a, "read_noise_e", 6.0 + 0.1 * d + rng.normal(0, 0.05))
            for _ in range(5) for d in (1, 2) for a in ("A", "B")]
    rows += [(1, "A", "pickup_peak_freq", 0.2)]
    df = qa.apply_thresholds(table(rows), cfg)
    new = qa.build_baseline(df, cfg, "bias", "known/good")
    assert new["placeholder"] is False
    assert new["baseline_from"] == {"bias": "known/good"}
    spec = new["bias"]["per_amp"]["2"]["B"]["read_noise_e"]
    assert spec["mode"] == "offset"
    assert spec["reference"] == pytest.approx(6.2, abs=0.05)
    assert "pickup_peak_freq" not in new["bias"]["per_amp"]["1"]["A"]  # info stays info
    assert "per_amp" not in CONFIG["bias"] or CONFIG["bias"]["per_amp"]["5"]  # input untouched

    path = tmp_path / "baseline.yaml"
    qa.write_baseline(new, path)
    reread = yaml.safe_load(path.read_text())
    again = qa.apply_thresholds(df[qa.ROW_COLUMNS[:11] + ["error"]], reread)
    assert set(again["flag"]) <= {"pass", "info"}
    shifted = table([(1, "A", "read_noise_e", 9.0)])
    assert qa.apply_thresholds(shifted, reread)["flag"].iloc[0] == "fail"


def test_build_baseline_upper_side_and_min_abs():
    cfg = {"bias": {"metrics": {"n_hot_pix": {"warn": [None, 5], "fail": [None, 50],
                                              "baseline_side": "upper",
                                              "baseline_min_abs": 10}}}}
    df = qa.apply_thresholds(table([(1, "A", "n_hot_pix", 0.0)] * 3), cfg)
    spec = qa.build_baseline(df, cfg, "bias", "c")["bias"]["per_amp"]["1"]["A"]["n_hot_pix"]
    assert spec["warn"] == [None, 10] and spec["fail"] == [None, 20]


def test_int_list():
    assert qa.int_list("1-3,7") == [1, 2, 3, 7]


def test_summary_counts_per_epoch():
    rows = []
    for begin, end, values in (("2016-01-01T00:00:00", "2017-01-01T00:00:00", (6.0, 10.0)),
                               ("2017-01-01T00:00:00", "2018-01-01T00:00:00", (6.0, 20.0, 6.1)),
                               ("", "", (6.0,))):
        info = syn.make_info("bias", 1, validity_begin=begin, validity_end=end)
        rows += [qa.make_row(info, "A", "read_noise_e", v, "electron") for v in values]
    df = qa.apply_thresholds(pd.DataFrame.from_records(rows), CONFIG)
    epochs = qa.summarize(df)["epochs"]
    assert list(epochs) == [qa.NO_EPOCH, "2016-01-01", "2017-01-01"]
    assert epochs["2016-01-01"]["totals"]["pass"] == 1 and epochs["2016-01-01"]["totals"]["warn"] == 1
    assert epochs["2017-01-01"]["metrics"]["read_noise_e"] == {
        "pass": 2, "warn": 0, "fail": 1, "nan": 0, "info": 0, "skip": 0}
    assert epochs["2017-01-01"]["validity_end"] == "2018-01-01"
    assert epochs[qa.NO_EPOCH]["n_rows"] == 1
    assert sum(e["n_rows"] for e in epochs.values()) == len(df)
    json.dumps(qa.summarize(df))  # stays serializable
