"""run_bias_qa.py end to end on a synthetic FITS directory."""

import json
import sys

import pandas as pd
import pytest
import yaml

import run_bias_qa
import synthetic as syn


@pytest.fixture(scope="module")
def bias_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("fits")
    truth = syn.write_bias_dir(root, detectors=(1, 2), defects={1: [(10, 20, 3, 4)]})
    return root, truth


def run(root, out, *extra):
    run_bias_qa.main(["--fits-dir", str(root), "--out", str(out), "--workers", "1", *extra])
    return pd.read_parquet(out / "bias_qa.parquet")


def test_end_to_end_outputs(bias_dir, tmp_path):
    root, _ = bias_dir
    df = run(root, tmp_path)
    assert set(df["detector"]) == {1, 2}
    assert set(df["amp"]) == {"A", "B"}
    noted = df[df["error"] != ""]
    assert set(zip(noted["detector"], noted["metric"])) == {(2, "n_new_defect_pix")}
    assert noted["error"].str.contains("no curated defects").all()  # detector 2 has none
    rn = df[df["metric"] == "read_noise_e"].set_index(["detector", "amp"])["value"]
    for (det, amp), value in rn.items():
        assert value == pytest.approx(syn.READ_NOISE_E[amp], rel=0.03)
    assert set(df["flag"]) <= {"pass", "warn", "fail", "info"}

    summary = json.loads((tmp_path / "bias_qa_summary.json").read_text())
    assert summary["run"]["source"] == "fits"
    assert sum(summary["totals"].values()) == len(df)
    trend = pd.read_parquet(tmp_path / "bias_qa_trend.parquet")
    assert len(trend) == 2 * 2 * 6  # detectors x amps x input files
    figs = {p.name for p in (tmp_path / "figures").iterdir()}
    assert "bias_fp_read_noise_e_all.png" in figs
    assert "bias_hist_read_noise_e.png" in figs
    assert any(f.startswith("bias_thumb_") for f in figs)
    assert "lsst.daf.butler" not in sys.modules  # FITS mode never imports the stack


def test_parallel_workers_match_serial(bias_dir, tmp_path):
    root, _ = bias_dir
    serial = run(root, tmp_path / "serial", "--no-plots")
    parallel = run_bias_qa.main(["--fits-dir", str(root), "--out", str(tmp_path / "par"),
                                 "--workers", "2", "--no-plots"])
    assert parallel is None
    parallel = pd.read_parquet(tmp_path / "par" / "bias_qa.parquet")
    pd.testing.assert_frame_equal(serial, parallel)


def test_custom_config_flags_fail(bias_dir, tmp_path):
    root, _ = bias_dir
    cfg = run_bias_qa.qa.load_config()
    cfg["bias"]["metrics"]["read_noise_e"] = {"warn": [None, 6.5], "fail": [None, 6.8]}
    cfg["known_bad"] = [{"detector": 2, "amp": "B"}]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    df = run(root, tmp_path / "out", "--config", str(path), "--no-plots")
    flags = df[df["metric"] == "read_noise_e"].set_index(["detector", "amp"])["flag"]
    assert flags[(1, "A")] == "pass"  # 6 e-
    assert flags[(1, "B")] == "fail"  # 7 e-
    assert flags[(2, "B")] == "skip"


def test_build_baseline_then_rerun_passes(bias_dir, tmp_path):
    root, _ = bias_dir
    baseline = tmp_path / "baseline.yaml"
    run(root, tmp_path / "a", "--build-baseline", str(baseline), "--no-plots")
    cfg = yaml.safe_load(baseline.read_text())
    assert cfg["placeholder"] is False
    assert cfg["bias"]["per_amp"]["1"]["A"]["read_noise_e"]["reference"] == pytest.approx(
        6.0, rel=0.03)
    df = run(root, tmp_path / "b", "--config", str(baseline), "--no-plots")
    assert set(df["flag"]) <= {"pass", "info"}


def test_no_matching_calibs_exits(bias_dir, tmp_path):
    root, _ = bias_dir
    with pytest.raises(SystemExit, match="No matching bias"):
        run(root, tmp_path, "--detectors", "40")
