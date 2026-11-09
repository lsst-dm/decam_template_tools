"""run_flat_qa.py end to end on a synthetic FITS directory."""

import json

import pandas as pd
import pytest
import yaml

import run_flat_qa
import synthetic as syn


@pytest.fixture(scope="module")
def flat_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("flats")
    syn.write_flat_dir(root, [("2016-01-01T00:00:00", {}),
                              ("2017-01-01T00:00:00", {"dust": [(300, 350, 6, 0.05)]})])
    cfg = run_flat_qa.qa.load_config()
    cfg["flat"]["params"].update(syn.FLAT_PARAMS)
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return root


def run(root, out, *extra):
    run_flat_qa.main(["--fits-dir", str(root), "--out", str(out), "--workers", "1",
                      "--config", str(root / "cfg.yaml"), *extra])
    return pd.read_parquet(out / "flat_qa.parquet")


def test_end_to_end_with_previous_period_as_reference(flat_dir, tmp_path, capsys):
    df = run(flat_dir, tmp_path, "--closure")
    assert "closure" in capsys.readouterr().err
    assert set(df["filter"]) == {"r"}
    assert df["validity_begin"].nunique() == 2
    new = df[df["metric"] == "n_new_dust_spots"].set_index(["validity_begin", "amp"])["value"]
    assert new[("2017-01-01T00:00:00", "A")] == 1
    assert new[("2017-01-01T00:00:00", "B")] == 0
    first = df[df["validity_begin"].str.startswith("2016") & df["metric"].str.startswith("ref_")]
    assert (first["flag"] == "nan").all()
    assert first["error"].str.contains("no reference flat").all()
    vs_gain = df[df["metric"] == "amp_ratio_vs_gain"]
    assert vs_gain["amp"].unique().tolist() == ["A/B"]
    assert (vs_gain["flag"] == "pass").all()

    summary = json.loads((tmp_path / "flat_qa_summary.json").read_text())
    assert summary["run"]["dataset_type"] == "flat"
    figs = {p.name for p in (tmp_path / "figures").iterdir()}
    assert any(f.startswith("flat_fp_pix_rms_frac_r_2016") for f in figs)
    assert any(f.startswith("flat_trend_level_adu") for f in figs)


def test_reference_collection(flat_dir, tmp_path):
    other = tmp_path / "ref"
    other.mkdir()
    syn.write_flat_dir(other, [("2015-01-01T00:00:00", {})], seed=7)
    df = run(flat_dir, tmp_path / "out", "--reference-collection", str(other), "--no-plots")
    rms = df[df["metric"] == "ref_ratio_rms"]
    assert rms["value"].notna().all()  # both periods now have a reference
