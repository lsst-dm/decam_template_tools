"""run_fringe_qa.py end to end on a synthetic FITS directory."""

import json

import pandas as pd
import pytest
import yaml

import run_fringe_qa
import synthetic as syn


@pytest.fixture(scope="module")
def fringe_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("fringes")
    syn.write_fringe_dir(root, [("2016-01-01T00:00:00", {}),
                                ("2017-01-01T00:00:00", {"amp_step": 3e-4})])
    cfg = run_fringe_qa.qa.load_config()
    cfg["fringe"]["params"].update(syn.FRINGE_PARAMS)
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return root


def run(root, out, *extra):
    run_fringe_qa.main(["--fits-dir", str(root), "--out", str(out), "--workers", "1",
                        "--config", str(root / "cfg.yaml"), *extra])
    return pd.read_parquet(out / "fringe_qa.parquet")


def test_end_to_end(fringe_dir, tmp_path):
    df = run(fringe_dir, tmp_path)
    assert set(df["filter"]) == {"z"}
    by = df.set_index(["validity_begin", "amp", "metric"])["value"]
    assert by[("2017-01-01T00:00:00", "A/B", "amp_step_frac")] == pytest.approx(3e-4, abs=3e-5)
    assert by[("2016-01-01T00:00:00", "", "n_science")] == 6
    assert by[("2017-01-01T00:00:00", "", "ref_template_coherence")] == pytest.approx(1, abs=0.15)
    flags = df.set_index(["validity_begin", "amp", "metric"])["flag"]
    assert flags[("2017-01-01T00:00:00", "A/B", "amp_step_frac")] == "warn"
    assert flags[("2016-01-01T00:00:00", "A/B", "amp_step_frac")] == "pass"
    first = df[df["validity_begin"].str.startswith("2016") & df["metric"].str.startswith("ref_")]
    assert first["error"].str.contains("no reference template").all()

    summary = json.loads((tmp_path / "fringe_qa_summary.json").read_text())
    assert summary["run"]["dataset_type"] == "fringe"
    trend = pd.read_parquet(tmp_path / "fringe_qa_trend.parquet")
    assert trend["scale"].notna().sum() == 2 * 6
    figs = {p.name for p in (tmp_path / "figures").iterdir()}
    assert any(f.startswith("fringe_fp_fringe_rms_frac_z_2016") for f in figs)


def test_science_collection(fringe_dir, tmp_path):
    sci = tmp_path / "sci"
    sci.mkdir()
    pattern = syn.fringe_pattern()
    for f in syn.fringe_frames(pattern, [0.9, 0.9, 0.9], [2000.0] * 3, seed=21,
                               normalized=False):
        syn.write_exposure_fits(sci / f"sci_{f.info.exposure}.fits", f.image,
                                {"CCDNUM": 1, "EXPNUM": f.info.exposure, "FILTER": "z DECam",
                                 "BGMEAN": 2000.0})
    df = run(fringe_dir, tmp_path / "out", "--science-collection", str(sci), "--no-plots")
    n = df[df["metric"] == "n_science"]["value"]
    assert (n == 3).all()
    scale = df[df["metric"] == "scale_median"]["value"]
    assert scale.between(0.7, 1.0).all()
