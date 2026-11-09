"""Flat metrics recover what synthetic.flat_set injects."""

import numpy as np
import pytest

import flat_metrics as fm
import synthetic as syn
from common.qa import entry_key, pick_references

P = syn.FLAT_PARAMS
CCD = ""


def by_metric(results):
    return {(r["amp"], r["metric"]): r["value"] for r in results}


def run(master, frames, meta, reference=None):
    return fm.compute_flat_metrics(master, frames, meta, reference, P)


@pytest.fixture(scope="module")
def clean():
    master, frames, meta = syn.flat_set(seed=1, prnu=0.005)
    results, trend = run(master, frames, meta)
    return master, frames, meta, by_metric(results), results, trend


def test_every_metric_reported_with_units(clean):
    *_, results, _ = clean
    names = {(r["amp"], r["metric"]) for r in results}
    per_amp = {m for m in fm.UNITS if m not in ("amp_ratio", "amp_ratio_vs_gain",
                                                 "gradient_ptp_frac")}
    assert {(a, m) for a in ("A", "B") for m in per_amp} <= names
    assert ("A/B", "amp_ratio") in names and (CCD, "gradient_ptp_frac") in names
    notes = {r["metric"] for r in results if r["error"]}
    assert notes == {"ref_amp_ratio", "ref_ratio_rms", "ref_ratio_frac_out", "n_new_dust_spots"}
    assert all("no reference flat" in r["error"] for r in results if r["error"])


def test_pixel_rms_and_poisson(clean):
    *_, v, _, _ = clean
    master_noise = 1 / np.sqrt(12 * 60000.0)
    for amp in ("A", "B"):
        assert v[(amp, "pix_rms_frac")] == pytest.approx(np.hypot(0.005, master_noise), rel=0.05)
        assert v[(amp, "master_poisson_frac")] == pytest.approx(master_noise, rel=0.01)
        assert v[(amp, "pix_rms_poisson_ratio")] > 3  # PRNU dominates
        assert v[(amp, "input_noise_ratio")] == pytest.approx(1.0, abs=0.03)


def test_input_levels_low_count_and_saturation():
    levels = [60000.0] * 9 + [8000.0, 8000.0, 230000.0]  # electrons
    master, frames, meta = syn.flat_set(seed=2, levels_e=levels)
    results, trend = run(master, frames, meta)
    v = by_metric(results)
    a_level = 60000.0 / syn.FLAT_GAINS["A"]
    assert v[("A", "input_level_adu")] == pytest.approx(a_level, rel=0.01)
    assert v[("A", "n_low_count_inputs")] == 2
    assert v[("A", "max_level_sat_frac")] == pytest.approx(230000 / 4.0 / 60000.0, rel=0.01)
    assert v[("A", "max_sat_pix_frac")] == 0
    assert len(trend) == 2 * 12
    usable = fm.well_exposed(meta, 5000.0)
    assert len(usable) == 9 and all(i.exposure < 2009 for i in usable)


def test_amp_ratio_follows_gains(clean):
    *_, v, _, _ = clean
    assert v[("A/B", "amp_ratio")] == pytest.approx(syn.FLAT_GAINS["B"] / syn.FLAT_GAINS["A"],
                                                    rel=0.002)
    assert v[("A/B", "amp_ratio_vs_gain")] == pytest.approx(1.0, abs=0.002)


def test_gradient_recovered_across_amps():
    master, frames, meta = syn.flat_set(seed=3, gradient=0.04, prnu=0.002)
    v = by_metric(run(master, frames, meta)[0])
    # bin centres span (ny - 2 * margin - bin) of the ny - 1 pixel ramp
    expected = 0.04 * (syn.FLAT_NY - 2 * P["edge_margin"] - P["gradient_bin"]) / (syn.FLAT_NY - 1)
    assert v[(CCD, "gradient_ptp_frac")] == pytest.approx(expected, rel=0.05)
    flat_v = by_metric(run(*syn.flat_set(seed=3, prnu=0.002))[0])
    assert flat_v[(CCD, "gradient_ptp_frac")] < 0.002


def test_dust_low_qe_and_new_dust():
    dust_a = [(300, 350, 6, 0.05), (700, 400, 5, 0.05)]
    dust_b = [(500, 100, 6, 0.05)]
    patch = [(800, 830, 60, 90, 0.8)]  # amp B
    resp_ref = syn.flat_response(4, dust=dust_a[:1] + dust_b, low_qe=patch)
    resp_new = syn.flat_response(4, dust=dust_a + dust_b, low_qe=patch)
    reference = syn.flat_set(seed=40, resp=resp_ref)[0]  # same flat, independent noise
    master, frames, meta = syn.flat_set(seed=4, resp=resp_new)
    v = by_metric(run(master, frames, meta, reference)[0])
    assert v[("A", "n_dust_spots")] == 2
    assert v[("B", "n_dust_spots")] == 2  # the donut and the low-QE patch
    core_b = (syn.FLAT_NY - 2 * P["edge_margin"]) * (syn.FLAT_NX // 2 - P["edge_margin"])
    assert v[("B", "low_qe_frac")] == pytest.approx(30 * 30 / core_b, rel=0.05)
    assert v[("A", "low_qe_frac")] == 0
    assert v[("A", "n_new_dust_spots")] == 1
    assert v[("B", "n_new_dust_spots")] == 0
    assert v[("B", "ref_ratio_rms")] < 3e-3  # two masters' noise, ~sqrt(2) x 0.0012
    assert v[("A", "ref_ratio_frac_out")] > 0


def test_reference_amp_ratio_change():
    resp = syn.flat_response(5)
    reference = syn.flat_set(seed=50, resp=resp)[0]
    changed = resp.copy()
    changed[:, syn.FLAT_NX // 2:] *= 1.01  # amp A 1% more responsive
    master, frames, meta = syn.flat_set(seed=5, resp=changed)
    v = by_metric(run(master, frames, meta, reference)[0])
    assert v[("A", "ref_amp_ratio")] / v[("B", "ref_amp_ratio")] == pytest.approx(1.01, abs=1e-3)
    assert v[("A", "ref_amp_ratio")] > 1.003 and v[("B", "ref_amp_ratio")] < 0.997


def test_edge_feature_and_tree_rings():
    master, frames, meta = syn.flat_set(seed=6, edge=0.5, rings=0.003)
    v = by_metric(run(master, frames, meta)[0])
    for amp in ("A", "B"):
        assert v[(amp, "edge_feature_frac")] == pytest.approx(0.5, abs=0.02)
    plain = by_metric(run(*syn.flat_set(seed=6))[0])
    assert v[("A", "treering_proxy_frac")] > 2.5 * plain[("A", "treering_proxy_frac")]
    assert abs(plain[("A", "edge_feature_frac")]) < 0.01


def test_no_inputs_gives_notes_not_errors():
    master, _, _ = syn.flat_set(seed=7)
    results, _ = run(master, [], [])
    v = by_metric(results)
    assert np.isnan(v[("A", "input_noise_ratio")]) and np.isnan(v[("A", "input_level_adu")])
    assert np.isfinite(v[("A", "pix_rms_frac")])
    note = next(r["error"] for r in results if r["metric"] == "input_noise_ratio")
    assert note == "no single input frames read"


def test_photometric_closure_is_a_stub():
    with pytest.raises(NotImplementedError):
        fm.photometric_closure()


def test_pick_references():
    mk = syn.make_info
    infos = [mk("flat", 1, filter="r", calib_id=f"r{y}", validity_begin=f"{y}-01-01")
             for y in (2013, 2014, 2016)]
    infos.append(mk("flat", 1, filter="g", calib_id="g2014", validity_begin="2014-01-01"))
    by_id = {i.calib_id: entry_key(i) for i in infos}
    refs = pick_references(infos, infos)
    assert refs[by_id["r2013"]] is None
    assert refs[by_id["r2014"]].calib_id == "r2013"
    assert refs[by_id["r2016"]].calib_id == "r2014"
    assert refs[by_id["g2014"]] is None  # other filters never match
    other = [mk("flat", 1, filter="r", calib_id="ref2015", validity_begin="2015-01-01")]
    refs = pick_references(infos, other, same_collection=False)
    assert refs[by_id["r2013"]].calib_id == "ref2015"  # nothing earlier: use what there is
    assert refs[by_id["r2016"]].calib_id == "ref2015"


def test_pick_references_dataset_reused_in_two_windows():
    """One flat certified for 2014 and again for 2015: two entries, and the 2015 entry is
    compared with the same dataset's 2014 window."""
    mk = syn.make_info
    infos = [mk("flat", 1, filter="r", calib_id="r2013", validity_begin="2013-01-01"),
             mk("flat", 1, filter="r", calib_id="shared", validity_begin="2014-01-01"),
             mk("flat", 1, filter="r", calib_id="shared", validity_begin="2015-01-01")]
    refs = pick_references(infos, infos)
    assert len(refs) == 3
    assert refs[entry_key(infos[1])].calib_id == "r2013"
    ref_2015 = refs[entry_key(infos[2])]
    assert (ref_2015.calib_id, ref_2015.validity_begin) == ("shared", "2014-01-01")
