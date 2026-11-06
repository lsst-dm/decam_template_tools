"""Fringe metrics recover what synthetic.fringe_* injects."""

import numpy as np
import pytest

import fringe_metrics as fm
import synthetic as syn

P = syn.FRINGE_PARAMS
SKIES = np.linspace(2000.0, 16000.0, 10)
TRUE_SCALES = 1800.0 / SKIES  # fringe fraction falls as the continuum sky rises


def by_metric(results):
    return {(r["amp"], r["metric"]): r["value"] for r in results}


@pytest.fixture(scope="module")
def pattern():
    return syn.fringe_pattern(seed=0, amplitude=2e-4)


@pytest.fixture(scope="module")
def fitted(pattern):
    template = syn.fringe_template(pattern, noise=2e-4, seed=1)
    frames = syn.fringe_frames(pattern, TRUE_SCALES, SKIES, seed=2)
    results, trend = fm.compute_fringe_metrics(template, frames, [], None, P)
    return template, frames, by_metric(results), results, trend


def test_every_metric_reported_with_units(fitted):
    *_, results, _ = fitted
    names = {r["metric"] for r in results}
    assert names == set(fm.UNITS)
    assert all(r["unit"] == fm.UNITS[r["metric"]] for r in results)
    notes = {r["metric"] for r in results if r["error"]}
    assert notes == {"ref_template_corr", "ref_template_coherence"}


def test_template_amplitude_noise_and_snr(fitted):
    _, _, v, _, _ = fitted
    for amp in ("A", "B"):
        assert v[(amp, "fringe_rms_frac")] == pytest.approx(2e-4, rel=0.1)
        assert v[(amp, "template_noise_frac")] == pytest.approx(2e-4, rel=0.05)
        assert v[(amp, "template_snr")] == pytest.approx(2e-4 / (2e-4 / P["bin"]), rel=0.15)
        assert v[(amp, "fringe_ptp_frac")] > 3 * v[(amp, "fringe_rms_frac")]
    assert v[("", "n_template_inputs")] == 60


def test_noise_only_template_has_no_fringe(pattern):
    template = syn.fringe_template(0 * pattern, noise=2e-4, seed=3)
    v = by_metric(fm.compute_fringe_metrics(template, [], [], None, P)[0])
    assert v[("A", "fringe_rms_frac")] < 2e-5  # noise is subtracted, not counted
    assert v[("A", "template_snr")] < 0.8


def test_amp_step(pattern):
    template = syn.fringe_template(pattern, seed=4, amp_step=3e-4)
    v = by_metric(fm.compute_fringe_metrics(template, [], [], None, P)[0])
    assert v[("A/B", "amp_step_frac")] == pytest.approx(3e-4, abs=3e-5)
    # the per-amp high-pass keeps the step out of the fringe amplitude
    assert v[("A", "fringe_rms_frac")] == pytest.approx(2e-4, rel=0.1)


def test_scales_recovered_and_fringes_removed(fitted):
    *_, v, _, trend = fitted
    rows = sorted((t for t in trend if "scale" in t), key=lambda t: t["exposure"])
    assert len(rows) == len(SKIES)
    attenuation = 1 / (1 + (1 / P["bin"]) ** 2)  # template S/N per bin = bin here
    for row, truth in zip(rows, TRUE_SCALES):
        assert row["scale"] == pytest.approx(truth * attenuation, abs=4 * row["scale_err"])
        assert row["sky_adu"] in SKIES
    assert v[("", "n_science")] == len(SKIES)
    assert v[("", "fringe_removed_frac")] > 0.8
    assert v[("", "residual_excess_frac")] < 5e-5
    assert v[("", "scale_sky_corr")] < -0.7
    assert v[("", "n_scale_outliers")] == 0
    assert v[("", "scale_time_jump_sigma")] < 3


def test_scale_outlier_flagged(pattern):
    template = syn.fringe_template(pattern, seed=5)
    scales = TRUE_SCALES.copy()
    scales[4] *= 4
    frames = syn.fringe_frames(pattern, scales, SKIES, seed=6)
    v = by_metric(fm.compute_fringe_metrics(template, frames, [], None, P)[0])
    assert v[("", "n_scale_outliers")] == 1


def test_unnormalized_science_frames(pattern):
    template = syn.fringe_template(pattern, seed=7)
    frames = syn.fringe_frames(pattern, [0.9, 0.9], [2000.0, 2000.0], seed=8, normalized=False)
    trend = fm.compute_fringe_metrics(template, frames, [], None, P)[1]
    for row in (t for t in trend if "scale" in t):
        assert row["scale"] == pytest.approx(0.9, abs=max(0.1, 4 * row["scale_err"]))


def test_coherence_same_vs_unrelated_pattern(pattern):
    a = syn.fringe_template(pattern, seed=10)
    b = syn.fringe_template(pattern, seed=11)
    other = syn.fringe_template(syn.fringe_pattern(seed=99), seed=12)
    v = by_metric(fm.compute_fringe_metrics(a, [], [], b, P)[0])
    assert v[("", "ref_template_coherence")] == pytest.approx(1.0, abs=0.1)
    assert v[("", "ref_template_corr")] < v[("", "ref_template_coherence")]
    v = by_metric(fm.compute_fringe_metrics(a, [], [], other, P)[0])
    assert abs(v[("", "ref_template_coherence")]) < 0.2


def test_no_frames_gives_notes(pattern):
    template = syn.fringe_template(pattern, seed=13)
    results, trend = fm.compute_fringe_metrics(template, [], [], None, P)
    v = by_metric(results)
    assert v[("", "n_science")] == 0
    note = next(r["error"] for r in results if r["metric"] == "scale_median")
    assert note == "no science frames fitted"
    assert trend == []


def test_scale_sky_z():
    sky = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    scale = 0.1 + 1.5 / sky
    err = np.full(sky.shape, 0.01)
    np.testing.assert_allclose(fm.scale_sky_z(scale, err, sky), 0, atol=1e-9)
    scale[2] += 0.5
    z = fm.scale_sky_z(scale, err, sky, frac=0.1)
    assert np.sum(np.abs(z) > 5) == 1 and np.argmax(np.abs(z)) == 2
    # bimodal sky: dark frames with ~10% OH variability are not outliers
    sky = np.array([2000.0] * 4 + [18000.0] * 8)
    rng = np.random.default_rng(0)
    scale = 4800.0 / sky * (1 + 0.06 * rng.standard_normal(12)) + 0.008 * rng.standard_normal(12)
    err = np.where(sky < 5000, 0.03, 0.008)
    assert np.sum(np.abs(fm.scale_sky_z(scale, err, sky, frac=0.1)) > 5) == 0
