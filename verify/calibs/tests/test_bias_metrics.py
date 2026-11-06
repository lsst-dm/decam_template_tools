"""Bias metrics recover what synthetic.py injects."""

from dataclasses import replace

import numpy as np
import pytest

import bias_metrics as bm
import synthetic as syn
from common.stats import mad_sigma


def by_metric(results):
    return {(r["amp"], r["metric"]): r["value"] for r in results}


@pytest.fixture(scope="module")
def clean():
    master, frames, meta = syn.bias_set(seed=3)
    results, trend = bm.compute_bias_metrics(master, frames, meta)
    return master, frames, meta, by_metric(results), results, trend


def test_every_metric_reported_per_amp_with_units(clean):
    *_, results, _ = clean
    assert {(r["amp"], r["metric"]) for r in results} == {
        (a, m) for a in ("A", "B") for m in bm.UNITS}
    assert all(r["error"] == "" for r in results)
    assert all(r["unit"] == bm.UNITS[r["metric"]] for r in results)


@pytest.mark.parametrize("amp", ["A", "B"])
def test_read_noise_recovered_in_electrons(clean, amp):
    *_, v, _, _ = clean
    assert v[(amp, "read_noise_e")] == pytest.approx(syn.READ_NOISE_E[amp], rel=0.03)
    assert v[(amp, "read_noise_overscan_e")] == pytest.approx(syn.READ_NOISE_E[amp], rel=1e-6)
    assert v[(amp, "single_bias_sigma_e")] == pytest.approx(syn.READ_NOISE_E[amp], rel=0.05)


@pytest.mark.parametrize("amp", ["A", "B"])
def test_overscan_level_and_stability(clean, amp):
    master, frames, meta, v, _, trend = clean
    assert v[(amp, "overscan_level_adu")] == pytest.approx(syn.OVERSCAN_ADU[amp], abs=1.0)
    assert v[(amp, "n_outlier_inputs")] == 0
    rows = [t for t in trend if t["amp"] == amp]
    assert len(rows) == 16
    assert sum(np.isfinite(t.get("frame_sigma_e", np.nan)) for t in rows) == 6

    # stability needs a long sequence: metadata only, as for a real 200-frame RUN
    rng = np.random.default_rng(0)
    key = bm.OVERSCAN_KEY.format(amp=amp)
    long_meta = [(info, {key: syn.OVERSCAN_ADU[amp] + rng.normal(0, 2.0)})
                 for info in [syn.make_info("cpBiasIsrExp", exposure=i) for i in range(400)]]
    v = by_metric(bm.compute_bias_metrics(master, frames, long_meta)[0])
    assert v[(amp, "overscan_stability_adu")] == pytest.approx(2.0, rel=0.15)


def test_outlier_input_counted_but_stability_robust():
    master, frames, meta = syn.bias_set(seed=4, outliers=(7,))
    results, trend = bm.compute_bias_metrics(master, frames, meta)
    v = by_metric(results)
    assert v[("A", "n_outlier_inputs")] == 1
    assert v[("A", "overscan_stability_adu")] < 3
    assert [t["exposure"] for t in trend if t["outlier"] and t["amp"] == "A"] == [1007]


def test_master_level_and_noise_scaling(clean):
    *_, v, _, _ = clean
    for amp in ("A", "B"):
        assert v[(amp, "bias_median_adu")] == pytest.approx(1.0, abs=0.02)
        assert v[(amp, "master_noise_ratio")] == pytest.approx(1.0, abs=0.05)
    assert bm.count_inputs(clean[0].metadata) == 16


def test_master_noise_ratio_flags_extra_noise():
    master, frames, meta = syn.bias_set(seed=5)
    rng = np.random.default_rng(0)
    noisy = replace(master, image=master.image + rng.normal(0, 1.0, master.image.shape))
    v = by_metric(bm.compute_bias_metrics(noisy, frames, meta)[0])
    assert v[("A", "master_noise_ratio")] > 2


def test_pickup_line_found_at_injected_frequency():
    master, frames, meta = syn.bias_set(seed=6, pickup=(1.0, 0.25))
    v = by_metric(bm.compute_bias_metrics(master, frames, meta)[0])
    for amp in ("A", "B"):
        assert v[(amp, "pickup_peak_freq")] == pytest.approx(0.25, abs=1.0 / 128)
        assert v[(amp, "pickup_peak_ratio")] > 3


def test_no_pickup_gives_ratio_near_one(clean):
    *_, v, _, _ = clean
    assert v[("A", "pickup_peak_ratio")] < 1.3


def test_interamp_correlation():
    master, frames, meta = syn.bias_set(seed=7, common_rows=1.0)
    v = by_metric(bm.compute_bias_metrics(master, frames, meta)[0])
    assert v[("A", "interamp_row_corr")] > 0.8
    assert v[("A", "interamp_row_corr")] == v[("B", "interamp_row_corr")]


def test_interamp_correlation_absent(clean):
    *_, v, _, _ = clean
    assert abs(v[("A", "interamp_row_corr")]) < 0.15


def test_structure_metrics_recover_gradient_and_columns():
    rng = np.random.default_rng(8)
    pattern = syn.fixed_pattern(gradient_ptp=4.0, col_sigma=0.5, rng=rng)
    master, frames, meta = syn.bias_set(seed=8, pattern=pattern)
    v = by_metric(bm.compute_bias_metrics(master, frames, meta)[0])
    # bin centres span (NY - bin) / (NY - 1) of the injected ramp
    expected = 4.0 * (syn.NY - 32) / (syn.NY - 1)
    for amp in ("A", "B"):
        assert v[(amp, "poly2d_ptp_adu")] == pytest.approx(expected, rel=0.05)
        assert v[(amp, "col_profile_rms_adu")] == pytest.approx(0.5, rel=0.2)
        # the y gradient is the row profile; its robust rms is the MAD-sigma of the ramp
        ramp = 4.0 * np.arange(syn.NY) / (syn.NY - 1)
        assert v[(amp, "row_profile_rms_adu")] == pytest.approx(mad_sigma(ramp), rel=0.05)


def test_hot_pixels_bad_columns_and_new_defects():
    # amp A is x >= 128, amp B x < 128
    hot = [(50 + 40 * i, 140 + 5 * i, 50.0) for i in range(10)]  # A
    hot += [(30 + 40 * i, 10 + 9 * i, 80.0) for i in range(10)]  # B
    pattern = syn.fixed_pattern(hot=hot, bad_cols=[(200, 5.0), (220, -5.0)])
    master, frames, meta = syn.bias_set(seed=9, pattern=pattern)
    defects = np.zeros(master.image.shape, dtype=bool)
    for y, x, _ in hot[:5]:
        defects[y, x] = True  # five of A's hot pixels are already known
    defects[:, 220] = True  # and one of the bad columns
    v = by_metric(bm.compute_bias_metrics(master, frames, meta, defects)[0])
    assert v[("A", "n_hot_pix")] == 10
    assert v[("B", "n_hot_pix")] == 10
    assert v[("A", "n_bad_cols")] == 2
    assert v[("B", "n_bad_cols")] == 0
    assert v[("A", "n_new_defect_pix")] == 5 + syn.NY
    assert v[("B", "n_new_defect_pix")] == 10


def test_masked_pixels_are_known_defects():
    pattern = syn.fixed_pattern(hot=[(100, 150, 60.0)])
    master, frames, meta = syn.bias_set(seed=10, pattern=pattern)
    master.bad[100, 150] = True
    v = by_metric(bm.compute_bias_metrics(master, frames, meta)[0])
    assert v[("A", "n_hot_pix")] == 1
    assert v[("A", "n_new_defect_pix")] == 0


def test_no_frames_gives_nan_with_a_note():
    master, _, meta = syn.bias_set(seed=11)
    results = bm.compute_bias_metrics(master, [], meta)[0]
    v = by_metric(results)
    assert np.isnan(v[("A", "read_noise_e")])
    note = next(r["error"] for r in results if r["metric"] == "read_noise_e")
    assert "no finite result" in note
    assert np.isnan(v[("A", "pickup_peak_ratio")])
    assert v[("A", "overscan_level_adu")] == pytest.approx(2300, abs=1)


def test_failing_metric_is_isolated():
    master, frames, meta = syn.bias_set(seed=12)
    amps = [replace(a, gain=None) for a in master.amps]  # gain missing -> TypeError
    results, _ = bm.compute_bias_metrics(replace(master, amps=amps), frames, meta)
    errors = {r["metric"] for r in results if r["error"]}
    assert "read_noise_e" in errors
    assert "bias_median_adu" not in errors
    assert np.isfinite(by_metric(results)[("A", "bias_median_adu")])


def test_read_noise_from_integer_raw_frames():
    """Raw ADU are integers; read noise must not be quantized (MAD would give 1.4826 * k)."""
    master, frames, meta = syn.bias_set(seed=14)
    ints = [replace(f, image=np.round(f.image + 2000.0).astype(np.float32)) for f in frames]
    v = by_metric(bm.compute_bias_metrics(master, ints, meta)[0])
    for amp in ("A", "B"):
        rn_adu = syn.READ_NOISE_E[amp] / syn.GAINS[amp]
        expected = np.sqrt(rn_adu ** 2 + 1 / 12) * syn.GAINS[amp]  # + rounding noise
        assert v[(amp, "read_noise_e")] == pytest.approx(expected, rel=0.03)


def test_raw_frames_overscan_corrected():
    rng = np.random.default_rng(13)
    ny, nx = 64, 40  # raw: B overscan | B data | A data | A overscan
    amps = [syn.AmpGeom("B", (0, ny, 4, 20), 4.0, 6.0, 6e4, overscan=(0, ny, 0, 4)),
            syn.AmpGeom("A", (0, ny, 20, 36), 4.0, 6.0, 6e4, overscan=(0, ny, 36, 40))]
    image = rng.normal(0, 1.5, (ny, nx))
    image[:, :20] += 1600
    image[:, 20:] += 2300
    frame = syn.CalibFrame(image.astype(np.float32), np.zeros((ny, nx), bool), amps, {},
                           syn.make_info("raw"), is_raw=True)
    assert bm.overscan_level(frame, frame.amp("A")) == pytest.approx(2300, abs=1)
    corrected = bm.overscan_corrected(frame)
    assert np.median(corrected.amp_pixels("B")[0]) == pytest.approx(0, abs=1)
    assert np.median(frame.amp_pixels("B")[0]) == pytest.approx(1600, abs=1)  # not modified
