import numpy as np
import pytest

from common import stats


def test_mad_sigma_gaussian_and_mask():
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 2.0, size=200_000)
    bad = np.zeros(x.shape, dtype=bool)
    x[:1000] = 1e6  # outliers, half of them masked
    bad[:500] = True
    x[1000] = np.nan
    assert stats.mad_sigma(x, bad) == pytest.approx(2.0, rel=0.01)
    assert stats.robust_median(x, bad) == pytest.approx(5.0, abs=0.02)
    assert np.isnan(stats.mad_sigma(np.full(10, np.nan)))


def test_clipped_sigma_unquantized_on_integer_data():
    rng = np.random.default_rng(3)
    x = np.round(rng.normal(1000.0, 1.5, size=500_000))  # raw ADU: integers
    x[:200] = 6e4  # hot pixels / saturation
    assert stats.clipped_sigma(x) == pytest.approx(np.sqrt(1.5 ** 2 + 1 / 12), rel=0.01)
    assert stats.mad_sigma(x) == pytest.approx(stats.MAD_TO_SIGMA)  # quantized: MAD == 1 ADU


def test_bin_image_ignores_bad_and_all_bad_bins_are_nan():
    x = np.arange(64, dtype=float).reshape(8, 8)
    bad = np.zeros_like(x, dtype=bool)
    bad[:4, :4] = True
    binned = stats.bin_image(x, bad, 4)
    assert binned.shape == (2, 2)
    assert np.isnan(binned[0, 0])
    assert binned[1, 1] == np.median(x[4:, 4:])


def test_fit_poly2d_recovers_quadratic():
    ny, nx = 40, 30
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
    truth = 3.0 + 2.0 * yy - 1.5 * xx ** 2 + 0.5 * xx * yy
    bad = np.zeros_like(truth, dtype=bool)
    noisy = truth.copy()
    noisy[5, 5] = 1e3
    bad[5, 5] = True
    model, coeffs = stats.fit_poly2d(noisy, bad, order=2)
    np.testing.assert_allclose(model, truth, atol=1e-9)
    assert coeffs.shape == (3, 3)


def test_row_power_spectrum_peak_and_line_ratio():
    rng = np.random.default_rng(2)
    nx, f0 = 256, 0.125
    x = rng.normal(0, 1, (300, nx))
    x += 0.8 * np.sin(2 * np.pi * f0 * np.arange(nx) + rng.uniform(0, 6.3, (300, 1)))
    freq, power = stats.row_power_spectrum(x)
    assert freq[np.argmax(power)] == pytest.approx(f0)
    ratio = stats.line_ratio(power)
    assert np.nanmax(ratio) > 5
    # a smooth red spectrum has no line
    red = 1.0 / np.linspace(0.01, 0.5, 200)
    assert np.nanmax(stats.line_ratio(red)) < 1.2


def test_separable_model_follows_edge_ramp():
    ny, nx = 128, 64
    x = np.zeros((ny, nx))
    x[:, -4:] += np.array([10, 40, 90, 180])  # edge ramp like DECam amp edges
    x[60, 20] += 50  # isolated hot pixel
    resid = x - stats.separable_model(x, None, box=16)
    assert np.abs(np.delete(resid.ravel(), 60 * nx + 20)).max() < 1e-6
    assert resid[60, 20] == pytest.approx(50, abs=1)
