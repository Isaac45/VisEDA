"""Tests for viseda.hyperspectral.HyperspectralEDA."""
import numpy as np
import pytest

from viseda import HyperspectralEDA


@pytest.fixture
def eda():
    rng = np.random.default_rng(42)
    cube = rng.random((64, 64, 50)).astype(np.float32)
    wl = np.linspace(400, 2500, 50)
    eda = HyperspectralEDA(verbose=False, wavelengths=wl)
    eda.load(cube)
    return eda


def test_shape(eda):
    s = eda.summary()
    assert s["shape"] == {"H": 64, "W": 64, "bands": 50}


def test_band_stats_length(eda):
    s = eda.summary()
    assert len(s["band_means"]) == 50
    assert len(s["band_snr"]) == 50


def test_pca(eda):
    scores, var = eda.pca(n_components=5)
    assert scores.shape == (64 * 64, 5)
    assert len(var) == 5
    assert np.isclose(var.sum(), sum(var))  # trivial but checks types


def test_ndvi(eda):
    ndvi = eda.compute_ndvi(nir_band=40, red_band=10)
    assert ndvi.shape == (64, 64)
    assert ndvi.min() >= -1.0
    assert ndvi.max() <= 1.0


def test_not_loaded_raises():
    with pytest.raises(RuntimeError):
        HyperspectralEDA(verbose=False).summary()
