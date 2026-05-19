"""Tests for viseda.pointcloud.PointCloudEDA."""
import numpy as np
import pytest

from viseda import PointCloudEDA


@pytest.fixture
def eda():
    rng = np.random.default_rng(7)
    # (N, 7) → X Y Z Intensity R G B
    pts = np.hstack([
        rng.random((2000, 3)).astype(np.float32) * 100,   # XYZ
        rng.random((2000, 1)).astype(np.float32) * 255,   # Intensity
        rng.integers(0, 255, (2000, 3), dtype=np.uint8),  # RGB
    ])
    eda = PointCloudEDA(verbose=False, max_points=None)
    eda.load(pts)
    return eda


def test_point_count(eda):
    assert eda._xyz.shape == (2000, 3)


def test_summary_keys(eda):
    s = eda.summary()
    for key in ("n_points", "bounding_box", "centroid",
                "density_pts_per_unit3", "x", "y", "z"):
        assert key in s


def test_intensity_present(eda):
    s = eda.summary()
    assert s["has_intensity"]
    assert "intensity" in s


def test_rgb_present(eda):
    s = eda.summary()
    assert s["has_rgb"]
    assert len(s["rgb_mean"]) == 3


def test_subsampling():
    rng = np.random.default_rng(0)
    pts = rng.random((100_000, 3)).astype(np.float32)
    eda = PointCloudEDA(verbose=False, max_points=1000)
    eda.load(pts)
    assert len(eda._xyz) == 1000


def test_not_loaded_raises():
    with pytest.raises(RuntimeError):
        PointCloudEDA(verbose=False).summary()
