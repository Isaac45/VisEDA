"""Tests for viseda.image.ImageEDA using synthetic data."""
import numpy as np
import pytest

from viseda import ImageEDA


@pytest.fixture
def eda_with_arrays():
    """Build an ImageEDA with 10 synthetic 64×64 RGB arrays."""
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    labels = [f"class_{i % 3}" for i in range(10)]
    eda = ImageEDA(verbose=False)
    eda.load_arrays(arrays, labels=labels)
    return eda


def test_load_arrays_count(eda_with_arrays):
    assert len(eda_with_arrays._records) == 10


def test_summary_keys(eda_with_arrays):
    s = eda_with_arrays.summary()
    for key in ("total_images", "valid_images", "brightness", "contrast",
                "sharpness", "height", "width", "label_distribution"):
        assert key in s, f"Missing key: {key}"


def test_summary_counts(eda_with_arrays):
    s = eda_with_arrays.summary()
    assert s["total_images"] == 10
    assert s["valid_images"] == 10
    assert s["corrupt_images"] == 0


def test_label_distribution(eda_with_arrays):
    s = eda_with_arrays.summary()
    assert s["label_distribution"] is not None
    # 10 images / 3 classes  → each class has 3 or 4 images
    assert sum(s["label_distribution"].values()) == 10


def test_stat_fields_finite(eda_with_arrays):
    s = eda_with_arrays.summary()
    for field in ("brightness", "contrast", "height", "width"):
        assert np.isfinite(s[field]["mean"])


def test_not_loaded_raises():
    with pytest.raises(RuntimeError):
        ImageEDA(verbose=False).summary()
