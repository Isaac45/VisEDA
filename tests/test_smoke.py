import subprocess
import sys

import numpy as np

from viseda import (
    HyperspectralEDA,
    ImageEDA,
    PointCloudEDA,
    TextEDA,
    VideoEDA,
)


def test_public_imports():
    assert ImageEDA
    assert VideoEDA
    assert HyperspectralEDA
    assert PointCloudEDA
    assert TextEDA


def test_imageeda_smoke():
    rng = np.random.default_rng(1)
    arrays = [
        rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        for _ in range(4)
    ]
    eda = ImageEDA(
        verbose=False,
        compute_glcm=False,
        compute_freq=False,
    )
    eda.load_arrays(arrays, labels=["a", "b", "a", "b"])
    s = eda.summary()

    assert s["inventory"]["total"] == 4
    assert s["inventory"]["valid"] == 4
    assert s["labels"]["label_distribution"] == {"a": 2, "b": 2}
    assert np.isfinite(s["quality"]["brightness"]["mean"])


def test_hyperspectraleda_smoke(tmp_path):
    rng = np.random.default_rng(2)
    cube = rng.random((12, 10, 8)).astype(np.float32)
    cube_path = tmp_path / "cube.npy"
    np.save(cube_path, cube)

    eda = HyperspectralEDA(
        verbose=False,
        wavelengths=np.linspace(400, 900, 8),
        compute_glcm=False,
        compute_pca=False,
    )
    eda.load(cube_path)
    s = eda.summary()

    assert s["inventory"]["total_cubes"] == 1
    assert s["inventory"]["valid_cubes"] == 1
    assert s["spatial"]["bands"]["mean"] == 8.0

    ndvi = eda.compute_index(0, "ndvi")
    assert ndvi.shape == cube.shape[:2]


def test_pointcloudeda_smoke():
    rng = np.random.default_rng(3)
    cloud = rng.random((250, 3)).astype(np.float32)

    eda = PointCloudEDA(
        verbose=False,
        compute_neighbors=False,
        compute_geometry=False,
    )
    eda.load_arrays([cloud], labels=["cloud"])
    s = eda.summary()

    assert s["inventory"]["total_clouds"] == 1
    assert s["inventory"]["valid_clouds"] == 1
    assert s["inventory"]["label_distribution"] == {"cloud": 1}
    assert s["inventory"]["point_count"]["mean"] == 250.0


def test_videoeda_smoke():
    rng = np.random.default_rng(4)
    video = rng.integers(
        0, 255, (6, 24, 32, 3), dtype=np.uint8
    )

    eda = VideoEDA(
        verbose=False,
        frame_sample_rate=1,
        max_frames_per_video=6,
        resize_width=32,
    )
    eda.load_arrays([video], labels=["action"], fps=12.0)
    s = eda.summary()

    assert s["inventory"]["total_videos"] == 1
    assert s["inventory"]["valid_videos"] == 1
    assert s["labels"]["label_distribution"] == {"action": 1}
    assert s["temporal"]["frame_count"]["mean"] == 6.0


def test_texteda_smoke():
    eda = TextEDA(verbose=False)
    eda.load_texts(
        [
            "Exploratory data analysis is useful.",
            "Text analysis finds useful patterns.",
        ],
        labels=["a", "b"],
    )
    s = eda.summary()

    assert s["inventory"]["total_documents"] == 2
    assert s["inventory"]["valid_documents"] == 2
    assert s["inventory"]["label_distribution"] == {"a": 1, "b": 1}
    assert s["lexical"]["dataset_unique_tokens"] > 0
    assert "useful" in eda.vocabulary()


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "viseda.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "image" in result.stdout
    assert "video" in result.stdout
    assert "hyper" in result.stdout
    assert "cloud" in result.stdout
    assert "text" in result.stdout
