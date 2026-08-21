# VisEDA — Visual Exploratory Data Analysis

VisEDA is a Python toolkit for exploratory data analysis of **image, video,
hyperspectral, point-cloud, and text/NLP datasets**. It provides numerical
summaries, dataset-level visualisations, per-sample diagnostics, duplicate or
similarity analysis where applicable, command-line workflows, and
self-contained HTML reports.

## Modules

- **ImageEDA** — spatial, pixel, quality, colour, texture, frequency, duplicate,
  class-balance, and normalisation analysis.
- **VideoEDA** — spatial, temporal, motion, blur, scene-change, colour, and
  video-similarity analysis.
- **HyperspectralEDA** — per-band statistics, SNR/noise, spectral quality,
  vegetation/water indices, PCA, false-colour, texture, and spectral-diversity analysis.
- **PointCloudEDA** — geometry, density, height, duplicate/outlier,
  nearest-neighbour, PCA shape-descriptor, and attribute analysis.
- **TextEDA** — length, vocabulary, lexical diversity, symbols, readability,
  writing scripts, duplicates, TF-IDF document distances, and label analysis.

## Installation

```bash
pip install viseda
```

Optional file-format support:

```bash
# MATLAB/ENVI/GeoTIFF hyperspectral files
pip install "viseda[hyperspectral]"

# LAS/LAZ point clouds
pip install "viseda[pointcloud]"

# All optional file-format dependencies
pip install "viseda[all]"
```

Python 3.9 or later is required.

## Quick start

### ImageEDA

```python
from viseda import ImageEDA

eda = ImageEDA(verbose=True)
eda.load("path/to/images", label_from_parent=True)

summary = eda.summary()
print(summary["inventory"])
print(summary["quality"])

eda.plot(save_path="image_dashboard.png")
eda.report("image_report.html")
```

In-memory images are supported through `load_arrays()`.

### VideoEDA

```python
from viseda import VideoEDA

eda = VideoEDA(verbose=True, frame_sample_rate=5)
eda.load("path/to/videos", label_from_parent=True)

print(eda.summary()["temporal"])
eda.plot_dataset(save_path="video_dashboard.png")
eda.report("video_report.html")
```

NumPy video arrays can be loaded with `load_arrays()`.

### HyperspectralEDA

```python
import numpy as np
from viseda import HyperspectralEDA

cube = np.random.default_rng(0).random((128, 128, 103)).astype("float32")
wavelengths = np.linspace(400, 1500, cube.shape[2])

np.save("scene.npy", cube)

eda = HyperspectralEDA(
    wavelengths=wavelengths,
    compute_glcm=False,
    compute_pca=True,
)
eda.load("scene.npy")

print(eda.summary()["spectral_quality"])
ndvi = eda.compute_index(cube_index=0, index_name="ndvi")
scores, variance_ratio = eda.pca_scores(cube_index=0, n_components=3)

eda.plot(save_path="hyper_dashboard.png")
eda.report("hyper_report.html")
```

The `hyperspectral` extra adds support for MATLAB `.mat`, ENVI, and
multi-band GeoTIFF files.

### PointCloudEDA

```python
import numpy as np
from viseda import PointCloudEDA

points = np.random.default_rng(0).random((10000, 3)).astype("float32")

eda = PointCloudEDA(
    max_points_per_cloud=200000,
    compute_neighbors=True,
    compute_geometry=True,
)
eda.load_arrays([points], labels=["sample"])

print(eda.summary()["geometry"])
eda.plot_dataset(save_path="pointcloud_dashboard.png")
eda.report("pointcloud_report.html")
```

The `pointcloud` extra adds LAS/LAZ support. NPY, NPZ, TXT, CSV, XYZ, PTS,
and ASCII PLY are supported by the base installation.

### TextEDA

```python
from viseda import TextEDA

eda = TextEDA()
eda.load_texts(
    [
        "Exploratory data analysis is useful before model training.",
        "TextEDA summarises vocabulary, length, readability and duplicates.",
    ],
    labels=["eda", "eda"],
)

print(eda.summary()["lexical"])
print(eda.vocabulary(top_n=10))

eda.plot_dataset(save_path="text_dashboard.png")
eda.report("text_report.html")
```

TextEDA also loads TXT, Markdown, HTML, CSV, TSV, JSON, JSONL and NDJSON
datasets through `load()`.

## Command line

The package installs the `viseda` command:

```bash
viseda --help
```

Available subcommands are:

```text
viseda image ...
viseda hyper ...
viseda cloud ...
viseda video ...
viseda text ...
```

Examples:

```bash
viseda image "C:/datasets/images" --label-from-parent --plot --report image_report.html

viseda video "C:/datasets/videos" --label-from-parent --plot --report video_report.html

viseda hyper "C:/datasets/hyper" --label-from-parent --plot --dataset-plot \
  --report hyper_report.html

viseda cloud "C:/datasets/pointclouds" --label-from-parent --plot \
  --report pointcloud_report.html

viseda text "C:/datasets/text" --label-from-parent --plot \
  --report text_report.html
```

Use `viseda <subcommand> --help` for the options of a particular modality.

## Supported input formats

| Module | Main file inputs |
|---|---|
| ImageEDA | JPG/JPEG, PNG, BMP, TIFF and other formats accepted by the current image loader |
| VideoEDA | MP4, AVI, MOV, MKV, WEBM, MPEG/MPG, M4V |
| HyperspectralEDA | MAT, NPY, NPZ, ENVI HDR/BIL/BIP/BSQ/ENVI, TIF/TIFF |
| PointCloudEDA | NPY, NPZ, TXT, CSV, XYZ, PTS, ASCII PLY, LAS/LAZ |
| TextEDA | TXT/TEXT, MD, RST, LOG, HTML/HTM, CSV/TSV, JSON, JSONL/NDJSON |

Some file formats require the optional extras described above.


## License

MIT. See `LICENSE`.
