# 🔬 VisEDA — Visual Exploratory Data Analysis

[![PyPI version](https://badge.fury.io/py/viseda.svg)](https://pypi.org/project/viseda/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/your-org/viseda/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/viseda/actions)

> **The missing EDA toolkit for non-tabular data.**
>
> While pandas-profiling and ydata-profiling cover tabular data beautifully,
> researchers working with **images**, **hyperspectral cubes**, and **point clouds**
> have had no equivalent. VisEDA fills that gap.

---

## ✨ Features

### 🖼 ImageEDA
| Analysis | Details |
|---|---|
| Dataset inventory | Count, corrupt detection, file size distribution |
| Spatial stats | Height, width, aspect ratio distributions |
| Pixel statistics | Brightness, contrast (std), sharpness (Laplacian variance), entropy |
| Channel analysis | Per-channel mean/std histograms, HSV hue distribution |
| Colour palette | Dominant colours via K-Means |
| Duplicate detection | Perceptual hash (pHash) near-duplicate grouping |
| Label distribution | From folder names or a user-supplied dict |
| Visualisation | Dark-theme EDA dashboard, sample grid, channel correlation matrix |
| Reports | Self-contained HTML report |

### 🌈 HyperspectralEDA
- Per-band mean, std, SNR, noise (MAD), saturation fraction
- Band correlation heatmap
- Mean spectral signature + ±1σ variance envelope
- PCA of spectral dimension (randomised SVD)
- NDVI and spectral index maps (when wavelengths provided)
- False-colour preview

### ☁️ PointCloudEDA
- Bounding box, spatial extent, density (pts/unit³)
- Height (Z) profile, XY density heatmap
- Intensity and RGB channel distributions
- Label / classification distribution
- Spatial outlier detection (Z-score)
- Top-down, side-view, and interactive 3D scatter plots

---

## 📦 Installation

```bash
# Core (image EDA only)
pip install viseda

# With hyperspectral support
pip install "viseda[hyperspectral]"

# With point cloud support
pip install "viseda[pointcloud]"

# Everything
pip install "viseda[all]"
```

---

## 🚀 Quick Start

### Image EDA

```python
from viseda import ImageEDA

eda = ImageEDA(verbose=True)

# Load from a directory (labels inferred from sub-folder names)
eda.load("path/to/dataset/", label_from_parent=True)

# Or load NumPy arrays directly
import numpy as np
arrays = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(100)]
eda.load_arrays(arrays, labels=["cat", "dog"] * 50)

# Summary dict
summary = eda.summary()
print(summary["brightness"])
# {'min': 94.3, 'max': 162.1, 'mean': 127.5, 'median': 128.0, 'std': 12.4, ...}

# Full matplotlib dashboard
eda.plot()

# Save dashboard to file
eda.plot(save_path="dashboard.png")

# Sample grid
eda.plot_samples(n=25, cols=5)

# Channel correlation matrix
eda.plot_channel_correlation()

# HTML report
eda.report("report.html")
```

### Hyperspectral EDA

```python
import numpy as np
from viseda import HyperspectralEDA

wavelengths = np.linspace(400, 2500, 200)  # VNIR-SWIR
eda = HyperspectralEDA(wavelengths=wavelengths)

# Load ENVI file
eda.load("scene.hdr")

# Or a NumPy cube (H × W × Bands)
cube = np.random.rand(512, 512, 200).astype(np.float32)
eda.load(cube)

summary = eda.summary()
print(summary["shape"])          # {'H': 512, 'W': 512, 'bands': 200}
print(summary["band_snr"][:5])   # SNR for first 5 bands

# Spectral signature of a pixel
wl, sig = eda.spectral_signature(row=100, col=200)

# NDVI map
ndvi = eda.compute_ndvi(nir_band=140, red_band=60)

# PCA
scores, variance_ratio = eda.pca(n_components=10)

# Dashboard
eda.plot()
```

### Point Cloud EDA

```python
from viseda import PointCloudEDA

eda = PointCloudEDA(max_points=500_000)

# Load LAS/LAZ file
eda.load("scan.las")

# Or PLY / PCD / XYZ / NPY
eda.load("cloud.ply")

# Or a NumPy array  (N × 3+  →  X Y Z [Intensity [R G B [Label]]])
import numpy as np
pts = np.random.rand(100_000, 3).astype(np.float32)
eda.load(pts)

summary = eda.summary()
print(summary["n_points"])              # 100000
print(summary["z"])                     # height stats dict
print(summary["estimated_outliers"])    # ~95 (≈ 0.1 %)

# 2D dashboard
eda.plot()

# Interactive 3D scatter
eda.plot_3d(color_by="z")
```

### CLI

```bash
# Image EDA from the command line
viseda image /path/to/images --label-from-parent --report report.html --plot

# Hyperspectral
viseda hyper scene.hdr --wavelengths wavelengths.npy --plot

# Point cloud
viseda cloud scan.las --max 500000 --plot
```

---

## 📐 Project Structure

```
viseda/
├── viseda/
│   ├── __init__.py              # Public API: ImageEDA, HyperspectralEDA, PointCloudEDA
│   ├── cli.py                   # viseda CLI
│   ├── core/
│   │   ├── __init__.py
│   │   └── base.py              # BaseEDA abstract class
│   ├── image/
│   │   ├── __init__.py
│   │   └── eda.py               # ImageEDA + ImageRecord
│   ├── hyperspectral/
│   │   ├── __init__.py
│   │   └── eda.py               # HyperspectralEDA
│   ├── pointcloud/
│   │   ├── __init__.py
│   │   └── eda.py               # PointCloudEDA
│   ├── report/
│   │   ├── __init__.py
│   │   └── html_report.py       # HTML report generator
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Shared utilities
├── tests/
│   ├── test_image/
│   ├── test_hyperspectral/
│   └── test_pointcloud/
├── examples/
│   ├── image_eda_demo.py
│   ├── hyperspectral_demo.py
│   └── pointcloud_demo.py
├── docs/
├── pyproject.toml               # PEP 517/518 build config + metadata
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## 🔧 Optional Dependencies

| Extra | Packages | Enables |
|---|---|---|
| `hyperspectral` | `spectral`, `rasterio` | ENVI, GeoTIFF reading |
| `pointcloud` | `laspy[lazrs]`, `open3d`, `plyfile`, `scipy` | LAS/LAZ, PLY, PCD, outlier analysis |
| `report` | `jinja2` | Richer HTML reports |
| `all` | all of the above | Everything |

---

## 🏗 Publishing to PyPI

```bash
# 1. Install build tools
pip install build twine

# 2. Build wheel + sdist
python -m build

# 3. Check the package
twine check dist/*

# 4. Upload to TestPyPI first
twine upload --repository testpypi dist/*

# 5. Test the install
pip install --index-url https://test.pypi.org/simple/ viseda

# 6. Upload to PyPI
twine upload dist/*
```

---

## 🧪 Running Tests

```bash
pip install "viseda[dev]"
pytest
```

---

## 🗺 Roadmap

- [ ] Video EDA module (temporal frame statistics)
- [ ] Depth map / RGB-D EDA
- [ ] SAR (Synthetic Aperture Radar) EDA
- [ ] Interactive HTML dashboard (Plotly / Bokeh)
- [ ] Jupyter widget integration
- [ ] GPU-accelerated stats via CuPy

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss your idea,
then submit a pull request against `main`.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
