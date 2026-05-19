# Changelog

All notable changes to **VisEDA** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2024-04-01 (Initial Release)

### Added
- `ImageEDA` — full EDA for image datasets
  - Directory/file/array loading
  - Per-image stats: brightness, contrast, sharpness, entropy, pHash
  - Dominant colour extraction (K-Means)
  - Duplicate/near-duplicate detection via perceptual hashing
  - Label distribution from folder names or user-supplied dict
  - Corrupt file detection
  - Matplotlib EDA dashboard (dark theme)
  - Sample grid visualisation
  - Channel correlation matrix
  - HTML self-contained report export
- `HyperspectralEDA` — EDA for H×W×B cubes
  - ENVI / GeoTIFF / NPY / NPZ loading
  - Per-band statistics + SNR + noise (MAD)
  - Band correlation heatmap
  - Mean spectral signature + variance envelope
  - PCA (randomised SVD)
  - NDVI / spectral index maps
  - False-colour preview
- `PointCloudEDA` — EDA for 3D point clouds
  - LAS/LAZ, PLY, PCD, XYZ, NPY/NPZ loading
  - Bounding box, extent, density
  - Height (Z) distribution + XY density heatmap
  - Intensity / RGB channel statistics
  - Spatial outlier detection (Z-score)
  - Label distribution
  - Top-down, side-view, and 3D scatter plots
- `viseda` CLI with `image`, `hyper`, `cloud` sub-commands
- HTML report generator
