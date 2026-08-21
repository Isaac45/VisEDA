# Changelog

All notable changes to **VisEDA** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses semantic versioning.

## [1.0.0] — 2026-08-21

### Added
- ImageEDA for image dataset inventory, quality, colour, texture, frequency,
  duplicate, class-balance, visualisation, and HTML-report workflows.
- VideoEDA for temporal, motion, blur, scene-change, colour, similarity,
  visualisation, and HTML-report workflows.
- HyperspectralEDA for per-band statistics, spectral-quality analysis,
  spectral indices, PCA, false-colour views, visualisation, and HTML reports.
- PointCloudEDA for geometry, density, height, nearest-neighbour, duplicate,
  outlier, shape-descriptor, visualisation, and HTML-report workflows.
- TextEDA for lexical, length, readability, script, duplicate,
  TF-IDF-distance, visualisation, and HTML-report workflows.
- `viseda` command-line interface with `image`, `hyper`, `cloud`, `video`,
  and `text` subcommands.
- Python 3.9+ packaging through `pyproject.toml`.
