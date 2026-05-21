"""
viseda.pointcloud.eda
=====================
Comprehensive exploratory data analysis for point cloud datasets.

The module is designed around one unified class: ``PointCloudEDA``.
It supports analysing a single point cloud, a list of point clouds, or a full
folder of point cloud files. The emphasis is dataset-level EDA, similar to the
HyperspectralEDA design used elsewhere in VisEDA.

Supported file formats
----------------------
* ``.npy`` / ``.npz`` — NumPy arrays with shape (N, D), D >= 3
* ``.txt`` / ``.csv`` / ``.xyz`` / ``.pts`` — text point clouds
* ``.ply`` — ASCII PLY point clouds
* ``.las`` / ``.laz`` — requires ``pip install laspy``

Expected data layout
--------------------
The first three columns must be X, Y, Z coordinates. Additional columns are
kept as attributes where possible, especially RGB/intensity-like channels.

Examples
--------
Dataset from directory
>>> eda = PointCloudEDA(max_points_per_cloud=200_000)
>>> eda.load("path/to/pointclouds", label_from_parent=True)
>>> eda.summary()
>>> eda.plot_dataset()
>>> eda.plot_clouds_grid()

Arrays directly
>>> eda = PointCloudEDA()
>>> eda.load_arrays([cloud1, cloud2], labels=["bridge", "road"])
>>> eda.summary()
>>> eda.plot_dataset()
"""

from __future__ import annotations

import html
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _mpl():
    import matplotlib as mpl
    return mpl


class PointCloudRecord:
    """Container for per-cloud statistics."""

    __slots__ = (
        "path", "label", "file_ext", "file_size_kb", "n_points", "n_dims",
        "dtype", "has_color", "has_intensity", "is_corrupt", "error",
        "xyz_min", "xyz_max", "xyz_mean", "xyz_std", "bbox_size",
        "bbox_volume", "centroid", "span_x", "span_y", "span_z",
        "density", "height_min", "height_max", "height_mean", "height_std",
        "z_percentiles", "radial_distance_mean", "radial_distance_std",
        "nearest_neighbor_mean", "nearest_neighbor_median", "nearest_neighbor_std",
        "outlier_fraction", "duplicate_fraction", "finite_fraction",
        "planarity", "linearity", "scattering", "curvature",
        "normal_entropy", "intensity_mean", "intensity_std",
        "rgb_mean", "rgb_std", "sample_points",
    )

    def __init__(self) -> None:
        for field in self.__slots__:
            setattr(self, field, None)
        self.is_corrupt = False
        self.error = None


class PointCloudEDA:
    """
    Comprehensive EDA for point cloud datasets.

    Parameters
    ----------
    verbose:
        Print progress information.
    max_clouds:
        Analyse at most this number of clouds when loading from disk.
    max_points_per_cloud:
        Downsample each cloud to at most this number of points for statistics
        and plotting. Use ``None`` to keep all points.
    sample_seed:
        Random seed used for reproducible downsampling.
    compute_neighbors:
        Compute nearest-neighbour spacing metrics. Requires scikit-learn.
    compute_geometry:
        Compute PCA-based geometry descriptors: linearity, planarity,
        scattering and curvature. Requires scikit-learn.
    neighbor_sample_size:
        Maximum number of points used for nearest-neighbour computations.
    duplicate_decimals:
        Decimal places used when estimating duplicate points.
    """

    SUPPORTED_EXTS = {".npy", ".npz", ".txt", ".csv", ".xyz", ".pts", ".ply", ".las", ".laz"}

    def __init__(
        self,
        verbose: bool = True,
        max_clouds: Optional[int] = None,
        max_points_per_cloud: Optional[int] = 200_000,
        sample_seed: int = 0,
        compute_neighbors: bool = True,
        compute_geometry: bool = True,
        neighbor_sample_size: int = 10_000,
        duplicate_decimals: int = 5,
    ) -> None:
        self.verbose = verbose
        self.max_clouds = max_clouds
        self.max_points_per_cloud = max_points_per_cloud
        self.sample_seed = sample_seed
        self.compute_neighbors = compute_neighbors
        self.compute_geometry = compute_geometry
        self.neighbor_sample_size = neighbor_sample_size
        self.duplicate_decimals = duplicate_decimals

        self._records: List[PointCloudRecord] = []
        self._arrays: Dict[str, np.ndarray] = {}
        self._label_map: Dict[str, str] = {}
        self._loaded = False
        self._results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        labels: Optional[Dict[str, str]] = None,
        label_from_parent: bool = False,
        recursive: bool = True,
    ) -> "PointCloudEDA":
        """Load one file, a list of files, or a directory of point clouds."""
        paths = self._resolve_paths(source, recursive=recursive)
        if self.max_clouds is not None:
            paths = paths[: self.max_clouds]

        if labels:
            self._label_map = {str(Path(k).resolve()): v for k, v in labels.items()}

        self._records = []
        self._arrays = {}
        self._log(f"Found {len(paths)} point cloud file(s) — computing statistics …")

        for i, path in enumerate(paths):
            if self.verbose:
                self._log(f"  [{i + 1}/{len(paths)}] {path.name}")
            rec = self._analyse_file(path, label_from_parent=label_from_parent)
            self._records.append(rec)

        self._loaded = True
        bad = sum(r.is_corrupt for r in self._records)
        self._log(f"Done. {len(self._records)} cloud(s) loaded ({bad} corrupt).")
        return self

    def load_arrays(
        self,
        arrays: Sequence[np.ndarray],
        labels: Optional[Sequence[str]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> "PointCloudEDA":
        """Load point clouds directly as arrays with shape (N, D), D >= 3."""
        self._records = []
        self._arrays = {}
        self._log(f"Loading {len(arrays)} point cloud array(s) …")

        if self.max_clouds is not None:
            arrays = arrays[: self.max_clouds]
            if labels is not None:
                labels = labels[: self.max_clouds]
            if names is not None:
                names = names[: self.max_clouds]

        for i, arr in enumerate(arrays):
            rec = PointCloudRecord()
            rec.path = names[i] if names and i < len(names) else f"<array_{i}>"
            rec.file_ext = "array"
            rec.label = labels[i] if labels and i < len(labels) else None
            try:
                cloud = self._normalise_cloud_array(arr)
                cloud = self._downsample(cloud)
                self._arrays[rec.path] = cloud
                self._fill_stats(rec, cloud)
            except Exception as exc:
                rec.is_corrupt = True
                rec.error = str(exc)
                self._log(f"  ✗ {rec.path}: {exc}")
            self._records.append(rec)

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a nested summary dictionary for all loaded clouds."""
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]
        if not valid:
            result = {
                "inventory": {
                    "total_clouds": len(self._records),
                    "valid_clouds": 0,
                    "corrupt_clouds": len(corrupt),
                    "corrupt_paths": [r.path for r in corrupt],
                    "format_distribution": {},
                    "label_distribution": None,
                },
                "spatial_extent": {},
                "point_counts": {},
                "density": {},
                "quality": {},
                "geometry": {},
                "attributes": {},
                "labels": {"label_distribution": None, "class_imbalance_ratio": None},
                "error": "No valid point clouds found.",
            }
            self._results["summary"] = result
            return result

        def arr(attr: str) -> np.ndarray:
            vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
            return np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)

        labels = [r.label for r in valid if r.label]
        label_dist = dict(Counter(labels)) if labels else None
        format_dist = dict(Counter(r.file_ext for r in valid))

        bbox_sizes = np.vstack([r.bbox_size for r in valid if r.bbox_size is not None])
        xyz_means = np.vstack([r.xyz_mean for r in valid if r.xyz_mean is not None])
        xyz_stds = np.vstack([r.xyz_std for r in valid if r.xyz_std is not None])

        summary = {
            "inventory": {
                "total_clouds": len(self._records),
                "valid_clouds": len(valid),
                "corrupt_clouds": len(corrupt),
                "corrupt_paths": [r.path for r in corrupt],
                "format_distribution": format_dist,
                "label_distribution": label_dist,
                "point_count": _stat_dict(arr("n_points")),
                "dimension_count": _stat_dict(arr("n_dims")),
                "has_color_count": int(sum(bool(r.has_color) for r in valid)),
                "has_intensity_count": int(sum(bool(r.has_intensity) for r in valid)),
            },
            "geometry": {
                "bbox_volume": _stat_dict(arr("bbox_volume")),
                "density": _stat_dict(arr("density")),
                "span_x": _stat_dict(arr("span_x")),
                "span_y": _stat_dict(arr("span_y")),
                "span_z": _stat_dict(arr("span_z")),
                "bbox_size_mean": bbox_sizes.mean(axis=0).tolist() if len(bbox_sizes) else None,
                "xyz_mean_mean": xyz_means.mean(axis=0).tolist() if len(xyz_means) else None,
                "xyz_std_mean": xyz_stds.mean(axis=0).tolist() if len(xyz_stds) else None,
            },
            "height": {
                "height_min": _stat_dict(arr("height_min")),
                "height_max": _stat_dict(arr("height_max")),
                "height_mean": _stat_dict(arr("height_mean")),
                "height_std": _stat_dict(arr("height_std")),
            },
            "quality": {
                "finite_fraction": _stat_dict(arr("finite_fraction")),
                "duplicate_fraction": _stat_dict(arr("duplicate_fraction")),
                "outlier_fraction": _stat_dict(arr("outlier_fraction")),
                "nearest_neighbor_mean": _stat_dict(arr("nearest_neighbor_mean")),
                "nearest_neighbor_median": _stat_dict(arr("nearest_neighbor_median")),
                "nearest_neighbor_std": _stat_dict(arr("nearest_neighbor_std")),
            },
            "shape_descriptors": {
                "linearity": _stat_dict(arr("linearity")),
                "planarity": _stat_dict(arr("planarity")),
                "scattering": _stat_dict(arr("scattering")),
                "curvature": _stat_dict(arr("curvature")),
            },
            "attributes": {
                "intensity_mean": _stat_dict(arr("intensity_mean")),
                "intensity_std": _stat_dict(arr("intensity_std")),
                "rgb_mean_mean": _mean_array([r.rgb_mean for r in valid if r.rgb_mean is not None]),
                "rgb_std_mean": _mean_array([r.rgb_std for r in valid if r.rgb_std is not None]),
            },
        }
        self._results["summary"] = summary
        return summary

    def get_record(self, index: int = 0) -> PointCloudRecord:
        """Return the record at *index*."""
        self._check_loaded()
        return self._records[index]

    def get_cloud(self, index: int = 0) -> np.ndarray:
        """Return the loaded/downsampled point cloud array at *index*."""
        self._check_loaded()
        rec = self._records[index]
        return self._load_cloud_array(rec)

    def pairwise_cloud_distances(self, max_clouds: int = 50) -> Tuple[np.ndarray, List[str]]:
        """
        Compute a pairwise dataset-level distance matrix between clouds.

        The distance is computed from normalised summary vectors, not from raw
        point-to-point Chamfer distance, so it remains fast for many clouds.
        """
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt][:max_clouds]
        if len(valid) < 2:
            raise ValueError("Need at least two valid clouds.")
        features = []
        names = []
        for rec in valid:
            vec = [
                rec.n_points, rec.span_x, rec.span_y, rec.span_z,
                rec.bbox_volume, rec.density, rec.height_mean, rec.height_std,
                rec.nearest_neighbor_mean or 0.0, rec.duplicate_fraction or 0.0,
                rec.outlier_fraction or 0.0, rec.linearity or 0.0,
                rec.planarity or 0.0, rec.scattering or 0.0, rec.curvature or 0.0,
            ]
            features.append(vec)
            names.append(rec.label or Path(str(rec.path)).stem)
        X = np.asarray(features, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        Xn = (X - X.mean(axis=0)) / std
        diff = Xn[:, None, :] - Xn[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        return dist, names

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_dataset(
        self,
        figsize: Tuple[int, int] = (22, 18),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Dataset-level dashboard summarising all loaded point clouds."""
        self._check_loaded()
        plt = _plt()
        mpl = _mpl()
        valid = [r for r in self._records if not r.is_corrupt]
        if not valid:
            raise RuntimeError("No valid point clouds to plot.")

        s = self.summary()
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("PointCloudEDA — Dataset Analysis", fontsize=18, fontweight="bold")
        gs = mpl.gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

        self._plot_dataset_card(fig.add_subplot(gs[0, :2]), s)
        self._plot_label_dist(fig.add_subplot(gs[0, 2:]), valid)
        self._plot_hist(fig.add_subplot(gs[1, 0]), [r.n_points for r in valid], "Point Count")
        self._plot_hist(fig.add_subplot(gs[1, 1]), [r.bbox_volume for r in valid], "Bounding Box Volume")
        self._plot_hist(fig.add_subplot(gs[1, 2]), [r.density for r in valid], "Point Density")
        self._plot_hist(fig.add_subplot(gs[1, 3]), [r.height_std for r in valid], "Height Std")
        self._plot_xyz_spans(fig.add_subplot(gs[2, :2]), valid)
        self._plot_quality_bars(fig.add_subplot(gs[2, 2:]), valid)
        self._plot_shape_descriptors(fig.add_subplot(gs[3, :2]), valid)
        self._plot_pairwise_distance(fig.add_subplot(gs[3, 2:]), valid)
        self._finalise(fig, save_path, dpi)

    def plot(
        self,
        cloud_index: int = 0,
        max_points: int = 20_000,
        figsize: Tuple[int, int] = (18, 14),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Single-cloud dashboard for one selected point cloud."""
        self._check_loaded()
        plt = _plt()
        mpl = _mpl()
        rec = self._records[cloud_index]
        cloud = self._load_cloud_array(rec)
        xyz = self._sample_xyz(cloud[:, :3], max_points=max_points)

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle(f"PointCloudEDA — {rec.label or rec.path}", fontsize=16, fontweight="bold")
        gs = mpl.gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        self._plot_cloud_card(fig.add_subplot(gs[0, 0]), rec)
        ax3d = fig.add_subplot(gs[0, 1:], projection="3d")
        self._plot_3d_scatter(ax3d, xyz, rec)
        self._plot_2d_projection(fig.add_subplot(gs[1, 0]), xyz, "X", "Y", 0, 1)
        self._plot_2d_projection(fig.add_subplot(gs[1, 1]), xyz, "X", "Z", 0, 2)
        self._plot_2d_projection(fig.add_subplot(gs[1, 2]), xyz, "Y", "Z", 1, 2)
        self._plot_height_hist(fig.add_subplot(gs[2, 0]), xyz)
        self._plot_density_map(fig.add_subplot(gs[2, 1]), xyz)
        self._plot_local_spacing(fig.add_subplot(gs[2, 2]), rec)
        self._finalise(fig, save_path, dpi)

    def plot_clouds_grid(
        self,
        n: int = 12,
        cols: int = 4,
        max_points: int = 8_000,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Grid of 3D previews for multiple loaded clouds."""
        self._check_loaded()
        plt = _plt()
        valid_indices = [i for i, r in enumerate(self._records) if not r.is_corrupt][:n]
        rows = int(math.ceil(len(valid_indices) / cols))
        figsize = figsize or (cols * 4, rows * 4)
        fig = plt.figure(figsize=figsize, facecolor="white")

        for panel, idx in enumerate(valid_indices):
            rec = self._records[idx]
            cloud = self._load_cloud_array(rec)
            xyz = self._sample_xyz(cloud[:, :3], max_points=max_points)
            ax = fig.add_subplot(rows, cols, panel + 1, projection="3d")
            self._plot_3d_scatter(ax, xyz, rec, compact=True)

        fig.suptitle("PointCloudEDA — Point Cloud Preview Grid", fontsize=14, fontweight="bold")
        self._finalise(fig, save_path, dpi)

    def plot_height_distribution(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Overlay height/Z distributions across loaded clouds."""
        self._check_loaded()
        plt = _plt()
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        valid = [r for r in self._records if not r.is_corrupt]
        for rec in valid[:30]:
            cloud = self._load_cloud_array(rec)
            z = self._sample_xyz(cloud[:, :3], max_points=20_000)[:, 2]
            ax.hist(z, bins=40, histtype="step", density=True, alpha=0.7, label=(rec.label or Path(str(rec.path)).stem)[:18])
        if len(valid) <= 10:
            ax.legend(fontsize=7)
        ax.set_title("Height/Z Distribution Across Clouds")
        ax.set_xlabel("Z")
        ax.set_ylabel("Density")
        self._finalise(fig, save_path, dpi)

    def plot_pairwise_cloud_distances(
        self,
        max_clouds: int = 50,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Heatmap of pairwise cloud distances based on summary descriptors."""
        self._check_loaded()
        plt = _plt()
        dist, names = self.pairwise_cloud_distances(max_clouds=max_clouds)
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        im = ax.imshow(dist, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        if len(names) <= 30:
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Pairwise Cloud Distance Matrix")
        self._finalise(fig, save_path, dpi)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def report(self, output_path: str = "viseda_pointcloud_report.html") -> str:
        """Generate a self-contained HTML report."""
        self._check_loaded()
        summary = self.summary()
        _generate_html_report(summary, output_path)
        self._log(f"Report saved → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Internal reading and statistics
    # ------------------------------------------------------------------
    def _analyse_file(self, path: Path, label_from_parent: bool) -> PointCloudRecord:
        rec = PointCloudRecord()
        rec.path = str(path)
        rec.file_ext = path.suffix.lower()
        rec.file_size_kb = path.stat().st_size / 1024 if path.exists() else None
        rec.label = path.parent.name if label_from_parent else self._label_map.get(str(path.resolve()))
        try:
            cloud = self._read_cloud(path)
            cloud = self._downsample(cloud)
            self._arrays[rec.path] = cloud
            self._fill_stats(rec, cloud)
        except Exception as exc:
            rec.is_corrupt = True
            rec.error = str(exc)
            self._log(f"  ✗ {path.name}: {exc}")
        return rec

    def _fill_stats(self, rec: PointCloudRecord, cloud: np.ndarray) -> None:
        cloud = self._normalise_cloud_array(cloud)
        original_rows = len(cloud)
        finite_mask = np.isfinite(cloud[:, :3]).all(axis=1)
        rec.finite_fraction = float(finite_mask.mean()) if original_rows else 0.0
        cloud = cloud[finite_mask]
        if len(cloud) == 0:
            raise ValueError("Point cloud contains no finite XYZ points.")

        xyz = cloud[:, :3].astype(np.float64)
        rec.n_points = int(len(xyz))
        rec.n_dims = int(cloud.shape[1])
        rec.dtype = str(cloud.dtype)
        rec.has_color = self._detect_color(cloud)
        rec.has_intensity = cloud.shape[1] >= 4

        rec.xyz_min = xyz.min(axis=0)
        rec.xyz_max = xyz.max(axis=0)
        rec.xyz_mean = xyz.mean(axis=0)
        rec.xyz_std = xyz.std(axis=0)
        rec.centroid = rec.xyz_mean.copy()
        rec.bbox_size = rec.xyz_max - rec.xyz_min
        rec.span_x, rec.span_y, rec.span_z = [float(v) for v in rec.bbox_size]
        rec.bbox_volume = float(np.prod(np.maximum(rec.bbox_size, 1e-12)))
        rec.density = float(rec.n_points / rec.bbox_volume) if rec.bbox_volume > 0 else None

        z = xyz[:, 2]
        rec.height_min = float(z.min())
        rec.height_max = float(z.max())
        rec.height_mean = float(z.mean())
        rec.height_std = float(z.std())
        rec.z_percentiles = np.percentile(z, [0, 5, 25, 50, 75, 95, 100]).tolist()

        radial = np.linalg.norm(xyz - rec.centroid, axis=1)
        rec.radial_distance_mean = float(radial.mean())
        rec.radial_distance_std = float(radial.std())
        if radial.std() > 0:
            rec.outlier_fraction = float(np.mean(radial > radial.mean() + 3.0 * radial.std()))
        else:
            rec.outlier_fraction = 0.0

        rounded = np.round(xyz, decimals=self.duplicate_decimals)
        unique_n = len(np.unique(rounded, axis=0))
        rec.duplicate_fraction = float(1.0 - unique_n / len(xyz))

        if cloud.shape[1] >= 4:
            intensity = cloud[:, 3].astype(float)
            intensity = intensity[np.isfinite(intensity)]
            if len(intensity):
                rec.intensity_mean = float(intensity.mean())
                rec.intensity_std = float(intensity.std())

        if rec.has_color:
            rgb = self._extract_rgb(cloud)
            if rgb is not None and len(rgb):
                rec.rgb_mean = rgb.mean(axis=0).tolist()
                rec.rgb_std = rgb.std(axis=0).tolist()

        rec.sample_points = self._sample_xyz(xyz, max_points=min(5000, len(xyz)))

        if self.compute_neighbors:
            self._fill_neighbor_stats(rec, xyz)
        else:
            rec.nearest_neighbor_mean = None
            rec.nearest_neighbor_median = None
            rec.nearest_neighbor_std = None

        if self.compute_geometry:
            self._fill_geometry_stats(rec, xyz)
        else:
            rec.linearity = rec.planarity = rec.scattering = rec.curvature = None
            rec.normal_entropy = None

    def _fill_neighbor_stats(self, rec: PointCloudRecord, xyz: np.ndarray) -> None:
        if len(xyz) < 2:
            rec.nearest_neighbor_mean = 0.0
            rec.nearest_neighbor_median = 0.0
            rec.nearest_neighbor_std = 0.0
            return
        try:
            from sklearn.neighbors import NearestNeighbors
            pts = self._sample_xyz(xyz, max_points=min(self.neighbor_sample_size, len(xyz)))
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(pts)
            dists, _ = nn.kneighbors(pts)
            nearest = dists[:, 1]
            rec.nearest_neighbor_mean = float(nearest.mean())
            rec.nearest_neighbor_median = float(np.median(nearest))
            rec.nearest_neighbor_std = float(nearest.std())
        except Exception:
            rec.nearest_neighbor_mean = None
            rec.nearest_neighbor_median = None
            rec.nearest_neighbor_std = None

    def _fill_geometry_stats(self, rec: PointCloudRecord, xyz: np.ndarray) -> None:
        if len(xyz) < 3:
            rec.linearity = rec.planarity = rec.scattering = rec.curvature = 0.0
            rec.normal_entropy = 0.0
            return
        pts = self._sample_xyz(xyz, max_points=min(50_000, len(xyz)))
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0))[::-1]
        l1, l2, l3 = eig + 1e-12
        rec.linearity = float((l1 - l2) / l1)
        rec.planarity = float((l2 - l3) / l1)
        rec.scattering = float(l3 / l1)
        rec.curvature = float(l3 / (l1 + l2 + l3))
        probs = eig / eig.sum()
        rec.normal_entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    def _resolve_paths(self, source: Union[str, Path, Sequence[Union[str, Path]]], recursive: bool) -> List[Path]:
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                globber = p.rglob if recursive else p.glob
                paths = [x for x in globber("*") if x.is_file() and x.suffix.lower() in self.SUPPORTED_EXTS]
                return sorted(paths)
            if p.is_file():
                return [p]
            raise FileNotFoundError(f"Source not found: {source}")
        paths = [Path(x) for x in source]
        return [p for p in paths if p.suffix.lower() in self.SUPPORTED_EXTS]

    def _read_cloud(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return self._normalise_cloud_array(np.load(str(path)))
        if suffix == ".npz":
            data = np.load(str(path))
            key = list(data.keys())[0]
            return self._normalise_cloud_array(data[key])
        if suffix in {".txt", ".csv", ".xyz", ".pts"}:
            delimiter = "," if suffix == ".csv" else None
            return self._read_text_cloud(path, delimiter=delimiter)
        if suffix == ".ply":
            return self._read_ascii_ply(path)
        if suffix in {".las", ".laz"}:
            return self._read_las(path)
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    def _read_text_cloud(self, path: Path, delimiter: Optional[str]) -> np.ndarray:
        try:
            arr = np.loadtxt(str(path), delimiter=delimiter, comments="#")
        except ValueError:
            arr = np.genfromtxt(str(path), delimiter=delimiter, comments="#", names=None)
        return self._normalise_cloud_array(arr)

    def _read_ascii_ply(self, path: Path) -> np.ndarray:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            header = []
            vertex_count = None
            properties = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Invalid PLY file: missing end_header.")
                stripped = line.strip()
                header.append(stripped)
                if stripped.startswith("format") and "ascii" not in stripped:
                    raise ValueError("Only ASCII .ply is supported without extra dependencies.")
                if stripped.startswith("element vertex"):
                    vertex_count = int(stripped.split()[-1])
                if stripped.startswith("property") and vertex_count is not None:
                    properties.append(stripped.split()[-1])
                if stripped == "end_header":
                    break
            if vertex_count is None:
                raise ValueError("Invalid PLY file: no vertex count.")
            rows = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                vals = [float(x) for x in line.strip().split()]
                rows.append(vals)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.shape[1] >= 3 and properties[:3] != ["x", "y", "z"]:
            lower = [p.lower() for p in properties]
            if all(k in lower for k in ["x", "y", "z"]):
                idx = [lower.index("x"), lower.index("y"), lower.index("z")]
                rest = [i for i in range(arr.shape[1]) if i not in idx]
                arr = arr[:, idx + rest]
        return self._normalise_cloud_array(arr)

    def _read_las(self, path: Path) -> np.ndarray:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError("LAS/LAZ support requires: pip install laspy") from exc
        las = laspy.read(str(path))
        cols = [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
        if hasattr(las, "intensity"):
            cols.append(np.asarray(las.intensity))
        for color_name in ("red", "green", "blue"):
            if hasattr(las, color_name):
                cols.append(np.asarray(getattr(las, color_name)))
        return self._normalise_cloud_array(np.column_stack(cols))

    def _normalise_cloud_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.size < 3:
                raise ValueError("Point cloud array must contain at least XYZ columns.")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("Point cloud array must have shape (N, D), with D >= 3.")
        return arr.astype(np.float32, copy=False)

    def _downsample(self, cloud: np.ndarray) -> np.ndarray:
        if self.max_points_per_cloud is None or len(cloud) <= self.max_points_per_cloud:
            return cloud
        rng = np.random.default_rng(self.sample_seed)
        idx = rng.choice(len(cloud), size=self.max_points_per_cloud, replace=False)
        return cloud[np.sort(idx)]

    def _load_cloud_array(self, rec: PointCloudRecord) -> np.ndarray:
        if rec.path in self._arrays:
            return self._arrays[rec.path]
        if rec.path and not str(rec.path).startswith("<array"):
            cloud = self._downsample(self._read_cloud(Path(str(rec.path))))
            self._arrays[rec.path] = cloud
            return cloud
        raise ValueError("Raw cloud array is unavailable for this record.")

    def _detect_color(self, cloud: np.ndarray) -> bool:
        if cloud.shape[1] < 6:
            return False
        rgb = cloud[:, -3:]
        return bool(np.nanmax(rgb) > 1.0 or np.nanmax(rgb) <= 1.0)

    def _extract_rgb(self, cloud: np.ndarray) -> Optional[np.ndarray]:
        if cloud.shape[1] < 6:
            return None
        rgb = cloud[:, -3:].astype(float)
        finite = np.isfinite(rgb).all(axis=1)
        rgb = rgb[finite]
        if len(rgb) == 0:
            return None
        if rgb.max() > 1.0:
            rgb = rgb / max(255.0, rgb.max())
        return np.clip(rgb, 0, 1)

    def _sample_xyz(self, xyz: np.ndarray, max_points: int = 10_000) -> np.ndarray:
        xyz = np.asarray(xyz)
        if len(xyz) <= max_points:
            return xyz
        rng = np.random.default_rng(self.sample_seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        return xyz[idx]

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("No point clouds loaded. Call load() or load_arrays() first.")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[viseda] {message}")

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _plot_dataset_card(self, ax, s: Dict[str, Any]) -> None:
        ax.axis("off")
        inv = s["inventory"]
        geo = s["geometry"]
        quality = s["quality"]
        lines = [
            f"Total clouds:     {inv['total_clouds']}",
            f"Valid clouds:     {inv['valid_clouds']}",
            f"Corrupt clouds:   {inv['corrupt_clouds']}",
            f"Formats:          {inv['format_distribution']}",
            f"Labels:           {inv['label_distribution']}",
            f"Mean points:      {inv['point_count'].get('mean', 'N/A')}",
            f"Mean bbox volume: {geo['bbox_volume'].get('mean', 'N/A')}",
            f"Mean density:     {geo['density'].get('mean', 'N/A')}",
            f"Duplicate frac:   {quality['duplicate_fraction'].get('mean', 'N/A')}",
            f"Outlier frac:     {quality['outlier_fraction'].get('mean', 'N/A')}",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", transform=ax.transAxes,
                fontsize=9, family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f2f2"))
        ax.set_title("Dataset Overview")

    def _plot_cloud_card(self, ax, rec: PointCloudRecord) -> None:
        ax.axis("off")
        lines = [
            f"Path:        {rec.path}",
            f"Label:       {rec.label or 'N/A'}",
            f"Points:      {rec.n_points:,}",
            f"Dimensions:  {rec.n_dims}",
            f"BBox:        {np.round(rec.bbox_size, 3).tolist()}",
            f"Volume:      {rec.bbox_volume:.4f}",
            f"Density:     {rec.density:.4f}",
            f"Height mean: {rec.height_mean:.4f}",
            f"NN mean:     {rec.nearest_neighbor_mean if rec.nearest_neighbor_mean is not None else 'N/A'}",
            f"Duplicates:  {rec.duplicate_fraction:.4f}",
            f"Outliers:    {rec.outlier_fraction:.4f}",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", transform=ax.transAxes,
                fontsize=8, family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f2f2"))
        ax.set_title("Cloud Overview")

    def _plot_label_dist(self, ax, valid: List[PointCloudRecord]) -> None:
        labels = [r.label for r in valid if r.label]
        if not labels:
            ax.text(0.5, 0.5, "No labels provided", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Label Distribution")
            return
        names, counts = zip(*Counter(labels).most_common())
        ax.barh(range(len(names)), counts)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_title("Label Distribution")
        ax.set_xlabel("Cloud count")

    def _plot_hist(self, ax, values: Sequence[Any], title: str, bins: int = 20) -> None:
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        if len(vals) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        ax.hist(vals, bins=min(bins, max(5, len(vals))))
        ax.axvline(vals.mean(), linestyle="--", linewidth=1, label=f"mean={vals.mean():.3g}")
        ax.legend(fontsize=7)
        ax.set_title(title)

    def _plot_xyz_spans(self, ax, valid: List[PointCloudRecord]) -> None:
        spans = np.asarray([r.bbox_size for r in valid if r.bbox_size is not None], dtype=float)
        if len(spans) == 0:
            ax.set_title("XYZ Spans")
            return
        x = np.arange(len(spans))
        ax.plot(x, spans[:, 0], marker=".", label="X span")
        ax.plot(x, spans[:, 1], marker=".", label="Y span")
        ax.plot(x, spans[:, 2], marker=".", label="Z span")
        ax.set_title("XYZ Bounding Box Spans")
        ax.set_xlabel("Cloud index")
        ax.legend(fontsize=8)

    def _plot_quality_bars(self, ax, valid: List[PointCloudRecord]) -> None:
        metrics = {
            "Finite": [r.finite_fraction for r in valid],
            "Duplicate": [r.duplicate_fraction for r in valid],
            "Outlier": [r.outlier_fraction for r in valid],
            "NN mean": [r.nearest_neighbor_mean for r in valid if r.nearest_neighbor_mean is not None],
        }
        labels, means = [], []
        for name, values in metrics.items():
            vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
            if len(vals):
                labels.append(name)
                means.append(float(vals.mean()))
        ax.bar(labels, means)
        ax.set_title("Quality Metrics")
        ax.tick_params(axis="x", rotation=30)

    def _plot_shape_descriptors(self, ax, valid: List[PointCloudRecord]) -> None:
        names = ["linearity", "planarity", "scattering", "curvature"]
        means = []
        for name in names:
            vals = np.asarray([getattr(r, name) for r in valid if getattr(r, name) is not None], dtype=float)
            means.append(vals.mean() if len(vals) else 0)
        ax.bar([n.title() for n in names], means)
        ax.set_title("PCA Shape Descriptors")
        ax.tick_params(axis="x", rotation=20)

    def _plot_pairwise_distance(self, ax, valid: List[PointCloudRecord]) -> None:
        if len(valid) < 2:
            ax.set_title("Pairwise Distances")
            return
        try:
            dist, names = self.pairwise_cloud_distances(max_clouds=min(30, len(valid)))
            im = ax.imshow(dist, aspect="auto")
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
            ax.set_title("Cloud Distance Heatmap")
            if len(names) <= 15:
                ax.set_xticks(range(len(names)))
                ax.set_yticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
                ax.set_yticklabels(names, fontsize=6)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Cloud Distance Heatmap")

    def _plot_3d_scatter(self, ax, xyz: np.ndarray, rec: PointCloudRecord, compact: bool = False) -> None:
        colors = xyz[:, 2]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1, alpha=0.6)
        ax.set_title((rec.label or Path(str(rec.path)).stem)[:30], fontsize=8 if compact else 10)
        if not compact:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    def _plot_2d_projection(self, ax, xyz: np.ndarray, xlab: str, ylab: str, xi: int, yi: int) -> None:
        ax.scatter(xyz[:, xi], xyz[:, yi], s=1, alpha=0.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{xlab}-{ylab} Projection")

    def _plot_height_hist(self, ax, xyz: np.ndarray) -> None:
        ax.hist(xyz[:, 2], bins=40)
        ax.set_title("Height/Z Distribution")
        ax.set_xlabel("Z")

    def _plot_density_map(self, ax, xyz: np.ndarray) -> None:
        h = ax.hist2d(xyz[:, 0], xyz[:, 1], bins=80)
        ax.figure.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        ax.set_title("XY Density Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    def _plot_local_spacing(self, ax, rec: PointCloudRecord) -> None:
        vals = [rec.nearest_neighbor_mean, rec.nearest_neighbor_median, rec.nearest_neighbor_std]
        vals = [0 if v is None else v for v in vals]
        ax.bar(["NN mean", "NN median", "NN std"], vals)
        ax.set_title("Nearest-Neighbour Spacing")
        ax.tick_params(axis="x", rotation=20)

    def _finalise(self, fig, save_path: Optional[str], dpi: int) -> None:
        plt = _plt()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()


def _stat_dict(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
        "min": round(float(values.min()), 6),
        "q25": round(float(np.percentile(values, 25)), 6),
        "median": round(float(np.median(values)), 6),
        "q75": round(float(np.percentile(values, 75)), 6),
        "max": round(float(values.max()), 6),
    }


def _mean_array(arrays: List[Any]) -> Optional[List[float]]:
    if not arrays:
        return None
    vals = np.asarray(arrays, dtype=float)
    if vals.ndim == 1:
        return vals.tolist()
    return vals.mean(axis=0).round(6).tolist()


def _generate_html_report(summary: Dict[str, Any], output_path: str) -> None:
    # Styled HTML report matching the HyperspectralEDA report design.
    # It uses cards, grids, statistic rows, and compact distribution bars
    # instead of raw JSON <pre> blocks.

    def fmt(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "N/A"
            return f"{float(value):.4f}"
        return html.escape(str(value))

    def stat_rows(stats: Dict[str, Any], order: Optional[List[str]] = None) -> str:
        if not isinstance(stats, dict) or not stats:
            return '<div class="stat"><span>No data</span><span class="val">N/A</span></div>'
        order = order or ["min", "max", "mean", "median", "std", "q25", "q75", "count"]
        labels = {"q25": "p25", "q75": "p75", "std": "std", "min": "min", "max": "max", "mean": "mean", "median": "median", "count": "count"}
        rows = []
        for key in order:
            if key in stats:
                rows.append(
                    f'<div class="stat"><span>{labels.get(key, html.escape(str(key)))}</span>'
                    f'<span class="val">{fmt(stats.get(key))}</span></div>'
                )
        if not rows:
            for key, value in stats.items():
                rows.append(
                    f'<div class="stat"><span>{html.escape(str(key))}</span>'
                    f'<span class="val">{fmt(value)}</span></div>'
                )
        return "".join(rows)

    def stat_card(title: str, stats: Dict[str, Any]) -> str:
        return f'<div class="card"><h3>{html.escape(title)}</h3>{stat_rows(stats)}</div>'

    def simple_card(title: str, items: Dict[str, Any]) -> str:
        rows = []
        for key, value in items.items():
            rows.append(
                f'<div class="stat"><span>{html.escape(str(key))}</span>'
                f'<span class="val">{fmt(value)}</span></div>'
            )
        body = "".join(rows) or '<div class="stat"><span>No data</span><span class="val">N/A</span></div>'
        return f'<div class="card"><h3>{html.escape(title)}</h3>{body}</div>'

    def vector_card(title: str, values: Any, labels: Optional[List[str]] = None) -> str:
        if values is None:
            return simple_card(title, {"value": None})
        labels = labels or [f"Value {i+1}" for i in range(len(values))]
        return simple_card(title, {label: val for label, val in zip(labels, values)})

    def bar_chart(title: str, data: Optional[Dict[Any, Any]], span: int = 1) -> str:
        data = data or {}
        if not data:
            body = '<div class="stat"><span>No data</span><span class="val">N/A</span></div>'
        else:
            vals = [float(v) for v in data.values() if v is not None]
            max_val = max(vals) if vals else 1.0
            if max_val <= 0:
                max_val = 1.0
            rows = []
            for key, value in data.items():
                value = 0 if value is None else value
                width = max(0.0, min(100.0, (float(value) / max_val) * 100.0))
                rows.append(
                    '<div class="bar-row">'
                    f'<span class="bar-label">{html.escape(str(key))}</span>'
                    f'<div class="bar"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
                    f'<span class="bar-count">{fmt(value)}</span>'
                    '</div>'
                )
            body = f'<div class="bar-wrap">{"".join(rows)}</div>'
        style = f' style="grid-column:span {span};"' if span > 1 else ""
        return f'<div class="card"{style}><h3>{html.escape(title)}</h3>{body}</div>'

    inv = summary.get("inventory", {}) or {}
    geom = summary.get("geometry", {}) or {}
    height = summary.get("height", {}) or {}
    quality = summary.get("quality", {}) or {}
    shape = summary.get("shape_descriptors", {}) or {}
    attrs = summary.get("attributes", {}) or {}

    html_text = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VisEDA — PointCloud Report</title>
<style>
:root{{--bg:white;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;
      --muted:#57606a;--accent:#58a6ff;--green:#3fb950;--red:#f78166;
      --yellow:#e3b341;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     padding:2rem;}}
h1{{font-size:1.9rem;margin-bottom:.25rem}}
h2{{font-size:1.05rem;color:var(--accent);margin:1.8rem 0 .6rem}}
h3{{font-size:.78rem;color:var(--muted);text-transform:uppercase;
    letter-spacing:.05em;margin-bottom:.5rem}}
.sub{{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:.9rem}}
.card{{background:var(--surface);border:1px solid var(--border);
       border-radius:8px;padding:1rem}}
.stat{{display:flex;justify-content:space-between;font-size:.82rem;
       padding:.18rem 0;border-bottom:1px solid var(--border);gap:.75rem}}
.stat:last-child{{border-bottom:none}}
.val{{color:var(--accent);font-variant-numeric:tabular-nums;text-align:right}}
.badge{{display:inline-block;padding:.15rem .5rem;border-radius:12px;
        font-size:.72rem;font-weight:600;margin:.15rem}}
.badge-blue{{background:rgba(88,166,255,.15);color:var(--accent)}}
.badge-green{{background:rgba(63,185,80,.15);color:var(--green)}}
.badge-red{{background:rgba(247,129,102,.15);color:var(--red)}}
.bar-wrap{{margin-top:.4rem}}
.bar-row{{display:flex;align-items:center;gap:.4rem;margin:.18rem 0;font-size:.76rem}}
.bar-label{{width:120px;overflow:hidden;text-overflow:ellipsis;
            white-space:nowrap;color:var(--muted)}}
.bar{{flex:1;background:var(--border);border-radius:3px;height:9px}}
.bar-fill{{height:100%;border-radius:3px;background:var(--accent)}}
.bar-count{{width:55px;text-align:right;color:var(--accent)}}
footer{{margin-top:3rem;color:var(--muted);font-size:.72rem;
        border-top:1px solid var(--border);padding-top:1rem}}
</style></head><body>
<h1>☁️ VisEDA — PointCloud EDA Report</h1>
<p class="sub">Generated by <strong>VisEDA</strong></p>
<p style="margin-bottom:1rem"><span class="badge badge-blue">{fmt(inv.get('total_clouds'))} clouds</span><span class="badge badge-green">{fmt(inv.get('valid_clouds'))} valid</span><span class="badge badge-red">{fmt(inv.get('corrupt_clouds'))} corrupt</span></p>

<h2>📦 Inventory</h2>
<div class="grid">
  {simple_card('Counts', {'Total clouds': inv.get('total_clouds'), 'Valid clouds': inv.get('valid_clouds'), 'Corrupt clouds': inv.get('corrupt_clouds'), 'Clouds with colour': inv.get('has_color_count'), 'Clouds with intensity': inv.get('has_intensity_count')})}
  {bar_chart('Label Distribution', inv.get('label_distribution'), span=2)}
  {bar_chart('Format Distribution', inv.get('format_distribution'))}
  {stat_card('Point Count', inv.get('point_count', {}))}
  {stat_card('Dimension Count', inv.get('dimension_count', {}))}
</div>

<h2>📐 Geometry</h2>
<div class="grid">
  {stat_card('Bounding Box Volume', geom.get('bbox_volume', {}))}
  {stat_card('Point Density', geom.get('density', {}))}
  {stat_card('Span X', geom.get('span_x', {}))}
  {stat_card('Span Y', geom.get('span_y', {}))}
  {stat_card('Span Z', geom.get('span_z', {}))}
  {vector_card('Mean Bounding Box Size', geom.get('bbox_size_mean'), ['X span', 'Y span', 'Z span'])}
  {vector_card('Mean XYZ Position', geom.get('xyz_mean_mean'), ['X mean', 'Y mean', 'Z mean'])}
  {vector_card('Mean XYZ Std', geom.get('xyz_std_mean'), ['X std', 'Y std', 'Z std'])}
</div>

<h2>↕️ Height</h2>
<div class="grid">
  {stat_card('Height Minimum', height.get('height_min', {}))}
  {stat_card('Height Maximum', height.get('height_max', {}))}
  {stat_card('Height Mean', height.get('height_mean', {}))}
  {stat_card('Height Std', height.get('height_std', {}))}
</div>

<h2>✅ Quality</h2>
<div class="grid">
  {stat_card('Finite Fraction', quality.get('finite_fraction', {}))}
  {stat_card('Duplicate Fraction', quality.get('duplicate_fraction', {}))}
  {stat_card('Outlier Fraction', quality.get('outlier_fraction', {}))}
  {stat_card('Nearest-Neighbour Mean', quality.get('nearest_neighbor_mean', {}))}
  {stat_card('Nearest-Neighbour Median', quality.get('nearest_neighbor_median', {}))}
  {stat_card('Nearest-Neighbour Std', quality.get('nearest_neighbor_std', {}))}
</div>

<h2>🧊 Shape Descriptors</h2>
<div class="grid">
  {stat_card('Linearity', shape.get('linearity', {}))}
  {stat_card('Planarity', shape.get('planarity', {}))}
  {stat_card('Scattering', shape.get('scattering', {}))}
  {stat_card('Curvature', shape.get('curvature', {}))}
</div>

<h2>🎨 Attributes</h2>
<div class="grid">
  {stat_card('Intensity Mean', attrs.get('intensity_mean', {}))}
  {stat_card('Intensity Std', attrs.get('intensity_std', {}))}
  {vector_card('RGB Mean', attrs.get('rgb_mean_mean'), ['R mean', 'G mean', 'B mean'])}
  {vector_card('RGB Std', attrs.get('rgb_std_mean'), ['R std', 'G std', 'B std'])}
</div>

<footer>Generated by VisEDA — Visual Exploratory Data Analysis</footer>
</body></html>'''
    Path(output_path).write_text(html_text, encoding="utf-8")
