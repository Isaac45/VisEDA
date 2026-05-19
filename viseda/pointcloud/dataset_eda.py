"""
viseda.pointcloud.dataset_eda
------------------------------
Dataset-level EDA across a **collection** of point cloud files.

This complements ``PointCloudEDA`` (single cloud) by treating a directory or
list of files as a dataset and computing aggregate statistics across all clouds
— analogous to what ``ImageEDA`` does for image datasets.

Analyses covered
~~~~~~~~~~~~~~~~
* Dataset inventory (count, point counts, file sizes, format distribution)
* Per-cloud bounding box / spatial extent distributions
* Per-cloud point-count distribution
* Density distribution across clouds
* Height (Z) range distribution
* Intensity statistics distribution (if available)
* Label / classification distribution aggregated across clouds
* Corrupt / unreadable file detection
* Cross-cloud label inventory from folder names
* Full matplotlib dashboard
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from viseda.core.base import BaseEDA
from viseda.utils.helpers import (
    POINTCLOUD_EXTENSIONS,
    discover_files,
    safe_divide,
)


# ---------------------------------------------------------------------------
# Per-cloud record
# ---------------------------------------------------------------------------

class CloudRecord:
    """Lightweight statistics for one point cloud file."""

    __slots__ = (
        "path", "label", "n_points", "file_size_kb",
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
        "x_extent", "y_extent", "z_extent",
        "density",             # points per unit volume
        "has_intensity",
        "intensity_mean", "intensity_std",
        "has_rgb",
        "has_labels",
        "label_counts",        # dict {class_id: count}
        "outlier_fraction",
        "is_corrupt",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.is_corrupt = False
        self.has_intensity = False
        self.has_rgb = False
        self.has_labels = False
        self.label_counts = {}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PointCloudDatasetEDA(BaseEDA):
    """
    Perform EDA across a **collection** of point cloud files.

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    max_points_per_cloud : int | None
        Subsample each cloud to at most this many points for fast stats.
    max_clouds : int | None
        Cap the number of clouds analysed.
    random_seed : int
        Seed for subsampling.

    Examples
    --------
    >>> from viseda.pointcloud import PointCloudDatasetEDA
    >>> eda = PointCloudDatasetEDA()
    >>> eda.load("path/to/clouds/")   # directory of .las / .ply / .xyz …
    >>> eda.summary()
    >>> eda.plot()
    """

    def __init__(
        self,
        verbose: bool = True,
        max_points_per_cloud: Optional[int] = 200_000,
        max_clouds: Optional[int] = None,
        random_seed: int = 42,
    ):
        super().__init__(verbose=verbose)
        self.max_points_per_cloud = max_points_per_cloud
        self.max_clouds = max_clouds
        self.random_seed = random_seed

        self._records: List[CloudRecord] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        label_from_parent: bool = False,
        labels: Optional[Dict[str, str]] = None,
        recursive: bool = True,
    ) -> "PointCloudDatasetEDA":
        """
        Load and analyse a collection of point cloud files.

        Parameters
        ----------
        source
            A directory, a single file, or a list of file paths.
            Supported: ``.las``, ``.laz``, ``.ply``, ``.pcd``,
            ``.xyz``, ``.txt``, ``.npy``, ``.npz``.
        label_from_parent
            Use the parent folder name as each cloud's label.
        labels
            Optional ``{path: label}`` mapping.
        recursive
            Recurse into sub-directories.
        """
        paths = self._resolve_paths(source, recursive)
        if self.max_clouds:
            paths = paths[: self.max_clouds]

        self._label_map: Dict[str, str] = {}
        if labels:
            self._label_map = {str(Path(k).resolve()): v for k, v in labels.items()}

        self._log(f"Found {len(paths)} cloud files — computing statistics …")
        self._records = []

        for i, p in enumerate(paths):
            if self.verbose and i % 10 == 0:
                self._log(f"  {i}/{len(paths)}  {p.name}")
            rec = self._analyse_single(p, label_from_parent=label_from_parent)
            self._records.append(rec)

        self._loaded = True
        n_corrupt = sum(r.is_corrupt for r in self._records)
        self._log(
            f"Loaded {len(self._records)} clouds "
            f"({n_corrupt} corrupt / unreadable)."
        )
        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics across all clouds."""
        self._check_loaded()
        valid   = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]

        n_pts    = np.array([r.n_points    for r in valid])
        densities = np.array([r.density    for r in valid if r.density])
        z_ranges = np.array([r.z_extent    for r in valid if r.z_extent])
        x_ranges = np.array([r.x_extent    for r in valid if r.x_extent])
        y_ranges = np.array([r.y_extent    for r in valid if r.y_extent])
        fsizes   = np.array([r.file_size_kb for r in valid if r.file_size_kb])
        outliers = np.array([r.outlier_fraction for r in valid
                             if r.outlier_fraction is not None])

        fmt_dist   = dict(Counter(Path(r.path).suffix.lower() for r in valid))
        label_dist = None
        if any(r.label for r in valid):
            label_dist = dict(Counter(r.label for r in valid))

        # Aggregate class label counts across all clouds
        agg_labels: Dict[int, int] = {}
        for r in valid:
            for cls, cnt in (r.label_counts or {}).items():
                agg_labels[cls] = agg_labels.get(cls, 0) + cnt

        result = {
            "total_clouds":   len(self._records),
            "valid_clouds":   len(valid),
            "corrupt_clouds": len(corrupt),
            "corrupt_paths":  [r.path for r in corrupt],
            "format_distribution": fmt_dist,
            "label_distribution":  label_dist,
            "n_points":       _stat_dict(n_pts),
            "total_points":   int(n_pts.sum()),
            "density":        _stat_dict(densities) if len(densities) else {},
            "z_extent":       _stat_dict(z_ranges)  if len(z_ranges)  else {},
            "x_extent":       _stat_dict(x_ranges)  if len(x_ranges)  else {},
            "y_extent":       _stat_dict(y_ranges)  if len(y_ranges)  else {},
            "file_size_kb":   _stat_dict(fsizes)    if len(fsizes)    else {},
            "outlier_fraction": _stat_dict(outliers) if len(outliers) else {},
            "has_intensity_count": sum(r.has_intensity for r in valid),
            "has_rgb_count":       sum(r.has_rgb       for r in valid),
            "has_labels_count":    sum(r.has_labels    for r in valid),
            "aggregated_class_distribution": agg_labels,
        }
        self._store("summary", result)
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        figsize: Tuple[int, int] = (20, 20),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Render a dataset-level EDA dashboard."""
        self._check_loaded()
        plt = self._plt()
        mpl = self._mpl()

        valid = [r for r in self._records if not r.is_corrupt]
        s = self.summary()

        fig = plt.figure(figsize=figsize, facecolor="#0d1117")
        fig.suptitle(
            "VisEDA — Point Cloud Dataset Analysis",
            fontsize=20, color="white", y=0.98, fontweight="bold",
        )

        gs = mpl.gridspec.GridSpec(
            4, 4, figure=fig,
            hspace=0.50, wspace=0.35,
            left=0.06, right=0.97, top=0.94, bottom=0.04,
        )

        def make_ax(*args):
            ax = fig.add_subplot(*args)
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            return ax

        # ── Row 0: overview + label dist ──────────────────────────────
        ax_info = make_ax(gs[0, :2])
        self._plot_info_text(ax_info, s)

        ax_lbl = make_ax(gs[0, 2:])
        self._plot_label_distribution(ax_lbl, valid)

        # ── Row 1: point-count, density, z-range, file size ───────────
        ax_npts  = make_ax(gs[1, 0])
        ax_dens  = make_ax(gs[1, 1])
        ax_zrng  = make_ax(gs[1, 2])
        ax_fsize = make_ax(gs[1, 3])
        self._plot_hist(ax_npts,  [r.n_points    for r in valid],
                        "Points per Cloud",   "#58a6ff")
        self._plot_hist(ax_dens,  [r.density     for r in valid if r.density],
                        "Density (pts/unit³)", "#3fb950")
        self._plot_hist(ax_zrng,  [r.z_extent    for r in valid if r.z_extent],
                        "Z Extent (height range)", "#d2a8ff")
        self._plot_hist(ax_fsize, [r.file_size_kb for r in valid if r.file_size_kb],
                        "File Size (KB)", "#ffa657")

        # ── Row 2: spatial extents + outlier + format ─────────────────
        ax_xrng    = make_ax(gs[2, 0])
        ax_yrng    = make_ax(gs[2, 1])
        ax_outlier = make_ax(gs[2, 2])
        ax_fmt     = make_ax(gs[2, 3])
        self._plot_hist(ax_xrng, [r.x_extent for r in valid if r.x_extent],
                        "X Extent", "#79c0ff")
        self._plot_hist(ax_yrng, [r.y_extent for r in valid if r.y_extent],
                        "Y Extent", "#56d364")
        self._plot_hist(ax_outlier,
                        [r.outlier_fraction * 100 for r in valid
                         if r.outlier_fraction is not None],
                        "Outlier Fraction (%)", "#f78166")
        self._plot_format_dist(ax_fmt, s)

        # ── Row 3: intensity + aggregated class labels + scatter ───────
        ax_int  = make_ax(gs[3, 0])
        ax_cls  = make_ax(gs[3, 1:3])
        ax_scat = make_ax(gs[3, 3])
        self._plot_intensity_dist(ax_int, valid)
        self._plot_class_distribution(ax_cls, s)
        self._plot_npoints_vs_density(ax_scat, valid)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            self._log(f"Dashboard saved → {save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Internal: per-cloud analysis
    # ------------------------------------------------------------------

    def _analyse_single(self, path: Path, label_from_parent: bool) -> CloudRecord:
        rec = CloudRecord()
        rec.path = str(path)
        rec.file_size_kb = path.stat().st_size / 1024 if path.exists() else None

        if label_from_parent:
            rec.label = path.parent.name
        else:
            rec.label = self._label_map.get(str(path.resolve()))

        try:
            xyz, intensity, rgb, labels = self._read_cloud(path)
        except Exception as e:
            self._log(f"  ✗ Could not read {path.name}: {e}")
            rec.is_corrupt = True
            return rec

        # Subsample for speed
        N = len(xyz)
        if self.max_points_per_cloud and N > self.max_points_per_cloud:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(N, self.max_points_per_cloud, replace=False)
            xyz = xyz[idx]
            if intensity is not None: intensity = intensity[idx]
            if rgb       is not None: rgb       = rgb[idx]
            if labels    is not None: labels    = labels[idx]

        rec.n_points = len(xyz)

        # Bounding box
        bmin = xyz.min(axis=0)
        bmax = xyz.max(axis=0)
        ext  = bmax - bmin
        rec.x_min, rec.y_min, rec.z_min = bmin.tolist()
        rec.x_max, rec.y_max, rec.z_max = bmax.tolist()
        rec.x_extent, rec.y_extent, rec.z_extent = ext.tolist()

        volume = float(np.prod(np.maximum(ext, 1e-9)))
        rec.density = rec.n_points / volume

        # Intensity
        if intensity is not None:
            rec.has_intensity  = True
            rec.intensity_mean = float(intensity.mean())
            rec.intensity_std  = float(intensity.std())

        # RGB
        rec.has_rgb = rgb is not None

        # Labels
        if labels is not None:
            rec.has_labels = True
            rec.label_counts = {int(k): int(v)
                                for k, v in zip(*np.unique(labels,
                                                           return_counts=True))}

        # Outliers (fast Z-score)
        z_scores = np.abs((xyz - xyz.mean(0)) / (xyz.std(0) + 1e-9))
        rec.outlier_fraction = float((z_scores > 3.5).any(axis=1).mean())

        return rec

    # ------------------------------------------------------------------
    # Internal: plot helpers
    # ------------------------------------------------------------------

    def _plot_info_text(self, ax, s):
        ax.axis("off")
        lines = [
            f"Total clouds:   {s['total_clouds']:,}",
            f"Valid:          {s['valid_clouds']:,}",
            f"Corrupt:        {s['corrupt_clouds']:,}",
            f"Total points:   {s['total_points']:,}",
            f"Formats:        {s['format_distribution']}",
            f"Has intensity:  {s['has_intensity_count']:,} clouds",
            f"Has RGB:        {s['has_rgb_count']:,} clouds",
            f"Has labels:     {s['has_labels_count']:,} clouds",
            f"Unique labels:  "
            f"{len(s['label_distribution']) if s['label_distribution'] else 'N/A'}",
        ]
        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, color="#e6edf3", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                      edgecolor="#30363d"),
        )
        ax.set_title("Dataset Overview", color="white", fontsize=11)

    def _plot_label_distribution(self, ax, valid):
        labels = [r.label for r in valid if r.label]
        if not labels:
            ax.text(0.5, 0.5, "No labels provided",
                    ha="center", va="center", color="#8b949e",
                    transform=ax.transAxes)
            ax.set_title("Label Distribution", color="white", fontsize=11)
            return
        cnt = Counter(labels).most_common(20)
        names, counts = zip(*cnt)
        y = np.arange(len(names))
        ax.barh(y, counts, color="#58a6ff", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Label Distribution (top 20)", color="white", fontsize=11)
        ax.set_xlabel("Count", color="#8b949e", fontsize=8)

    def _plot_hist(self, ax, data, title, color, bins=25):
        data = [d for d in data if d is not None and np.isfinite(d)]
        if not data:
            ax.set_title(title, color="white", fontsize=9)
            return
        ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(data), color="white", lw=1.2, linestyle="--",
                   alpha=0.7, label=f"μ={np.mean(data):.2f}")
        ax.set_title(title, color="white", fontsize=9)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_format_dist(self, ax, s):
        fmt = s.get("format_distribution", {})
        if not fmt:
            ax.set_title("Format Distribution", color="white", fontsize=9)
            return
        names, counts = zip(*sorted(fmt.items(), key=lambda x: -x[1]))
        y = np.arange(len(names))
        ax.barh(y, counts, color="#e3b341", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("File Format Distribution", color="white", fontsize=9)
        ax.set_xlabel("Count", color="#8b949e", fontsize=8)

    def _plot_intensity_dist(self, ax, valid):
        means = [r.intensity_mean for r in valid if r.has_intensity
                 and r.intensity_mean is not None]
        if not means:
            ax.text(0.5, 0.5, "No intensity data",
                    ha="center", va="center", color="#8b949e",
                    transform=ax.transAxes)
            ax.set_title("Intensity Mean (per cloud)", color="white", fontsize=9)
            return
        ax.hist(means, bins=25, color="#e3b341", alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(means), color="white", lw=1.2, linestyle="--",
                   alpha=0.7, label=f"μ={np.mean(means):.1f}")
        ax.set_title("Intensity Mean (per cloud)", color="white", fontsize=9)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_class_distribution(self, ax, s):
        agg = s.get("aggregated_class_distribution", {})
        if not agg:
            ax.text(0.5, 0.5, "No class labels found",
                    ha="center", va="center", color="#8b949e",
                    transform=ax.transAxes)
            ax.set_title("Aggregated Class Distribution", color="white", fontsize=9)
            return
        items = sorted(agg.items(), key=lambda x: -x[1])[:20]
        cls_ids, counts = zip(*items)
        y = np.arange(len(cls_ids))
        ax.barh(y, counts, color="#58a6ff", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([f"Class {c}" for c in cls_ids], fontsize=7)
        ax.set_title("Aggregated Class Distribution (all clouds)",
                     color="white", fontsize=9)
        ax.set_xlabel("Total points", color="#8b949e", fontsize=8)

    def _plot_npoints_vs_density(self, ax, valid):
        npts    = [r.n_points for r in valid if r.n_points and r.density]
        density = [r.density  for r in valid if r.n_points and r.density]
        if not npts:
            ax.set_title("Points vs Density", color="white", fontsize=9)
            return
        ax.scatter(npts, density, alpha=0.6, s=20, color="#d2a8ff",
                   linewidths=0)
        ax.set_xlabel("Points per cloud", color="#8b949e", fontsize=8)
        ax.set_ylabel("Density (pts/unit³)", color="#8b949e", fontsize=8)
        ax.set_title("Points vs Density", color="white", fontsize=9)
        # log scale if range is large
        if max(npts) / (min(npts) + 1) > 100:
            ax.set_xscale("log")

    # ------------------------------------------------------------------
    # File readers (thin wrappers around the single-cloud reader)
    # ------------------------------------------------------------------

    def _read_cloud(self, path: Path):
        """Read a point cloud file, returning (xyz, intensity, rgb, labels)."""
        suffix = path.suffix.lower()

        if suffix in (".las", ".laz"):
            return self._read_las(path)
        if suffix == ".ply":
            return self._read_ply(path)
        if suffix == ".pcd":
            return self._read_pcd(path)
        if suffix in (".xyz", ".txt"):
            arr = np.loadtxt(str(path))
            return self._parse_array(arr)
        if suffix == ".npy":
            arr = np.load(str(path))
            return self._parse_array(arr)
        if suffix == ".npz":
            data = np.load(str(path))
            key = "points" if "points" in data else list(data.keys())[0]
            return self._parse_array(data[key])
        raise ValueError(f"Unsupported format: {suffix}")

    def _read_las(self, path):
        try:
            import laspy
        except ImportError:
            raise ImportError("pip install laspy[lazrs]")
        las = laspy.read(str(path))
        xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
        intensity = np.array(las.intensity, dtype=np.float32) \
            if hasattr(las, "intensity") else None
        rgb = None
        if hasattr(las, "red"):
            r = np.array(las.red,   dtype=np.float32)
            g = np.array(las.green, dtype=np.float32)
            b = np.array(las.blue,  dtype=np.float32)
            if r.max() > 255:
                r, g, b = r / 256, g / 256, b / 256
            rgb = np.stack([r, g, b], axis=1).astype(np.uint8)
        labels = np.array(las.classification, dtype=np.int32) \
            if hasattr(las, "classification") else None
        return xyz, intensity, rgb, labels

    def _read_ply(self, path):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            xyz = np.asarray(pcd.points, dtype=np.float32)
            rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8) \
                if pcd.has_colors() else None
            return xyz, None, rgb, None
        except ImportError:
            pass
        try:
            from plyfile import PlyData
            data = PlyData.read(str(path))["vertex"]
            xyz = np.stack(
                [data["x"], data["y"], data["z"]], axis=1
            ).astype(np.float32)
            rgb = None
            if "red" in data.dtype.names:
                rgb = np.stack(
                    [data["red"], data["green"], data["blue"]], axis=1
                ).astype(np.uint8)
            return xyz, None, rgb, None
        except ImportError:
            raise ImportError("pip install open3d  OR  pip install plyfile")

    def _read_pcd(self, path):
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("pip install open3d")
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8) \
            if pcd.has_colors() else None
        return xyz, None, rgb, None

    def _parse_array(self, arr):
        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xyz = arr[:, :3]
        intensity = arr[:, 3].copy() if arr.shape[1] > 3 else None
        rgb = arr[:, 4:7].astype(np.uint8) if arr.shape[1] >= 7 else None
        labels = arr[:, -1].astype(np.int32) if arr.shape[1] >= 8 else None
        return xyz, intensity, rgb, labels

    def _resolve_paths(self, source, recursive) -> List[Path]:
        if isinstance(source, (list, tuple)):
            return [Path(p) for p in source]
        source = Path(source)
        if source.is_file():
            return [source]
        return discover_files(source, POINTCLOUD_EXTENSIONS, recursive=recursive)

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first.")

    @staticmethod
    def _plt():
        import matplotlib.pyplot as plt
        return plt

    @staticmethod
    def _mpl():
        import matplotlib as mpl
        return mpl


# ---------------------------------------------------------------------------
def _stat_dict(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {}
    return {
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }