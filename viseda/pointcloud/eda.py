"""
viseda.pointcloud.eda
---------------------
EDA for 3-D point cloud datasets (LiDAR, RGB-D, photogrammetry …).

Analyses covered
~~~~~~~~~~~~~~~~
* Point count, bounding box, centroid, spatial extent
* Density estimation (points per unit volume)
* Height (Z) distribution and profile
* XY density heatmap (top-down bird's-eye view)
* Return / intensity statistics (if available)
* Colour (RGB) statistics for coloured point clouds
* Normal estimation and planarity analysis
* Outlier/noise detection (statistical outlier removal)
* Class-label distribution (if available)
* Subsampling for memory-safe analysis of large clouds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from viseda.core.base import BaseEDA
from viseda.utils.helpers import safe_divide, entropy


class PointCloudEDA(BaseEDA):
    """
    Perform exploratory data analysis on a 3D point cloud.

    Supported input formats
    -----------------------
    * LAS / LAZ files (requires ``laspy`` ≥ 2.0)
    * PLY files (requires ``open3d`` or ``plyfile``)
    * PCD files (requires ``open3d``)
    * XYZ / TXT plain-text files (x y z [i r g b] columns)
    * NumPy arrays of shape (N, 3+)
    * NPZ files with key ``"points"``

    Parameters
    ----------
    verbose : bool
        Print progress.
    max_points : int | None
        Subsample to at most *max_points* for memory-safe analysis.
    random_seed : int
        Seed for subsampling.

    Examples
    --------
    >>> from viseda import PointCloudEDA
    >>> eda = PointCloudEDA(max_points=500_000)
    >>> eda.load("scan.las")
    >>> eda.summary()
    >>> eda.plot()
    """

    def __init__(
        self,
        verbose: bool = True,
        max_points: Optional[int] = 1_000_000,
        random_seed: int = 42,
    ):
        super().__init__(verbose=verbose)
        self.max_points = max_points
        self.random_seed = random_seed

        self._xyz: Optional[np.ndarray] = None         # (N, 3)
        self._intensity: Optional[np.ndarray] = None   # (N,)
        self._rgb: Optional[np.ndarray] = None         # (N, 3) uint8
        self._labels: Optional[np.ndarray] = None      # (N,) int
        self._normals: Optional[np.ndarray] = None     # (N, 3)
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        source: Union[str, Path, np.ndarray],
        intensity: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> "PointCloudEDA":
        """
        Load a point cloud.

        Parameters
        ----------
        source
            File path or NumPy array of shape (N, 3+).
            Column order: X, Y, Z [, Intensity [, R, G, B [, Label]]].
        intensity, rgb, labels
            Override arrays when *source* is an ndarray without these
            fields, or to attach them to any source.
        """
        if isinstance(source, np.ndarray):
            self._xyz, self._intensity, self._rgb, self._labels = \
                self._parse_array(source)
        else:
            self._xyz, self._intensity, self._rgb, self._labels = \
                self._read_file(Path(source))

        # Apply overrides
        if intensity is not None:
            self._intensity = np.asarray(intensity, dtype=np.float32)
        if rgb is not None:
            self._rgb = np.asarray(rgb, dtype=np.uint8)
        if labels is not None:
            self._labels = np.asarray(labels)

        # Subsample
        N = len(self._xyz)
        if self.max_points and N > self.max_points:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(N, self.max_points, replace=False)
            self._xyz = self._xyz[idx]
            if self._intensity is not None:
                self._intensity = self._intensity[idx]
            if self._rgb is not None:
                self._rgb = self._rgb[idx]
            if self._labels is not None:
                self._labels = self._labels[idx]
            self._log(f"Subsampled from {N:,} → {self.max_points:,} points")

        self._loaded = True
        self._log(f"Loaded {len(self._xyz):,} points. "
                  f"Intensity={'yes' if self._intensity is not None else 'no'}, "
                  f"RGB={'yes' if self._rgb is not None else 'no'}, "
                  f"Labels={'yes' if self._labels is not None else 'no'}")
        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        self._check_loaded()
        xyz = self._xyz
        N = len(xyz)

        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        extent = bbox_max - bbox_min
        volume = float(np.prod(np.maximum(extent, 1e-9)))
        density = N / volume

        result = {
            "n_points": N,
            "bounding_box": {
                "min": bbox_min.tolist(),
                "max": bbox_max.tolist(),
                "extent": extent.tolist(),
            },
            "centroid": xyz.mean(axis=0).tolist(),
            "volume_bounding_box": volume,
            "density_pts_per_unit3": density,
            "x": _stat_dict(xyz[:, 0]),
            "y": _stat_dict(xyz[:, 1]),
            "z": _stat_dict(xyz[:, 2]),
            "has_intensity": self._intensity is not None,
            "has_rgb": self._rgb is not None,
            "has_labels": self._labels is not None,
        }

        if self._intensity is not None:
            result["intensity"] = _stat_dict(self._intensity)

        if self._rgb is not None:
            result["rgb_mean"] = self._rgb.mean(axis=0).tolist()
            result["rgb_std"] = self._rgb.std(axis=0).tolist()

        if self._labels is not None:
            from collections import Counter
            result["label_distribution"] = dict(
                Counter(int(l) for l in self._labels))

        # Outlier count (simple Z-score)
        from scipy.spatial.distance import cdist
        z_scores = np.abs((xyz - xyz.mean(0)) / (xyz.std(0) + 1e-9))
        outlier_mask = (z_scores > 3.5).any(axis=1)
        result["estimated_outliers"] = int(outlier_mask.sum())
        result["outlier_fraction"] = float(outlier_mask.mean())

        self._store("summary", result)
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        figsize: Tuple[int, int] = (20, 16),
        save_path: Optional[str] = None,
        dpi: int = 150,
        point_size: float = 0.5,
    ) -> None:
        """Render EDA dashboard for the point cloud."""
        self._check_loaded()
        plt = self._plt()
        mpl = self._mpl()

        xyz = self._xyz

        fig = plt.figure(figsize=figsize, facecolor="#0d1117")
        fig.suptitle("VisEDA — Point Cloud Analysis",
                     fontsize=20, color="white", y=0.98, fontweight="bold")

        gs = mpl.gridspec.GridSpec(3, 4, figure=fig,
                                   hspace=0.45, wspace=0.35,
                                   left=0.05, right=0.97,
                                   top=0.94, bottom=0.04)

        def make_ax(*args, projection=None):
            ax = fig.add_subplot(*args, projection=projection)
            if projection != "3d":
                ax.set_facecolor("#161b22")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#30363d")
            ax.tick_params(colors="#8b949e", labelsize=8)
            return ax

        # ── Top-down (XY) density ─────────────────────────────────────
        ax_xy = make_ax(gs[0, :2])
        self._plot_xy_density(ax_xy, point_size)

        # ── Side view (XZ) ────────────────────────────────────────────
        ax_xz = make_ax(gs[0, 2:])
        self._plot_xz_view(ax_xz, point_size)

        # ── Z height distribution ─────────────────────────────────────
        ax_z = make_ax(gs[1, 0])
        self._plot_histogram_ax(ax_z, xyz[:, 2], "Height (Z) Distribution",
                                "#58a6ff", orientation="horizontal")

        # ── Intensity distribution ────────────────────────────────────
        ax_int = make_ax(gs[1, 1])
        if self._intensity is not None:
            self._plot_histogram_ax(ax_int, self._intensity,
                                    "Intensity Distribution", "#e3b341")
        else:
            ax_int.text(0.5, 0.5, "No intensity data",
                        ha="center", va="center", color="#8b949e",
                        transform=ax_int.transAxes)
            ax_int.set_title("Intensity", color="white", fontsize=9)

        # ── Label distribution ────────────────────────────────────────
        ax_lbl = make_ax(gs[1, 2])
        self._plot_label_dist(ax_lbl)

        # ── Density heatmap ───────────────────────────────────────────
        ax_heat = make_ax(gs[1, 3])
        self._plot_density_heatmap(ax_heat)

        # ── XYZ range box ─────────────────────────────────────────────
        ax_info = make_ax(gs[2, 0])
        self._plot_info_text(ax_info)

        # ── RGB channel hists ─────────────────────────────────────────
        ax_rgb = make_ax(gs[2, 1:3])
        self._plot_rgb_channels(ax_rgb)

        # ── Spatial outlier map ───────────────────────────────────────
        ax_out = make_ax(gs[2, 3])
        self._plot_outlier_map(ax_out)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            self._log(f"Dashboard saved → {save_path}")
        else:
            plt.show()

    def plot_3d(
        self,
        max_display: int = 100_000,
        color_by: str = "z",
        point_size: float = 0.5,
    ) -> None:
        """
        Interactive 3-D scatter plot (requires matplotlib with mpl_toolkits).

        Parameters
        ----------
        color_by
            One of ``'z'``, ``'intensity'``, ``'label'``, ``'rgb'``.
        """
        self._check_loaded()
        plt = self._plt()
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        xyz = self._xyz
        if len(xyz) > max_display:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(xyz), max_display, replace=False)
            xyz = xyz[idx]

        fig = plt.figure(figsize=(10, 8), facecolor="#0d1117")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0d1117")

        if color_by == "z":
            c = xyz[:, 2]
            cmap = "plasma"
        elif color_by == "intensity" and self._intensity is not None:
            c = self._intensity[:len(xyz)]
            cmap = "viridis"
        else:
            c = xyz[:, 2]
            cmap = "plasma"

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=c, cmap=cmap, s=point_size, alpha=0.5)
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.set_title(f"3D Point Cloud (colour={color_by})", color="white")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal plot helpers
    # ------------------------------------------------------------------

    def _plot_xy_density(self, ax, ps):
        plt = self._plt()
        xyz = self._xyz
        c = xyz[:, 2] if self._rgb is None else None
        cmap = "plasma" if c is not None else None

        if self._rgb is not None:
            colors = self._rgb / 255.0
            ax.scatter(xyz[:, 0], xyz[:, 1], c=colors,
                       s=ps, alpha=0.4, linewidths=0)
        else:
            sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=c, cmap=cmap,
                            s=ps, alpha=0.4, linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                         label="Z").ax.tick_params(labelcolor="#8b949e", labelsize=7)

        ax.set_title("Top-down View (XY)", color="white", fontsize=9)
        ax.set_xlabel("X", color="#8b949e", fontsize=8)
        ax.set_ylabel("Y", color="#8b949e", fontsize=8)
        ax.set_aspect("equal")

    def _plot_xz_view(self, ax, ps):
        plt = self._plt()
        xyz = self._xyz
        sc = ax.scatter(xyz[:, 0], xyz[:, 2], c=xyz[:, 1], cmap="viridis",
                        s=ps, alpha=0.4, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                     label="Y").ax.tick_params(labelcolor="#8b949e", labelsize=7)
        ax.set_title("Side View (XZ)", color="white", fontsize=9)
        ax.set_xlabel("X", color="#8b949e", fontsize=8)
        ax.set_ylabel("Z (height)", color="#8b949e", fontsize=8)

    def _plot_histogram_ax(self, ax, data, title, color,
                           orientation="vertical", bins=50):
        data = data[np.isfinite(data)]
        if orientation == "horizontal":
            ax.barh(
                *self._hist_data(data, bins),
                color=color, alpha=0.8, edgecolor="none",
            )
            ax.axhline(np.mean(data), color="white", lw=1, linestyle="--",
                       alpha=0.7, label=f"μ={np.mean(data):.1f}")
        else:
            counts, edges = np.histogram(data, bins=bins)
            centres = (edges[:-1] + edges[1:]) / 2
            ax.bar(centres, counts, width=(edges[1] - edges[0]),
                   color=color, alpha=0.8, edgecolor="none")
            ax.axvline(np.mean(data), color="white", lw=1, linestyle="--",
                       alpha=0.7, label=f"μ={np.mean(data):.1f}")
        ax.set_title(title, color="white", fontsize=9)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    @staticmethod
    def _hist_data(data, bins):
        counts, edges = np.histogram(data, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        return centres, counts

    def _plot_label_dist(self, ax):
        if self._labels is None:
            ax.text(0.5, 0.5, "No labels", ha="center", va="center",
                    color="#8b949e", transform=ax.transAxes)
            ax.set_title("Label Distribution", color="white", fontsize=9)
            return
        from collections import Counter
        cnt = Counter(int(l) for l in self._labels)
        labels, counts = zip(*sorted(cnt.items()))
        ax.barh(np.arange(len(labels)), counts, color="#58a6ff", alpha=0.85)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels([f"Class {l}" for l in labels], fontsize=7)
        ax.set_title("Label Distribution", color="white", fontsize=9)

    def _plot_density_heatmap(self, ax):
        plt = self._plt()
        xyz = self._xyz
        hist, xedges, yedges = np.histogram2d(
            xyz[:, 0], xyz[:, 1], bins=64)
        ax.imshow(hist.T, origin="lower",
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap="hot", aspect="auto")
        ax.set_title("XY Density Heatmap", color="white", fontsize=9)
        ax.set_xlabel("X", color="#8b949e", fontsize=8)
        ax.set_ylabel("Y", color="#8b949e", fontsize=8)

    def _plot_info_text(self, ax):
        ax.axis("off")
        xyz = self._xyz
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        ext = bbox_max - bbox_min
        lines = [
            f"Points:    {len(xyz):,}",
            f"X range:   {bbox_min[0]:.2f} – {bbox_max[0]:.2f}",
            f"Y range:   {bbox_min[1]:.2f} – {bbox_max[1]:.2f}",
            f"Z range:   {bbox_min[2]:.2f} – {bbox_max[2]:.2f}",
            f"Extent:    {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f}",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color="#e6edf3",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                          edgecolor="#30363d"))
        ax.set_title("Cloud Info", color="white", fontsize=9)

    def _plot_rgb_channels(self, ax):
        if self._rgb is None:
            ax.text(0.5, 0.5, "No RGB data", ha="center", va="center",
                    color="#8b949e", transform=ax.transAxes)
            ax.set_title("RGB Channels", color="white", fontsize=9)
            return
        colors = ["#ff6b6b", "#51cf66", "#339af0"]
        names = ["Red", "Green", "Blue"]
        for i, (c, n) in enumerate(zip(colors, names)):
            ax.hist(self._rgb[:, i], bins=50, color=c, alpha=0.55,
                    label=n, edgecolor="none")
        ax.set_title("RGB Channel Distribution", color="white", fontsize=9)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_outlier_map(self, ax):
        xyz = self._xyz
        z_scores = np.abs((xyz - xyz.mean(0)) / (xyz.std(0) + 1e-9))
        outlier_mask = (z_scores > 3.5).any(axis=1)
        normal = xyz[~outlier_mask]
        outliers = xyz[outlier_mask]
        ax.scatter(normal[:, 0], normal[:, 1], c="#3fb950",
                   s=0.3, alpha=0.3, label="Normal", linewidths=0)
        ax.scatter(outliers[:, 0], outliers[:, 1], c="#f78166",
                   s=2, alpha=0.8, label=f"Outliers ({len(outliers):,})",
                   linewidths=0)
        ax.set_title("Spatial Outliers (Z>3.5σ)", color="white", fontsize=9)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")
        ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    def _read_file(
        self, path: Path
    ) -> Tuple[np.ndarray, Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray]]:
        suffix = path.suffix.lower()

        if suffix in (".las", ".laz"):
            return self._read_las(path)
        if suffix == ".ply":
            return self._read_ply(path)
        if suffix == ".pcd":
            return self._read_pcd(path)
        if suffix in (".xyz", ".txt"):
            return self._read_xyz_txt(path)
        if suffix == ".npy":
            arr = np.load(str(path))
            return self._parse_array(arr)
        if suffix == ".npz":
            data = np.load(str(path))
            key = "points" if "points" in data else list(data.keys())[0]
            return self._parse_array(data[key])
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    def _read_las(self, path: Path):
        try:
            import laspy
        except ImportError:
            raise ImportError("Install laspy: pip install laspy[lazrs]")
        las = laspy.read(str(path))
        xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
        intensity = np.array(las.intensity, dtype=np.float32) \
            if hasattr(las, "intensity") else None
        rgb = None
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            r = np.array(las.red, dtype=np.float32)
            g = np.array(las.green, dtype=np.float32)
            b = np.array(las.blue, dtype=np.float32)
            # LAS stores 16-bit colour
            scale = r.max()
            if scale > 255:
                r, g, b = r / 256, g / 256, b / 256
            rgb = np.stack([r, g, b], axis=1).astype(np.uint8)
        labels = np.array(las.classification, dtype=np.int32) \
            if hasattr(las, "classification") else None
        return xyz, intensity, rgb, labels

    def _read_ply(self, path: Path):
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
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
            rgb = None
            if "red" in data.dtype.names:
                rgb = np.stack(
                    [data["red"], data["green"], data["blue"]], axis=1
                ).astype(np.uint8)
            return xyz, None, rgb, None
        except ImportError:
            raise ImportError(
                "Install open3d or plyfile to read PLY: "
                "pip install open3d  OR  pip install plyfile"
            )

    def _read_pcd(self, path: Path):
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Install open3d: pip install open3d")
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8) \
            if pcd.has_colors() else None
        return xyz, None, rgb, None

    def _read_xyz_txt(self, path: Path):
        arr = np.loadtxt(str(path))
        return self._parse_array(arr)

    def _parse_array(self, arr: np.ndarray):
        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xyz = arr[:, :3]
        intensity = arr[:, 3].copy() if arr.shape[1] > 3 else None
        rgb = None
        if arr.shape[1] >= 7:
            rgb = arr[:, 4:7].astype(np.uint8)
        labels = arr[:, -1].astype(np.int32) \
            if arr.shape[1] >= 8 else None
        return xyz, intensity, rgb, labels

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

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


# helpers
def _stat_dict(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }