"""
viseda.hyperspectral.dataset_eda
---------------------------------
Dataset-level EDA across a **collection** of hyperspectral cubes.

This complements ``HyperspectralEDA`` (single cube) by treating a directory
or list of cube files as a dataset and computing aggregate statistics across
all cubes — analogous to what ``ImageEDA`` does for image datasets.

Analyses covered
~~~~~~~~~~~~~~~~
* Dataset inventory (count, shapes, band counts, file sizes)
* Cross-cube band-mean spectra (mean ± std envelope across all cubes)
* Per-cube brightness / dynamic-range / SNR distributions
* Band-count and spatial-resolution distributions
* Corrupt / unreadable file detection
* Label distribution (if sub-folder names are used as labels)
* Spectral diversity: pairwise cosine similarity between mean spectra
* Optional per-cube NDVI summary (if wavelengths provided)
* Full matplotlib dashboard
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from viseda.core.base import BaseEDA
from viseda.utils.helpers import (
    HYPERSPECTRAL_EXTENSIONS,
    discover_files,
    safe_divide,
)


# ---------------------------------------------------------------------------
# Per-cube record
# ---------------------------------------------------------------------------

class CubeRecord:
    """Lightweight container for per-cube statistics."""

    __slots__ = (
        "path", "label", "height", "width", "bands",
        "dtype", "file_size_kb",
        "global_mean", "global_std", "global_min", "global_max",
        "band_means",          # (B,) mean reflectance per band
        "band_stds",           # (B,) std per band
        "snr_mean",            # scalar: mean SNR across bands
        "dynamic_range",       # max - min
        "ndvi_mean",           # scalar NDVI mean (if wavelengths provided)
        "is_corrupt",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.is_corrupt = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HyperspectralDatasetEDA(BaseEDA):
    """
    Perform EDA across a **collection** of hyperspectral cubes.

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    wavelengths : array-like, optional
        Shared wavelength axis (nm) for all cubes.
        If supplied, spectral plots use real wavelength values on the x-axis
        and NDVI can be computed automatically.
    max_cubes : int | None
        Cap the number of cubes analysed (useful for large datasets).
    ndvi_nir_band : int | None
        Band index for NIR when computing NDVI. Auto-detected from
        ``wavelengths`` (~850 nm) if not supplied.
    ndvi_red_band : int | None
        Band index for Red when computing NDVI (~670 nm).

    Examples
    --------
    >>> from viseda.hyperspectral import HyperspectralDatasetEDA
    >>> eda = HyperspectralDatasetEDA(wavelengths=np.linspace(400, 2500, 200))
    >>> eda.load("path/to/cubes/")          # directory of .npy / .hdr / .tif
    >>> eda.summary()
    >>> eda.plot()
    """

    def __init__(
        self,
        verbose: bool = True,
        wavelengths: Optional[np.ndarray] = None,
        max_cubes: Optional[int] = None,
        ndvi_nir_band: Optional[int] = None,
        ndvi_red_band: Optional[int] = None,
    ):
        super().__init__(verbose=verbose)
        self.wavelengths = np.asarray(wavelengths) if wavelengths is not None else None
        self.max_cubes = max_cubes
        self.ndvi_nir_band = ndvi_nir_band
        self.ndvi_red_band = ndvi_red_band

        self._records: List[CubeRecord] = []
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
    ) -> "HyperspectralDatasetEDA":
        """
        Load and analyse a collection of hyperspectral cubes.

        Parameters
        ----------
        source
            A directory, a single file, or a list of file paths.
            Supported formats: ``.npy``, ``.npz``, ``.hdr`` (ENVI),
            ``.tif`` / ``.tiff`` (multi-band GeoTIFF), ``.mat`` (MATLAB).
        label_from_parent
            Use the parent folder name as the cube's label.
        labels
            Optional ``{path: label}`` mapping.
        recursive
            Recurse into sub-directories.
        """
        paths = self._resolve_paths(source, recursive)
        if self.max_cubes:
            paths = paths[: self.max_cubes]

        self._label_map: Dict[str, str] = {}
        if labels:
            self._label_map = {str(Path(k).resolve()): v for k, v in labels.items()}

        self._log(f"Found {len(paths)} cube files — computing statistics …")
        self._records = []

        for i, p in enumerate(paths):
            if self.verbose and i % 10 == 0:
                self._log(f"  {i}/{len(paths)}  {p.name}")
            rec = self._analyse_single(p, label_from_parent=label_from_parent)
            self._records.append(rec)

        self._loaded = True
        n_corrupt = sum(r.is_corrupt for r in self._records)
        self._log(
            f"Loaded {len(self._records)} cubes "
            f"({n_corrupt} corrupt / unreadable)."
        )
        return self

    def load_arrays(
        self,
        arrays: List[np.ndarray],
        labels: Optional[List[str]] = None,
    ) -> "HyperspectralDatasetEDA":
        """
        Load a list of NumPy cubes ``(H, W, B)`` directly.

        Parameters
        ----------
        arrays
            List of hyperspectral cubes.
        labels
            Optional label for each cube.
        """
        self._log(f"Loading {len(arrays)} arrays …")
        self._records = []
        for i, arr in enumerate(arrays):
            rec = CubeRecord()
            rec.path = f"array_{i}"
            rec.label = labels[i] if labels and i < len(labels) else None
            self._fill_stats(rec, arr.astype(np.float32))
            self._records.append(rec)
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics across all cubes."""
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]

        heights   = np.array([r.height  for r in valid])
        widths    = np.array([r.width   for r in valid])
        bands_arr = np.array([r.bands   for r in valid])
        means     = np.array([r.global_mean    for r in valid])
        stds      = np.array([r.global_std     for r in valid])
        snrs      = np.array([r.snr_mean       for r in valid])
        dranges   = np.array([r.dynamic_range  for r in valid])
        fsizes    = np.array([r.file_size_kb   for r in valid
                              if r.file_size_kb is not None])

        band_dist = dict(Counter(r.bands for r in valid))
        label_dist = None
        if any(r.label for r in valid):
            label_dist = dict(Counter(r.label for r in valid))

        # Cross-cube mean spectrum (only where band counts match)
        dominant_bands = int(Counter(r.bands for r in valid).most_common(1)[0][0]) \
            if valid else 0
        matching = [r for r in valid if r.bands == dominant_bands
                    and r.band_means is not None]
        cross_mean_spectrum = None
        cross_std_spectrum = None
        if matching:
            stack = np.stack([r.band_means for r in matching], axis=0)
            cross_mean_spectrum = stack.mean(axis=0).tolist()
            cross_std_spectrum  = stack.std(axis=0).tolist()

        result = {
            "total_cubes":   len(self._records),
            "valid_cubes":   len(valid),
            "corrupt_cubes": len(corrupt),
            "corrupt_paths": [r.path for r in corrupt],
            "band_distribution": band_dist,
            "label_distribution": label_dist,
            "height":        _stat_dict(heights),
            "width":         _stat_dict(widths),
            "bands":         _stat_dict(bands_arr),
            "file_size_kb":  _stat_dict(fsizes) if len(fsizes) else {},
            "global_mean":   _stat_dict(means),
            "global_std":    _stat_dict(stds),
            "snr_mean":      _stat_dict(snrs),
            "dynamic_range": _stat_dict(dranges),
            "cross_cube_mean_spectrum": cross_mean_spectrum,
            "cross_cube_std_spectrum":  cross_std_spectrum,
            "dominant_band_count": dominant_bands,
            "n_cubes_matching_dominant": len(matching),
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
            "VisEDA — Hyperspectral Dataset Analysis",
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

        # ── Row 0: overview text + label dist ─────────────────────────
        ax_info = make_ax(gs[0, :2])
        self._plot_info_text(ax_info, s)

        ax_lbl = make_ax(gs[0, 2:])
        self._plot_label_distribution(ax_lbl, valid)

        # ── Row 1: spatial + band distributions ───────────────────────
        ax_h  = make_ax(gs[1, 0])
        ax_w  = make_ax(gs[1, 1])
        ax_b  = make_ax(gs[1, 2])
        ax_fs = make_ax(gs[1, 3])
        self._plot_hist(ax_h,  [r.height for r in valid],
                        "Heights (px)", "#58a6ff")
        self._plot_hist(ax_w,  [r.width  for r in valid],
                        "Widths (px)",  "#3fb950")
        self._plot_hist(ax_b,  [r.bands  for r in valid],
                        "Band Count",   "#d2a8ff")
        self._plot_hist(ax_fs, [r.file_size_kb for r in valid
                                if r.file_size_kb],
                        "File Size (KB)", "#ffa657")

        # ── Row 2: spectral statistics ────────────────────────────────
        ax_spec = make_ax(gs[2, :2])
        self._plot_cross_spectrum(ax_spec, s)

        ax_snr = make_ax(gs[2, 2])
        self._plot_hist(ax_snr, [r.snr_mean for r in valid if r.snr_mean],
                        "Mean SNR (per cube)", "#e3b341")

        ax_dr = make_ax(gs[2, 3])
        self._plot_hist(ax_dr, [r.dynamic_range for r in valid if r.dynamic_range],
                        "Dynamic Range", "#79c0ff")

        # ── Row 3: pixel value distributions + spectral diversity ─────
        ax_mean = make_ax(gs[3, 0])
        ax_std  = make_ax(gs[3, 1])
        self._plot_hist(ax_mean, [r.global_mean for r in valid],
                        "Global Mean (per cube)", "#58a6ff")
        self._plot_hist(ax_std,  [r.global_std  for r in valid],
                        "Global Std (per cube)",  "#f78166")

        ax_div = make_ax(gs[3, 2:])
        self._plot_spectral_diversity(ax_div, valid)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            self._log(f"Dashboard saved → {save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Internal: per-cube analysis
    # ------------------------------------------------------------------

    def _analyse_single(self, path: Path, label_from_parent: bool) -> CubeRecord:
        rec = CubeRecord()
        rec.path = str(path)
        rec.file_size_kb = path.stat().st_size / 1024 if path.exists() else None

        if label_from_parent:
            rec.label = path.parent.name
        else:
            rec.label = self._label_map.get(str(path.resolve()))

        try:
            cube = self._read_cube(path)
        except Exception as e:
            self._log(f"  ✗ Could not read {path.name}: {e}")
            rec.is_corrupt = True
            return rec

        self._fill_stats(rec, cube)
        return rec

    def _fill_stats(self, rec: CubeRecord, cube: np.ndarray) -> None:
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        cube = cube.astype(np.float32)

        rec.height, rec.width, rec.bands = cube.shape
        rec.dtype = str(cube.dtype)

        rec.global_mean = float(cube.mean())
        rec.global_std  = float(cube.std())
        rec.global_min  = float(cube.min())
        rec.global_max  = float(cube.max())
        rec.dynamic_range = rec.global_max - rec.global_min

        band_means = cube.mean(axis=(0, 1))
        band_stds  = cube.std(axis=(0, 1))
        rec.band_means = band_means
        rec.band_stds  = band_stds

        snr = safe_divide(band_means, band_stds + 1e-9, fill=0.0)
        rec.snr_mean = float(snr.mean())

        # Compute per-cube NDVI if wavelengths + band indices available
        if self.wavelengths is not None:
            wl = self.wavelengths
            nir = self.ndvi_nir_band if self.ndvi_nir_band is not None \
                else int(np.argmin(np.abs(wl - 850)))
            red = self.ndvi_red_band if self.ndvi_red_band is not None \
                else int(np.argmin(np.abs(wl - 670)))
            if nir < rec.bands and red < rec.bands:
                nir_b = cube[:, :, nir].astype(np.float32)
                red_b = cube[:, :, red].astype(np.float32)
                ndvi  = safe_divide(nir_b - red_b, nir_b + red_b, fill=0.0)
                rec.ndvi_mean = float(ndvi.mean())
            else:
                rec.ndvi_mean = None
        else:
            rec.ndvi_mean = None

    # ------------------------------------------------------------------
    # Internal: plot helpers
    # ------------------------------------------------------------------

    def _plot_info_text(self, ax, s):
        ax.axis("off")
        lines = [
            f"Total cubes:      {s['total_cubes']:,}",
            f"Valid:            {s['valid_cubes']:,}",
            f"Corrupt:          {s['corrupt_cubes']:,}",
            f"Band counts:      {s['band_distribution']}",
            f"Dominant bands:   {s['dominant_band_count']}",
            f"Unique labels:    "
            f"{len(s['label_distribution']) if s['label_distribution'] else 'N/A'}",
            f"Height median:    {s['height'].get('median', 'N/A')}",
            f"Width median:     {s['width'].get('median', 'N/A')}",
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
        ax.axvline(np.mean(data), color="white", linewidth=1.2,
                   linestyle="--", alpha=0.7,
                   label=f"μ={np.mean(data):.2f}")
        ax.set_title(title, color="white", fontsize=9)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_cross_spectrum(self, ax, s):
        wl = self.wavelengths
        mean_spec = s.get("cross_cube_mean_spectrum")
        std_spec  = s.get("cross_cube_std_spectrum")

        if mean_spec is None:
            ax.text(0.5, 0.5, "No matching-band cubes found",
                    ha="center", va="center", color="#8b949e",
                    transform=ax.transAxes)
            ax.set_title("Cross-Cube Mean Spectrum", color="white", fontsize=9)
            return

        mean_spec = np.array(mean_spec)
        std_spec  = np.array(std_spec)
        x = wl[:len(mean_spec)] if wl is not None else np.arange(len(mean_spec))

        ax.plot(x, mean_spec, color="#58a6ff", lw=1.5, label="Dataset mean")
        ax.fill_between(x, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.25, color="#58a6ff", label="±1σ across cubes")
        ax.set_title(
            f"Cross-Cube Mean Spectrum "
            f"(n={s['n_cubes_matching_dominant']} cubes)",
            color="white", fontsize=9,
        )
        xlabel = "Wavelength (nm)" if wl is not None else "Band index"
        ax.set_xlabel(xlabel, color="#8b949e", fontsize=8)
        ax.set_ylabel("Reflectance", color="#8b949e", fontsize=8)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_spectral_diversity(self, ax, valid):
        """Pairwise cosine similarity heatmap between cube mean spectra."""
        # Use cubes with the dominant band count
        dominant = Counter(r.bands for r in valid).most_common(1)
        if not dominant:
            ax.set_title("Spectral Diversity", color="white", fontsize=9)
            return
        dom_b = dominant[0][0]
        matching = [r for r in valid if r.bands == dom_b
                    and r.band_means is not None][:50]  # cap at 50

        if len(matching) < 2:
            ax.text(0.5, 0.5,
                    "Need ≥2 cubes with same band count\nfor diversity plot",
                    ha="center", va="center", color="#8b949e",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("Spectral Diversity (cosine sim)", color="white", fontsize=9)
            return

        spectra = np.stack([r.band_means for r in matching])  # (N, B)
        norms   = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-9
        normed  = spectra / norms
        sim     = normed @ normed.T  # cosine similarity matrix

        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        self._plt().colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="Cosine sim").ax.tick_params(
            labelcolor="#8b949e", labelsize=7)

        labels = [r.label or Path(r.path).stem[:12] for r in matching]
        if len(labels) <= 20:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right",
                               fontsize=6, color="#8b949e")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6, color="#8b949e")
        ax.set_title(
            f"Spectral Diversity — cosine similarity\n"
            f"(first {len(matching)} cubes with {dom_b} bands)",
            color="white", fontsize=9,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_paths(self, source, recursive) -> List[Path]:
        if isinstance(source, (list, tuple)):
            return [Path(p) for p in source]
        source = Path(source)
        if source.is_file():
            return [source]
        # Also accept .mat files for the dataset scanner
        exts = HYPERSPECTRAL_EXTENSIONS | {".mat"}
        return discover_files(source, exts, recursive=recursive)

    def _read_cube(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(str(path)).astype(np.float32)
        if suffix == ".npz":
            data = np.load(str(path))
            key = list(data.keys())[0]
            return data[key].astype(np.float32)
        if suffix == ".mat":
            import scipy.io
            mat = scipy.io.loadmat(str(path))
            # grab first non-private key
            keys = [k for k in mat if not k.startswith("_")]
            if not keys:
                raise ValueError("No data keys found in .mat file")
            arr = mat[keys[0]]
            return arr.astype(np.float32)
        if suffix in (".hdr", ".bil", ".bip", ".bsq", ".envi"):
            try:
                import spectral
                img = spectral.open_image(str(path))
                return img.load().astype(np.float32)
            except ImportError:
                raise ImportError("pip install spectral")
        if suffix in (".tif", ".tiff"):
            try:
                import rasterio
                with rasterio.open(str(path)) as src:
                    arr = src.read()
                return arr.transpose(1, 2, 0).astype(np.float32)
            except ImportError:
                raise ImportError("pip install rasterio")
        raise ValueError(f"Unsupported format: {suffix}")

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() or .load_arrays() first.")

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