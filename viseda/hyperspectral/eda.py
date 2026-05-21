"""
viseda.hyperspectral.eda
========================
Comprehensive EDA for hyperspectral / multispectral image datasets.

Supports analysing a **single cube** or a **whole directory** of cubes
through one unified class — ``HyperspectralEDA``.

Supported file formats
----------------------
* ``.mat``   — MATLAB (scipy.io)            e.g. Indian Pines, Pavia, Salinas
* ``.npy``   — NumPy array  (H × W × B)
* ``.npz``   — NumPy archive (first key)
* ``.hdr``   — ENVI header + sidecar        requires ``pip install spectral``
* ``.tif / .tiff`` — Multi-band GeoTIFF     requires ``pip install rasterio``

Cube layout
-----------
All cubes are expected / normalised to shape ``(H, W, B)`` — height × width
× bands.  Single-band images are treated as ``(H, W, 1)``.

Analyses — per cube
-------------------
INVENTORY
    shape, dtype, file size, label (from folder name or user dict)

SPECTRAL STATISTICS  (computed per band, stored as arrays of length B)
    mean, std, min, max, SNR (mean/std), noise (MAD),
    saturation fraction, dynamic range per band

SPATIAL STATISTICS
    spatial mean map (H × W average across bands)
    spatial std map

SPECTRAL INDICES  (when wavelengths are provided)
    NDVI  (NIR ~850 nm, Red ~670 nm)
    NDWI  (Green ~560 nm, NIR ~850 nm)
    EVI   (NIR, Red, Blue ~490 nm)
    SAVI  (soil-adjusted vegetation index)

TEXTURE
    GLCM on the first principal component image:
    contrast, dissimilarity, homogeneity, energy, correlation, ASM

DIMENSIONALITY
    PCA: explained variance per component (up to 20 components)

QUALITY
    band dropout detection (bands with near-zero variance)
    spectral smoothness (mean absolute difference between adjacent bands)
    inter-band correlation (mean off-diagonal correlation)

Analyses — dataset level (multiple cubes)
-----------------------------------------
    cross-cube mean / std spectrum (± envelope)
    spectral diversity (pairwise cosine similarity)
    distribution of per-cube scalar metrics
        (brightness, contrast, SNR, NDVI mean, dynamic range …)
    label / class distribution
    band-count distribution
    spatial size distribution
    corrupt-file detection

Visualisations
--------------
    plot()               — single-cube deep-dive dashboard
    plot_dataset()       — dataset-level aggregate dashboard
    plot_spectra()       — mean spectra overlay for all cubes
    plot_false_colour()  — false-colour previews (grid)
    plot_ndvi()          — NDVI maps grid
    plot_pca_components()— first N PC images
    plot_band_stats()    — per-band stats line chart
    plot_spectral_diversity() — pairwise cosine similarity heatmap
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ── lazy imports ─────────────────────────────────────────────────────────────
def _plt():
    import matplotlib.pyplot as plt; return plt

def _mpl():
    import matplotlib as mpl; return mpl

def _ski():
    from skimage.feature import graycomatrix, graycoprops
    return graycomatrix, graycoprops


# ═════════════════════════════════════════════════════════════════════════════
# Per-cube record
# ═════════════════════════════════════════════════════════════════════════════

class CubeRecord:
    """All per-cube statistics in one lightweight object."""

    __slots__ = (
        # identity
        "path", "label", "file_ext", "file_size_kb",
        # shape
        "height", "width", "bands", "dtype",
        # global scalars
        "global_mean", "global_std", "global_min", "global_max",
        "dynamic_range",
        # per-band arrays  (length B each)
        "band_means", "band_stds", "band_mins", "band_maxs",
        "band_snr", "band_noise_mad", "band_saturation_frac",
        # spectral quality
        "spectral_smoothness",    # mean |diff| between adjacent bands
        "inter_band_corr",        # mean off-diagonal correlation
        "n_dropout_bands",        # bands with std < dropout_thresh
        "snr_mean",               # scalar: mean SNR across bands
        # spatial
        "spatial_mean_map",       # (H, W) mean across bands
        "spatial_std_map",        # (H, W) std across bands
        # spectral indices
        "ndvi_mean", "ndvi_std",
        "ndwi_mean", "ndwi_std",
        "evi_mean",  "evi_std",
        "savi_mean", "savi_std",
        # PCA
        "pca_variance_ratio",     # array of length min(20, B)
        # texture (on first PC)
        "glcm_contrast", "glcm_dissimilarity",
        "glcm_homogeneity", "glcm_energy",
        "glcm_correlation", "glcm_asm",
        # status
        "is_corrupt",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.is_corrupt = False


# ═════════════════════════════════════════════════════════════════════════════
# Main class
# ═════════════════════════════════════════════════════════════════════════════

class HyperspectralEDA:
    """
    Comprehensive EDA for hyperspectral / multispectral datasets.

    Works for a **single cube** or a **directory / list of cubes**.

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    wavelengths : array-like, optional
        1-D array of wavelength values (nm) — one per band.
        Enables real-axis spectral plots and automatic index computation.
    max_cubes : int | None
        Analyse at most *max_cubes* (useful for large datasets).
    compute_glcm : bool
        Compute GLCM texture on the first PC image (default True).
        Requires ``scikit-image``.
    compute_pca : bool
        Compute PCA variance profile (default True).
        Requires ``scikit-learn``.
    dropout_threshold : float
        Bands whose std is below this fraction of the global std are
        flagged as "dropout" bands (default 0.01).
    ndvi_nir_band / ndvi_red_band : int | None
        Override auto-detected band indices for spectral index computation.

    Examples
    --------
    Single cube from file
    >>>  eda = HyperspectralEDA(wavelengths=np.linspace(400,2500,200))
    >>>  eda.load("Indian_pines_corrected.mat")
    >>>  eda.summary()
    >>>  eda.plot()

    Whole dataset from directory
    >>>  eda = HyperspectralEDA(wavelengths=np.linspace(400,2500,200))
    >>>  eda.load("path/to/cubes/", label_from_parent=True)
    >>>  eda.summary()
    >>>  eda.plot_dataset()
    >>>  eda.plot_spectra()

    NumPy arrays directly
    >>>  eda = HyperspectralEDA()
    >>>  eda.load_arrays([cube1, cube2, cube3], labels=["A","B","C"])
    >>>  eda.plot_dataset()
    """

    SUPPORTED_EXTS = {".mat", ".npy", ".npz", ".hdr", ".bil", ".bip",
                      ".bsq", ".envi", ".tif", ".tiff"}

    def __init__(
        self,
        verbose: bool = True,
        wavelengths: Optional[np.ndarray] = None,
        max_cubes: Optional[int] = None,
        compute_glcm: bool = True,
        compute_pca: bool = True,
        dropout_threshold: float = 0.01,
        ndvi_nir_band: Optional[int] = None,
        ndvi_red_band: Optional[int] = None,
        ndwi_green_band: Optional[int] = None,
        ndwi_nir_band: Optional[int] = None,
    ):
        self.verbose          = verbose
        self.wavelengths      = np.asarray(wavelengths) if wavelengths is not None else None
        self.max_cubes        = max_cubes
        self.compute_glcm     = compute_glcm
        self.compute_pca      = compute_pca
        self.dropout_threshold = dropout_threshold
        self._ndvi_nir        = ndvi_nir_band
        self._ndvi_red        = ndvi_red_band
        self._ndwi_green      = ndwi_green_band
        self._ndwi_nir        = ndwi_nir_band

        self._records: List[CubeRecord] = []
        self._label_map: Dict[str, str] = {}
        self._loaded = False
        self._results: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────

    def load(
        self,
        source: Union[str, Path, List],
        labels: Optional[Dict[str, str]] = None,
        label_from_parent: bool = False,
        recursive: bool = True,
    ) -> "HyperspectralEDA":
        """
        Load one or many hyperspectral cubes from files.

        Parameters
        ----------
        source
            A directory, a single file, or a list of file paths.
        labels
            ``{path: label}`` mapping.
        label_from_parent
            Use the parent folder name as the cube's label.
        recursive
            Recurse into sub-directories when *source* is a folder.
        """
        paths = self._resolve_paths(source, recursive)
        if self.max_cubes:
            paths = paths[: self.max_cubes]

        if labels:
            self._label_map = {str(Path(k).resolve()): v
                               for k, v in labels.items()}

        self._log(f"Found {len(paths)} cube file(s) — computing statistics …")
        self._records = []

        for i, p in enumerate(paths):
            if self.verbose and i % max(1, len(paths) // 20) == 0:
                self._log(f"  [{i:>{len(str(len(paths)))}}/{len(paths)}] {p.name}")
            rec = self._analyse_file(p, label_from_parent)
            self._records.append(rec)

        self._loaded = True
        n_bad = sum(r.is_corrupt for r in self._records)
        self._log(f"Done. {len(self._records)} cube(s) loaded ({n_bad} corrupt).")
        return self

    def load_arrays(
        self,
        arrays: List[np.ndarray],
        labels: Optional[List[str]] = None,
    ) -> "HyperspectralEDA":
        """
        Load cubes directly as NumPy arrays of shape ``(H, W, B)``.

        Parameters
        ----------
        arrays
            List of hyperspectral cubes.
        labels
            Optional label for each cube.
        """
        self._log(f"Loading {len(arrays)} array(s) …")
        self._records = []
        for i, arr in enumerate(arrays):
            rec = CubeRecord()
            rec.path    = f"<array_{i}>"
            rec.label   = labels[i] if labels and i < len(labels) else None
            rec.file_ext = "array"
            try:
                self._fill_stats(rec, arr.astype(np.float32))
            except Exception as e:
                rec.is_corrupt = True
                self._log(f"  ✗ array_{i}: {e}")
            self._records.append(rec)
        self._loaded = True
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive summary dictionary.

        Sections: inventory, spatial, spectral_stats, spectral_quality,
                  spectral_indices, texture, pca, dataset_stats, labels
        """
        self._check_loaded()
        valid   = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]

        if not valid:
            return {"error": "No valid cubes found."}

        def arr(attr):
            return np.array([getattr(r, attr) for r in valid
                             if getattr(r, attr) is not None])

        # ── inventory ────────────────────────────────────────────────
        band_dist  = dict(Counter(r.bands    for r in valid))
        label_dist = None
        if any(r.label for r in valid):
            lc = Counter(r.label for r in valid)
            label_dist = dict(lc)

        # ── cross-cube mean spectrum ──────────────────────────────────
        dom_bands = int(Counter(r.bands for r in valid).most_common(1)[0][0])
        matching  = [r for r in valid if r.bands == dom_bands
                     and r.band_means is not None]
        cross_mean = None; cross_std = None
        if matching:
            stack = np.stack([r.band_means for r in matching])
            cross_mean = stack.mean(axis=0).tolist()
            cross_std  = stack.std(axis=0).tolist()

        # ── per-band stats averaged across dataset ────────────────────
        if matching:
            snr_stack  = np.stack([r.band_snr  for r in matching
                                   if r.band_snr  is not None])
            noise_stack= np.stack([r.band_noise_mad for r in matching
                                   if r.band_noise_mad is not None])
            mean_snr_per_band  = snr_stack.mean(axis=0).tolist()
            mean_noise_per_band= noise_stack.mean(axis=0).tolist()
        else:
            mean_snr_per_band  = []
            mean_noise_per_band = []

        # ── spectral index summary ────────────────────────────────────
        indices = {}
        for idx_name in ("ndvi", "ndwi", "evi", "savi"):
            means = arr(f"{idx_name}_mean")
            stds  = arr(f"{idx_name}_std")
            if len(means):
                indices[idx_name] = {
                    "per_cube_mean": _stat_dict(means),
                    "per_cube_std":  _stat_dict(stds),
                }

        # ── texture summary ───────────────────────────────────────────
        texture = {}
        for feat in ("glcm_contrast","glcm_dissimilarity","glcm_homogeneity",
                     "glcm_energy","glcm_correlation","glcm_asm"):
            a = arr(feat)
            if len(a):
                texture[feat] = _stat_dict(a)

        # ── PCA summary ───────────────────────────────────────────────
        pca_vars = [r.pca_variance_ratio for r in valid
                    if r.pca_variance_ratio is not None]
        pca_summary = {}
        if pca_vars:
            min_len   = min(len(v) for v in pca_vars)
            pca_stack = np.stack([v[:min_len] for v in pca_vars])
            pca_summary = {
                "mean_variance_ratio": pca_stack.mean(axis=0).tolist(),
                "n_components_95pct_mean": int(np.searchsorted(
                    pca_stack.mean(axis=0).cumsum(), 0.95) + 1),
            }

        result = {
            "inventory": {
                "total_cubes":   len(self._records),
                "valid_cubes":   len(valid),
                "corrupt_cubes": len(corrupt),
                "corrupt_paths": [r.path for r in corrupt],
                "band_distribution": band_dist,
                "format_distribution": dict(Counter(
                    r.file_ext for r in valid)),
                "label_distribution": label_dist,
            },
            "spatial": {
                "height":      _stat_dict(arr("height")),
                "width":       _stat_dict(arr("width")),
                "bands":       _stat_dict(np.array([r.bands for r in valid])),
                "file_size_kb": _stat_dict(arr("file_size_kb")),
            },
            "spectral_stats": {
                "global_mean":     _stat_dict(arr("global_mean")),
                "global_std":      _stat_dict(arr("global_std")),
                "dynamic_range":   _stat_dict(arr("dynamic_range")),
                "cross_cube_mean_spectrum": cross_mean,
                "cross_cube_std_spectrum":  cross_std,
                "dataset_mean_snr_per_band":   mean_snr_per_band,
                "dataset_mean_noise_per_band":  mean_noise_per_band,
                "dominant_band_count": dom_bands,
                "n_matching_cubes":    len(matching),
            },
            "spectral_quality": {
                "snr_mean":           _stat_dict(arr("snr_mean")),
                "spectral_smoothness":_stat_dict(arr("spectral_smoothness")),
                "inter_band_corr":    _stat_dict(arr("inter_band_corr")),
                "n_dropout_bands":    _stat_dict(arr("n_dropout_bands")),
            },
            "spectral_indices": indices,
            "texture":   texture,
            "pca":       pca_summary,
            "labels": {
                "label_distribution": label_dist,
                "class_imbalance_ratio": (
                    round(max(Counter(r.label for r in valid).values()) /
                          max(min(Counter(r.label for r in valid).values()), 1), 3)
                    if label_dist and len(label_dist) > 1 else None
                ),
            },
        }
        self._results["summary"] = result
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Single-cube accessors
    # ─────────────────────────────────────────────────────────────────────

    def get_record(self, index: int = 0) -> CubeRecord:
        """Return the CubeRecord for cube at *index*."""
        self._check_loaded()
        return self._records[index]

    def spectral_signature(
        self, cube_index: int = 0, row: int = 0, col: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (wavelengths, reflectance) for a single pixel in a cube.

        Parameters
        ----------
        cube_index : int
            Which loaded cube to use.
        row, col : int
            Pixel coordinates within that cube.
        """
        self._check_loaded()
        rec = self._records[cube_index]
        if rec.is_corrupt or rec.path.startswith("<array"):
            raise ValueError("Cannot read pixel from corrupt or in-memory cube.")
        cube = self._read_cube(Path(rec.path))
        wl   = (self.wavelengths if self.wavelengths is not None
                else np.arange(cube.shape[2]))
        return wl, cube[row, col, :].copy()

    def compute_index(
        self,
        cube_index: int = 0,
        index_name: str = "ndvi",
    ) -> np.ndarray:
        """
        Compute a spectral index map for a single cube.

        Parameters
        ----------
        index_name : ``"ndvi"`` | ``"ndwi"`` | ``"evi"`` | ``"savi"``
        """
        self._check_loaded()
        rec = self._records[cube_index]
        cube = self._load_cube_array(rec)
        return self._compute_index(cube, index_name)

    def pca_scores(
        self,
        cube_index: int = 0,
        n_components: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PCA on a single cube's spectral dimension.

        Returns
        -------
        scores : ndarray (H*W, n_components)
        variance_ratio : ndarray (n_components,)
        """
        self._check_loaded()
        from sklearn.decomposition import PCA
        rec  = self._records[cube_index]
        cube = self._load_cube_array(rec)
        H, W, B = cube.shape
        X   = cube.reshape(-1, B)
        pca = PCA(n_components=min(n_components, B), svd_solver="randomized")
        scores = pca.fit_transform(X)
        return scores, pca.explained_variance_ratio_

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — single cube deep-dive
    # ─────────────────────────────────────────────────────────────────────

    def plot(
        self,
        cube_index: int = 0,
        figsize: Tuple[int, int] = (22, 20),
        save_path: Optional[str] = None,
        dpi: int = 150,
        rgb_bands: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Single-cube deep-dive dashboard (5 rows × 3 cols).

        Parameters
        ----------
        cube_index
            Which loaded cube to display (default 0).
        rgb_bands
            Band indices for false-colour preview (R, G, B).
            Auto-selected if not provided.
        """
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        rec  = self._records[cube_index]
        if rec.is_corrupt:
            self._log(f"Cube {cube_index} is corrupt — cannot plot.")
            return

        cube = self._load_cube_array(rec)
        H, W, B = cube.shape
        wl = self.wavelengths if self.wavelengths is not None else np.arange(B)

        if rgb_bands is None:
            step = max(1, B // 3)
            rgb_bands = (min(2 * step, B-1), min(step, B-1), 0)

        fig = plt.figure(figsize=figsize, facecolor="white")
        title = (f"HyperspectralEDA — {Path(rec.path).name}"
                 if not rec.path.startswith("<array") else
                 f"HyperspectralEDA — {rec.path}")
        fig.suptitle(title, fontsize=18, color="#1f2328",
                     y=0.99, fontweight="bold")

        gs = mpl.gridspec.GridSpec(5, 3, figure=fig,
                                   hspace=0.55, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.96, bottom=0.03)

        def ax(*args, **kw):
            a = fig.add_subplot(*args, **kw)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        # ── Row 0: info card + false colour + spatial mean ────────────
        self._plot_cube_info(ax(gs[0, 0]), rec, cube)
        self._plot_false_colour(ax(gs[0, 1]), cube, rgb_bands,
                                title="False Colour (R/G/B bands)")
        self._plot_spatial_mean(ax(gs[0, 2]), rec)

        # ── Row 1: mean spectrum + band means + band stds ─────────────
        self._plot_mean_spectrum_single(ax(gs[1, :2]), rec, wl)
        self._plot_band_snr(ax(gs[1, 2]), rec, wl)

        # ── Row 2: band stats ─────────────────────────────────────────
        self._plot_band_stats_lines(ax(gs[2, :]), rec, wl)

        # ── Row 3: spectral indices ───────────────────────────────────
        idx_names = ["ndvi", "ndwi", "evi", "savi"]
        for col, name in enumerate(idx_names[:3]):
            self._plot_index_map_single(ax(gs[3, col]), cube, name)

        # ── Row 4: PCA variance + GLCM radar + dropout ───────────────
        self._plot_pca_variance_single(ax(gs[4, 0]), rec)
        self._plot_glcm_radar(ax(gs[4, 1]), [rec])
        self._plot_spectral_quality_bars(ax(gs[4, 2]), [rec])

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — dataset-level dashboard
    # ─────────────────────────────────────────────────────────────────────

    def plot_dataset(
        self,
        figsize: Tuple[int, int] = (24, 24),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """
        Dataset-level aggregate dashboard (6 rows × 4 cols).
        Shows distributions across ALL loaded cubes.
        """
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        valid = [r for r in self._records if not r.is_corrupt]

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("HyperspectralEDA — Dataset Analysis",
                     fontsize=20, color="#1f2328",
                     y=0.99, fontweight="bold")

        gs = mpl.gridspec.GridSpec(6, 4, figure=fig,
                                   hspace=0.55, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.96, bottom=0.02)

        def ax(*args, **kw):
            a = fig.add_subplot(*args, **kw)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        s = self.summary()

        # ── Row 0: overview card + label dist ─────────────────────────
        self._plot_dataset_info(ax(gs[0, :2]), valid, s)
        self._plot_label_dist(ax(gs[0, 2:]), valid)

        # ── Row 1: spatial & band distributions ───────────────────────
        self._plot_hist(ax(gs[1, 0]), [r.height for r in valid],
                        "Heights (px)", "#58a6ff")
        self._plot_hist(ax(gs[1, 1]), [r.width  for r in valid],
                        "Widths (px)",  "#3fb950")
        self._plot_hist(ax(gs[1, 2]), [r.bands  for r in valid],
                        "Band Count",   "#d2a8ff")
        self._plot_hist(ax(gs[1, 3]),
                        [r.file_size_kb for r in valid if r.file_size_kb],
                        "File Size (KB)", "#ffa657")

        # ── Row 2: cross-cube mean spectrum ───────────────────────────
        self._plot_cross_spectrum(ax(gs[2, :3]), s)
        self._plot_band_count_dist(ax(gs[2, 3]), valid)

        # ── Row 3: scalar quality distributions ───────────────────────
        self._plot_hist(ax(gs[3, 0]),
                        [r.snr_mean for r in valid if r.snr_mean],
                        "Mean SNR (per cube)", "#e3b341")
        self._plot_hist(ax(gs[3, 1]),
                        [r.dynamic_range for r in valid if r.dynamic_range],
                        "Dynamic Range", "#79c0ff")
        self._plot_hist(ax(gs[3, 2]),
                        [r.spectral_smoothness for r in valid
                         if r.spectral_smoothness],
                        "Spectral Smoothness", "#56d364")
        self._plot_hist(ax(gs[3, 3]),
                        [r.inter_band_corr for r in valid
                         if r.inter_band_corr],
                        "Inter-band Correlation", "#f78166")

        # ── Row 4: spectral indices distributions ─────────────────────
        idx_colors = {"ndvi": "#3fb950", "ndwi": "#58a6ff",
                      "evi":  "#e3b341", "savi": "#f78166"}
        for col, (name, color) in enumerate(idx_colors.items()):
            vals = [getattr(r, f"{name}_mean") for r in valid
                    if getattr(r, f"{name}_mean") is not None]
            self._plot_hist(ax(gs[4, col]), vals,
                            f"{name.upper()} Mean (per cube)", color)

        # ── Row 5: spectral diversity + PCA + dropout ─────────────────
        self._plot_spectral_diversity(ax(gs[5, :2]), valid)
        self._plot_pca_variance_dataset(ax(gs[5, 2]), valid)
        self._plot_hist(ax(gs[5, 3]),
                        [r.n_dropout_bands for r in valid
                         if r.n_dropout_bands is not None],
                        "Dropout Bands (per cube)", "#ff6b6b")

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — spectra overlay
    # ─────────────────────────────────────────────────────────────────────

    def plot_spectra(
        self,
        max_cubes: int = 20,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """
        Overlay mean spectra for all (or up to *max_cubes*) loaded cubes.
        Each curve is coloured by label if labels are available.
        """
        self._check_loaded()
        plt = _plt()
        valid = [r for r in self._records if not r.is_corrupt
                 and r.band_means is not None][:max_cubes]

        dom_bands = Counter(r.bands for r in valid).most_common(1)[0][0]
        matching  = [r for r in valid if r.bands == dom_bands]
        wl = (self.wavelengths[:dom_bands]
              if self.wavelengths is not None else np.arange(dom_bands))

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("#f6f8fa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d0d7de")
        ax.tick_params(colors="#57606a", labelsize=8)

        labels_present = any(r.label for r in matching)
        all_labels = sorted(set(r.label for r in matching if r.label))
        cmap = plt.cm.get_cmap("tab10", max(len(all_labels), 1))
        label_color = {lbl: cmap(i) for i, lbl in enumerate(all_labels)}
        legend_handles = {}

        for rec in matching:
            color = label_color.get(rec.label, "#57606a") \
                    if labels_present else "#58a6ff"
            lbl = rec.label or Path(rec.path).stem[:20]
            line, = ax.plot(wl, rec.band_means, lw=1, alpha=0.7,
                            color=color, label=lbl)
            if rec.label and rec.label not in legend_handles:
                legend_handles[rec.label] = line

        if legend_handles:
            ax.legend(legend_handles.values(), legend_handles.keys(),
                      fontsize=7, labelcolor="#1f2328",
                      facecolor="white", edgecolor="#d0d7de",
                      loc="upper right")

        ax.set_title(f"Mean Spectral Signatures ({len(matching)} cubes)",
                     color="#1f2328", fontsize=12)
        xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
        ax.set_xlabel(xlabel, color="#57606a", fontsize=9)
        ax.set_ylabel("Mean Reflectance", color="#57606a", fontsize=9)
        fig.suptitle("HyperspectralEDA — Spectral Overlay",
                     color="#1f2328", fontsize=14, fontweight="bold")
        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — false colour grid
    # ─────────────────────────────────────────────────────────────────────

    def plot_false_colour(
        self,
        n: int = 12,
        cols: int = 4,
        rgb_bands: Optional[Tuple[int, int, int]] = None,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Grid of false-colour previews for up to *n* cubes."""
        self._check_loaded()
        plt = _plt()
        valid = [r for r in self._records if not r.is_corrupt][:n]
        rows  = int(np.ceil(len(valid) / cols))
        figsize = figsize or (cols * 4, rows * 3.5)

        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                 facecolor="white", squeeze=False)

        for i, rec in enumerate(valid):
            ax  = axes[i // cols][i % cols]
            ax.axis("off")
            try:
                cube = self._load_cube_array(rec)
                B    = cube.shape[2]
                if rgb_bands is None:
                    step = max(1, B // 3)
                    rb   = (min(2*step, B-1), min(step, B-1), 0)
                else:
                    rb = rgb_bands
                fc = cube[:, :, list(rb)].astype(np.float32)
                fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-9)
                ax.imshow(np.clip(fc, 0, 1))
                title = (rec.label or Path(rec.path).stem)[:22]
                ax.set_title(title, fontsize=7, color="#1f2328")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center",
                        fontsize=7, color="#f78166",
                        transform=ax.transAxes)

        for j in range(len(valid), rows * cols):
            axes[j // cols][j % cols].axis("off")

        fig.suptitle("False Colour Previews", color="#1f2328",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — NDVI grid
    # ─────────────────────────────────────────────────────────────────────

    def plot_ndvi(
        self,
        n: int = 12,
        cols: int = 4,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Grid of NDVI maps for up to *n* cubes."""
        self._check_loaded()
        plt = _plt()
        valid = [r for r in self._records if not r.is_corrupt][:n]
        rows  = int(np.ceil(len(valid) / cols))
        figsize = figsize or (cols * 4, rows * 3.5)

        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                 facecolor="white", squeeze=False)

        for i, rec in enumerate(valid):
            ax = axes[i // cols][i % cols]
            ax.axis("off")
            try:
                cube = self._load_cube_array(rec)
                ndvi = self._compute_index(cube, "ndvi")
                im   = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             ).ax.tick_params(labelsize=6)
                title = (rec.label or Path(rec.path).stem)[:22]
                ax.set_title(
                    f"{title}\nμ={rec.ndvi_mean:.3f}" if rec.ndvi_mean else title,
                    fontsize=7, color="#1f2328")
            except Exception as e:
                ax.text(0.5, 0.5, f"No NDVI\n{e}", ha="center", va="center",
                        fontsize=7, color="#f78166", transform=ax.transAxes)

        for j in range(len(valid), rows * cols):
            axes[j // cols][j % cols].axis("off")

        fig.suptitle("NDVI Maps", color="#1f2328",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — PCA component images
    # ─────────────────────────────────────────────────────────────────────

    def plot_pca_components(
        self,
        cube_index: int = 0,
        n_components: int = 6,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Visualise the first *n_components* PCA component images."""
        self._check_loaded()
        plt = _plt()
        rec  = self._records[cube_index]
        cube = self._load_cube_array(rec)
        H, W, B = cube.shape
        scores, var = self.pca_scores(cube_index, n_components)

        cols = min(n_components, 3)
        rows = int(np.ceil(n_components / cols))
        figsize = figsize or (cols * 4, rows * 3.5 + 1)

        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                 facecolor="white", squeeze=False)

        for i in range(n_components):
            comp_img = scores[:, i].reshape(H, W)
            ax = axes[i // cols][i % cols]
            ax.axis("off")
            im = ax.imshow(comp_img, cmap="RdBu_r")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04
                         ).ax.tick_params(labelsize=6)
            ax.set_title(f"PC{i+1}  ({var[i]*100:.1f}%)",
                         fontsize=8, color="#1f2328")

        for j in range(n_components, rows * cols):
            axes[j // cols][j % cols].axis("off")

        title = (rec.label or Path(rec.path).stem)[:30]
        fig.suptitle(f"PCA Components — {title}",
                     color="#1f2328", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — band statistics line chart
    # ─────────────────────────────────────────────────────────────────────

    def plot_band_stats(
        self,
        cube_index: int = 0,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Per-band statistics line chart for a single cube."""
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        rec = self._records[cube_index]
        if rec.is_corrupt or rec.band_means is None:
            self._log("No band stats available.")
            return

        wl = (self.wavelengths[:rec.bands]
              if self.wavelengths is not None else np.arange(rec.bands))

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle(
            f"Per-band Statistics — "
            f"{Path(rec.path).name if not rec.path.startswith('<') else rec.path}",
            color="#1f2328", fontsize=13, fontweight="bold")

        gs  = mpl.gridspec.GridSpec(2, 3, figure=fig,
                                    hspace=0.45, wspace=0.35)

        def ax(*args):
            a = fig.add_subplot(*args)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        def line_plot(axis, y, title, color, ylabel=""):
            axis.plot(wl, y, color=color, lw=1.5)
            axis.set_title(title, color="#1f2328", fontsize=9)
            xlabel = ("Wavelength (nm)" if self.wavelengths is not None
                      else "Band index")
            axis.set_xlabel(xlabel, color="#57606a", fontsize=8)
            if ylabel:
                axis.set_ylabel(ylabel, color="#57606a", fontsize=8)

        line_plot(ax(gs[0, 0]), rec.band_means,
                  "Band Means", "#58a6ff", "Mean reflectance")
        line_plot(ax(gs[0, 1]), rec.band_stds,
                  "Band Std Dev", "#3fb950", "Std dev")
        line_plot(ax(gs[0, 2]), rec.band_snr,
                  "Band SNR", "#e3b341", "SNR")
        line_plot(ax(gs[1, 0]), rec.band_noise_mad,
                  "Band Noise (MAD)", "#f78166", "Noise (MAD)")
        line_plot(ax(gs[1, 1]), rec.band_saturation_frac,
                  "Band Saturation Fraction", "#d2a8ff", "Fraction")
        line_plot(ax(gs[1, 2]),
                  np.abs(np.diff(rec.band_means, prepend=rec.band_means[0])),
                  "Spectral Gradient |Δmean|", "#79c0ff", "|Δ|")

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — spectral diversity heatmap
    # ─────────────────────────────────────────────────────────────────────

    def plot_spectral_diversity(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Pairwise cosine-similarity heatmap between all cube mean spectra."""
        self._check_loaded()
        plt = _plt()
        valid = [r for r in self._records if not r.is_corrupt
                 and r.band_means is not None]
        dom_b = Counter(r.bands for r in valid).most_common(1)[0][0]
        matching = [r for r in valid if r.bands == dom_b][:50]

        if len(matching) < 2:
            self._log("Need ≥ 2 cubes with matching band counts.")
            return

        spectra = np.stack([r.band_means for r in matching])
        norms   = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-9
        sim     = (spectra / norms) @ (spectra / norms).T

        fig, axis = plt.subplots(figsize=figsize, facecolor="white")
        axis.set_facecolor("#f6f8fa")
        for sp in axis.spines.values():
            sp.set_edgecolor("#d0d7de")
        axis.tick_params(colors="#57606a", labelsize=7)

        im = axis.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=axis, label="Cosine similarity"
                     ).ax.tick_params(labelsize=7)

        tick_labels = [(r.label or Path(r.path).stem)[:16] for r in matching]
        if len(tick_labels) <= 30:
            axis.set_xticks(range(len(tick_labels)))
            axis.set_xticklabels(tick_labels, rotation=45,
                                  ha="right", fontsize=6, color="#57606a")
            axis.set_yticks(range(len(tick_labels)))
            axis.set_yticklabels(tick_labels, fontsize=6, color="#57606a")

        axis.set_title(
            f"Spectral Diversity — cosine similarity ({len(matching)} cubes)",
            color="#1f2328", fontsize=11)
        fig.suptitle("HyperspectralEDA — Spectral Diversity",
                     color="#1f2328", fontsize=13, fontweight="bold")
        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # HTML Report
    # ─────────────────────────────────────────────────────────────────────

    def report(self, output_path: str = "viseda_hyperspectral_report.html") -> str:
        """Generate a self-contained HTML report."""
        self._check_loaded()
        s = self.summary()
        _generate_html_report(s, output_path)
        self._log(f"Report saved → {output_path}")
        return output_path

    # ─────────────────────────────────────────────────────────────────────
    # Per-cube analysis
    # ─────────────────────────────────────────────────────────────────────

    def _analyse_file(self, path: Path, label_from_parent: bool) -> CubeRecord:
        rec = CubeRecord()
        rec.path     = str(path)
        rec.file_ext = path.suffix.lower()
        rec.file_size_kb = path.stat().st_size / 1024 if path.exists() else None
        rec.label    = (path.parent.name if label_from_parent
                        else self._label_map.get(str(path.resolve())))
        try:
            cube = self._read_cube(path)
        except Exception as e:
            self._log(f"  ✗ {path.name}: {e}")
            rec.is_corrupt = True
            return rec
        self._fill_stats(rec, cube)
        return rec

    def _fill_stats(self, rec: CubeRecord, cube: np.ndarray) -> None:
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        cube = cube.astype(np.float32)

        H, W, B = cube.shape
        rec.height = H; rec.width = W; rec.bands = B
        rec.dtype  = str(cube.dtype)

        # ── global scalars ────────────────────────────────────────────
        rec.global_mean    = float(cube.mean())
        rec.global_std     = float(cube.std())
        rec.global_min     = float(cube.min())
        rec.global_max     = float(cube.max())
        rec.dynamic_range  = float(cube.max() - cube.min())

        # ── per-band statistics ───────────────────────────────────────
        band_means = cube.mean(axis=(0, 1))
        band_stds  = cube.std(axis=(0, 1))
        rec.band_means = band_means
        rec.band_stds  = band_stds
        rec.band_mins  = cube.min(axis=(0, 1))
        rec.band_maxs  = cube.max(axis=(0, 1))

        eps = 1e-9
        rec.band_snr         = band_means / (band_stds + eps)
        rec.snr_mean         = float(rec.band_snr.mean())
        rec.band_noise_mad   = np.median(
            np.abs(cube - np.median(cube, axis=(0,1), keepdims=True)),
            axis=(0, 1))
        data_max = float(cube.max())
        rec.band_saturation_frac = (cube == data_max).mean(axis=(0, 1))

        # ── spectral quality ─────────────────────────────────────────
        if B > 1:
            rec.spectral_smoothness = float(
                np.abs(np.diff(band_means)).mean()
            )
        else:
            rec.spectral_smoothness = 0.0
        glob_std = float(band_stds.std()) + eps
        rec.n_dropout_bands = int(
            (band_stds < self.dropout_threshold * glob_std).sum())

        # inter-band correlation (mean off-diagonal) on pixel sample
        if B > 1:
            flat = cube.reshape(-1, B)
            if len(flat) > 5000:
                idx  = np.random.default_rng(0).choice(len(flat), 5000, replace=False)
                flat = flat[idx]
            corr = np.corrcoef(flat.T)
            mask = ~np.eye(B, dtype=bool)
            rec.inter_band_corr = float(corr[mask].mean())
        else:
            rec.inter_band_corr = 0.0

        # ── spatial maps ─────────────────────────────────────────────
        rec.spatial_mean_map = cube.mean(axis=2)
        rec.spatial_std_map  = cube.std(axis=2)

        # ── spectral indices ─────────────────────────────────────────
        for name in ("ndvi", "ndwi", "evi", "savi"):
            try:
                idx_map = self._compute_index(cube, name)
                setattr(rec, f"{name}_mean", float(idx_map.mean()))
                setattr(rec, f"{name}_std",  float(idx_map.std()))
            except Exception:
                pass

        # ── GLCM texture on first PC ──────────────────────────────────
        if self.compute_glcm and B > 1:
            self._fill_glcm(rec, cube)

        # ── PCA variance profile ─────────────────────────────────────
        if self.compute_pca and B > 1:
            self._fill_pca(rec, cube)

    def _fill_glcm(self, rec: CubeRecord, cube: np.ndarray) -> None:
        try:
            graycomatrix, graycoprops = _ski()
            from sklearn.decomposition import PCA as _PCA
            H, W, B = cube.shape
            flat = cube.reshape(-1, B)
            if len(flat) > 10_000:
                idx  = np.random.default_rng(0).choice(len(flat), 10_000, replace=False)
                flat_s = flat[idx]
            else:
                flat_s = flat
            pc1  = _PCA(n_components=1, svd_solver="randomized").fit_transform(
                flat_s)
            # use the full image projection
            pc1_full = _PCA(n_components=1, svd_solver="randomized"
                            ).fit(flat_s).transform(flat).reshape(H, W)
            small = __import__("cv2").resize(pc1_full, (128, 128))
            # normalise to 0-63
            mn, mx = small.min(), small.max()
            quant  = ((small - mn) / (mx - mn + 1e-9) * 63).astype(np.uint8)
            glcm   = graycomatrix(quant, distances=[1],
                                  angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                  levels=64, symmetric=True, normed=True)
            rec.glcm_contrast      = float(graycoprops(glcm, "contrast").mean())
            rec.glcm_dissimilarity = float(graycoprops(glcm, "dissimilarity").mean())
            rec.glcm_homogeneity   = float(graycoprops(glcm, "homogeneity").mean())
            rec.glcm_energy        = float(graycoprops(glcm, "energy").mean())
            rec.glcm_correlation   = float(graycoprops(glcm, "correlation").mean())
            rec.glcm_asm           = float(graycoprops(glcm, "ASM").mean())
        except Exception:
            pass

    def _fill_pca(self, rec: CubeRecord, cube: np.ndarray) -> None:
        try:
            from sklearn.decomposition import PCA
            H, W, B = cube.shape
            flat = cube.reshape(-1, B)
            n    = min(20, B, len(flat))
            pca  = PCA(n_components=n, svd_solver="randomized")
            pca.fit(flat if len(flat) <= 10_000 else
                    flat[np.random.default_rng(0).choice(len(flat), 10_000,
                                                          replace=False)])
            rec.pca_variance_ratio = pca.explained_variance_ratio_.copy()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # Spectral index computation
    # ─────────────────────────────────────────────────────────────────────

    def _band_for_wl(self, target_nm: float, B: int) -> int:
        """Return band index closest to *target_nm* wavelength."""
        if self.wavelengths is not None and len(self.wavelengths) >= B:
            return int(np.argmin(np.abs(self.wavelengths[:B] - target_nm)))
        # Fallback: assume VNIR 400-1000 nm linear mapping
        frac = (target_nm - 400) / 600
        return int(np.clip(frac * B, 0, B - 1))

    def _compute_index(self, cube: np.ndarray, name: str) -> np.ndarray:
        B = cube.shape[2]
        nir   = self._ndvi_nir   or self._band_for_wl(850, B)
        red   = self._ndvi_red   or self._band_for_wl(670, B)
        green = self._ndwi_green or self._band_for_wl(560, B)
        blue  =                     self._band_for_wl(490, B)
        nir   = min(nir, B-1); red = min(red, B-1)
        green = min(green, B-1); blue = min(blue, B-1)

        NIR = cube[:, :, nir].astype(np.float32)
        RED = cube[:, :, red].astype(np.float32)
        GRN = cube[:, :, green].astype(np.float32)
        BLU = cube[:, :, blue].astype(np.float32)

        eps = 1e-9
        if name == "ndvi":
            return (NIR - RED) / (NIR + RED + eps)
        if name == "ndwi":
            return (GRN - NIR) / (GRN + NIR + eps)
        if name == "evi":
            return 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLU + 1 + eps)
        if name == "savi":
            L = 0.5
            return (1 + L) * (NIR - RED) / (NIR + RED + L + eps)
        raise ValueError(f"Unknown index: {name}")

    # ─────────────────────────────────────────────────────────────────────
    # File reading
    # ─────────────────────────────────────────────────────────────────────

    def _read_cube(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(str(path)).astype(np.float32)
        if suffix == ".npz":
            data = np.load(str(path))
            key  = list(data.keys())[0]
            return data[key].astype(np.float32)
        if suffix == ".mat":
            import scipy.io
            mat  = scipy.io.loadmat(str(path))
            keys = [k for k in mat if not k.startswith("_")]
            if not keys:
                raise ValueError("No data arrays found in .mat file.")
            # prefer the key with the largest array
            key = max(keys, key=lambda k: np.prod(mat[k].shape)
                      if hasattr(mat[k], "shape") else 0)
            return np.array(mat[key]).astype(np.float32)
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
                    arr = src.read()        # (B, H, W)
                return arr.transpose(1, 2, 0).astype(np.float32)
            except ImportError:
                raise ImportError("pip install rasterio")
        raise ValueError(f"Unsupported format: {suffix}")

    def _load_cube_array(self, rec: CubeRecord) -> np.ndarray:
        """Load the raw cube array for a record (re-reads from disk)."""
        if rec.path.startswith("<array"):
            raise ValueError(
                "Cannot reload in-memory arrays. "
                "Store arrays externally and use load() with file paths.")
        return self._read_cube(Path(rec.path))

    # ─────────────────────────────────────────────────────────────────────
    # Plot helpers — single cube
    # ─────────────────────────────────────────────────────────────────────

    def _plot_cube_info(self, ax, rec, cube):
        ax.axis("off")
        H, W, B = cube.shape
        wl_range = (f"{self.wavelengths[0]:.0f}–{self.wavelengths[B-1]:.0f} nm"
                    if self.wavelengths is not None else "unknown")
        lines = [
            f"Shape:          {H} × {W} × {B} bands",
            f"Wavelengths:    {wl_range}",
            f"File size:      {rec.file_size_kb:.1f} KB"
                             if rec.file_size_kb else "File size: N/A",
            f"Global mean:    {rec.global_mean:.4f}",
            f"Global std:     {rec.global_std:.4f}",
            f"Dynamic range:  {rec.dynamic_range:.4f}",
            f"Mean SNR:       {rec.snr_mean:.2f}",
            f"Dropout bands:  {rec.n_dropout_bands}",
            f"Spectral smooth:{rec.spectral_smoothness:.5f}",
            f"Inter-band corr:{rec.inter_band_corr:.4f}",
            f"Label:          {rec.label or 'N/A'}",
        ]
        ax.text(0.04, 0.97, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, color="#1f2328", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#eaeef2", edgecolor="#d0d7de"))
        ax.set_title("Cube Overview", color="#1f2328", fontsize=11)

    def _plot_false_colour(self, ax, cube, rgb_bands, title="False Colour"):
        B = cube.shape[2]
        b = [min(x, B-1) for x in rgb_bands]
        fc = cube[:, :, b].astype(np.float32)
        fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-9)
        ax.imshow(np.clip(fc, 0, 1))
        ax.set_title(title, color="#1f2328", fontsize=9)
        ax.axis("off")

    def _plot_spatial_mean(self, ax, rec):
        if rec.spatial_mean_map is None:
            ax.set_title("Spatial Mean", color="#1f2328", fontsize=9)
            return
        im = ax.imshow(rec.spatial_mean_map, cmap="gray")
        _plt().colorbar(im, ax=ax, fraction=0.046, pad=0.04
                        ).ax.tick_params(labelsize=6)
        ax.set_title("Spatial Mean Map (avg across bands)",
                     color="#1f2328", fontsize=9)
        ax.axis("off")

    def _plot_mean_spectrum_single(self, ax, rec, wl):
        if rec.band_means is None:
            return
        B = len(rec.band_means)
        x = wl[:B]
        ax.plot(x, rec.band_means, color="#58a6ff", lw=1.8, label="Mean")
        ax.fill_between(x,
                        rec.band_means - rec.band_stds,
                        rec.band_means + rec.band_stds,
                        alpha=0.2, color="#58a6ff", label="±1σ")
        ax.set_title("Mean Spectral Signature ± 1σ",
                     color="#1f2328", fontsize=9)
        xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
        ax.set_xlabel(xlabel, color="#57606a", fontsize=8)
        ax.set_ylabel("Reflectance", color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    def _plot_band_snr(self, ax, rec, wl):
        if rec.band_snr is None:
            return
        B = len(rec.band_snr)
        ax.plot(wl[:B], rec.band_snr, color="#e3b341", lw=1.5)
        ax.axhline(rec.snr_mean, color="#f78166", lw=1,
                   linestyle="--", alpha=0.8,
                   label=f"Mean SNR = {rec.snr_mean:.1f}")
        ax.set_title("SNR per Band", color="#1f2328", fontsize=9)
        ax.set_xlabel("Band / Wavelength", color="#57606a", fontsize=8)
        ax.set_ylabel("SNR", color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    def _plot_band_stats_lines(self, ax, rec, wl):
        if rec.band_means is None:
            return
        B = len(rec.band_means)
        x = wl[:B]
        ax.plot(x, rec.band_means, color="#58a6ff", lw=1.5, label="Mean")
        ax.plot(x, rec.band_maxs,  color="#f78166", lw=1,
                alpha=0.7, label="Max")
        ax.plot(x, rec.band_mins,  color="#3fb950", lw=1,
                alpha=0.7, label="Min")
        ax.fill_between(x, rec.band_mins, rec.band_maxs,
                        alpha=0.07, color="#58a6ff")
        ax.set_title("Per-band Min / Mean / Max",
                     color="#1f2328", fontsize=9)
        xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band"
        ax.set_xlabel(xlabel, color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    def _plot_index_map_single(self, ax, cube, name):
        try:
            idx_map = self._compute_index(cube, name)
            cmap = {"ndvi": "RdYlGn", "ndwi": "RdBu",
                    "evi":  "YlGn",   "savi": "Greens"}.get(name, "viridis")
            im = ax.imshow(idx_map, cmap=cmap, vmin=-1, vmax=1)
            _plt().colorbar(im, ax=ax, fraction=0.046, pad=0.04
                            ).ax.tick_params(labelsize=6)
            ax.set_title(
                f"{name.upper()} (μ={idx_map.mean():.3f})",
                color="#1f2328", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"{name.upper()}\nN/A\n{e}",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(name.upper(), color="#1f2328", fontsize=9)
        ax.axis("off")

    def _plot_pca_variance_single(self, ax, rec):
        if rec.pca_variance_ratio is None:
            ax.text(0.5, 0.5, "PCA not computed",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes)
            ax.set_title("PCA Variance", color="#1f2328", fontsize=9)
            return
        v = rec.pca_variance_ratio
        cv = np.cumsum(v) * 100
        x  = np.arange(1, len(v)+1)
        ax.bar(x, v*100, color="#d2a8ff", alpha=0.8)
        ax.plot(x, cv, color="#1f2328", lw=1.5, marker=".", markersize=4)
        ax.axhline(95, color="#f78166", lw=1, linestyle="--",
                   alpha=0.7, label="95%")
        ax.set_title("PCA Variance Explained",
                     color="#1f2328", fontsize=9)
        ax.set_xlabel("Component", color="#57606a", fontsize=8)
        ax.set_ylabel("% variance", color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    def _plot_glcm_radar(self, ax, records):
        feats  = ["glcm_contrast","glcm_homogeneity",
                  "glcm_energy","glcm_correlation","glcm_asm"]
        labels = ["Contrast","Homogeneity","Energy","Correlation","ASM"]
        vals   = []
        for f in feats:
            a = [getattr(r, f) for r in records if getattr(r, f) is not None]
            vals.append(float(np.mean(a)) if a else 0.0)

        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()

        if all(v == 0 for v in vals):
            ax2 = fig.add_axes(pos)
            ax2.set_facecolor("#f6f8fa")
            ax2.text(0.5, 0.5, "GLCM not computed\n(pip install scikit-image)",
                     ha="center", va="center", color="#57606a",
                     transform=ax2.transAxes, fontsize=9)
            ax2.set_title("Texture (GLCM)", color="#1f2328", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
            return

        ax_p = fig.add_axes(pos, projection="polar")
        ax_p.set_facecolor("#f6f8fa")
        mx = max(vals) or 1.0
        nv = [v/mx for v in vals] + [vals[0]/mx]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        ax_p.set_theta_offset(np.pi/2)
        ax_p.set_theta_direction(-1)
        ax_p.plot(angles, nv, color="#58a6ff", lw=2)
        ax_p.fill(angles, nv, color="#58a6ff", alpha=0.25)
        ax_p.set_xticks(angles[:-1])
        ax_p.set_xticklabels(labels, color="#1f2328", fontsize=7)
        ax_p.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_p.set_yticklabels(["25%","50%","75%","100%"],
                              color="#57606a", fontsize=6)
        ax_p.spines["polar"].set_edgecolor("#d0d7de")
        ax_p.tick_params(colors="#57606a")
        ax_p.set_title("Texture Features (GLCM, normalised)",
                       color="#1f2328", fontsize=9, pad=12)

    def _plot_spectral_quality_bars(self, ax, records):
        metrics = {
            "Mean SNR":    [r.snr_mean            for r in records
                            if r.snr_mean          is not None],
            "Dropout\nbands": [r.n_dropout_bands  for r in records
                               if r.n_dropout_bands is not None],
            "Smooth-\nness": [r.spectral_smoothness for r in records
                              if r.spectral_smoothness is not None],
            "Inter-band\ncorr": [r.inter_band_corr for r in records
                                 if r.inter_band_corr is not None],
        }
        colors = ["#e3b341","#f78166","#3fb950","#58a6ff"]
        labels, means, stds = [], [], []
        for (lbl, vals), color in zip(metrics.items(), colors):
            if vals:
                labels.append(lbl)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))

        if not labels:
            ax.set_title("Spectral Quality", color="#1f2328", fontsize=9)
            return
        y = np.arange(len(labels))
        ax.barh(y, means, xerr=stds, color=colors[:len(labels)],
                alpha=0.8, edgecolor="none",
                error_kw=dict(ecolor="#57606a", elinewidth=1))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Spectral Quality Metrics",
                     color="#1f2328", fontsize=9)

    # ─────────────────────────────────────────────────────────────────────
    # Plot helpers — dataset level
    # ─────────────────────────────────────────────────────────────────────

    def _plot_dataset_info(self, ax, valid, s):
        ax.axis("off")
        inv = s["inventory"]
        sq  = s["spectral_quality"]
        lines = [
            f"Total cubes:    {inv['total_cubes']:,}",
            f"Valid:          {inv['valid_cubes']:,}",
            f"Corrupt:        {inv['corrupt_cubes']:,}",
            f"Unique labels:  "
            f"{len(inv['label_distribution']) if inv['label_distribution'] else 'N/A'}",
            f"Band counts:    {inv['band_distribution']}",
            f"Formats:        {inv['format_distribution']}",
            f"Mean SNR:       {sq['snr_mean'].get('mean','N/A')}",
            f"Median H × W:   "
            f"{s['spatial']['height'].get('median','?'):.0f} × "
            f"{s['spatial']['width'].get('median','?'):.0f}",
        ]
        ax.text(0.04, 0.97, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color="#1f2328", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#eaeef2", edgecolor="#d0d7de"))
        ax.set_title("Dataset Overview", color="#1f2328", fontsize=11)

    def _plot_label_dist(self, ax, valid):
        labels = [r.label for r in valid if r.label]
        if not labels:
            ax.text(0.5, 0.5, "No labels provided\n(use label_from_parent=True)",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("Label Distribution", color="#1f2328", fontsize=11)
            return
        cnt = Counter(labels).most_common(25)
        names, counts = zip(*cnt)
        y = np.arange(len(names))
        ax.barh(y, counts, color="#58a6ff", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Label Distribution", color="#1f2328", fontsize=11)
        ax.set_xlabel("Count", color="#57606a", fontsize=8)
        if len(counts) > 1:
            ratio = max(counts) / min(counts)
            ax.text(0.97, 0.02, f"imbalance: {ratio:.1f}×",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#e3b341")

    def _plot_hist(self, ax, data, title, color, bins=20):
        data = [d for d in data if d is not None and np.isfinite(d)]
        if not data:
            ax.set_title(title, color="#1f2328", fontsize=9); return
        ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="none")
        mu = float(np.mean(data))
        ax.axvline(mu, color="#1f2328", lw=1.2, linestyle="--",
                   alpha=0.7, label=f"μ={mu:.3f}")
        ax.set_title(title, color="#1f2328", fontsize=9)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    def _plot_cross_spectrum(self, ax, s):
        ss = s["spectral_stats"]
        mean_spec = ss.get("cross_cube_mean_spectrum")
        std_spec  = ss.get("cross_cube_std_spectrum")
        n         = ss.get("n_matching_cubes", 0)
        dom_b     = ss.get("dominant_band_count", 0)

        if not mean_spec:
            ax.text(0.5, 0.5, "No matching-band cubes",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes)
            ax.set_title("Cross-Cube Mean Spectrum", color="#1f2328", fontsize=9)
            return

        mean_spec = np.array(mean_spec)
        std_spec  = np.array(std_spec) if std_spec else np.zeros_like(mean_spec)
        wl = (self.wavelengths[:dom_b]
              if self.wavelengths is not None else np.arange(dom_b))

        ax.plot(wl, mean_spec, color="#58a6ff", lw=1.8, label="Dataset mean")
        ax.fill_between(wl, mean_spec-std_spec, mean_spec+std_spec,
                        alpha=0.2, color="#58a6ff", label="±1σ across cubes")
        ax.set_title(f"Cross-Cube Mean Spectrum (n={n} cubes, {dom_b} bands)",
                     color="#1f2328", fontsize=9)
        xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
        ax.set_xlabel(xlabel, color="#57606a", fontsize=8)
        ax.set_ylabel("Mean Reflectance", color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    def _plot_band_count_dist(self, ax, valid):
        cnt = Counter(r.bands for r in valid)
        names, counts = zip(*sorted(cnt.items()))
        ax.bar([str(n) for n in names], counts,
               color="#d2a8ff", alpha=0.85, edgecolor="none")
        ax.set_title("Band Count Distribution", color="#1f2328", fontsize=9)
        ax.set_xlabel("Bands", color="#57606a", fontsize=8)
        ax.set_ylabel("Cubes", color="#57606a", fontsize=8)

    def _plot_spectral_diversity(self, ax, valid):
        dom_b = Counter(r.bands for r in valid).most_common(1)[0][0]
        m = [r for r in valid if r.bands == dom_b
             and r.band_means is not None][:50]

        if len(m) < 2:
            ax.text(0.5, 0.5, "Need ≥2 cubes\nwith same bands",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("Spectral Diversity", color="#1f2328", fontsize=9)
            return

        spectra = np.stack([r.band_means for r in m])
        norms   = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-9
        sim     = (spectra / norms) @ (spectra / norms).T

        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        _plt().colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        label="Cosine sim").ax.tick_params(labelsize=6)
        tick_labels = [(r.label or Path(r.path).stem)[:12] for r in m]
        if len(tick_labels) <= 20:
            ax.set_xticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels, rotation=45, ha="right",
                               fontsize=6, color="#57606a")
            ax.set_yticks(range(len(tick_labels)))
            ax.set_yticklabels(tick_labels, fontsize=6, color="#57606a")
        ax.set_title(f"Spectral Diversity ({len(m)} cubes, {dom_b} bands)",
                     color="#1f2328", fontsize=9)

    def _plot_pca_variance_dataset(self, ax, valid):
        pca_vars = [r.pca_variance_ratio for r in valid
                    if r.pca_variance_ratio is not None]
        if not pca_vars:
            ax.text(0.5, 0.5, "PCA not computed",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes)
            ax.set_title("PCA (dataset mean)", color="#1f2328", fontsize=9)
            return
        min_len = min(len(v) for v in pca_vars)
        stack   = np.stack([v[:min_len] for v in pca_vars])
        mean_v  = stack.mean(axis=0)
        std_v   = stack.std(axis=0)
        cumvar  = np.cumsum(mean_v) * 100
        x = np.arange(1, min_len+1)
        ax.bar(x, mean_v*100, yerr=std_v*100, color="#d2a8ff",
               alpha=0.8, error_kw=dict(ecolor="#57606a", elinewidth=0.8))
        ax.plot(x, cumvar, color="#1f2328", lw=1.5,
                marker=".", markersize=3)
        ax.axhline(95, color="#f78166", lw=1, linestyle="--",
                   alpha=0.7, label="95%")
        ax.set_title("PCA Variance (dataset mean ± std)",
                     color="#1f2328", fontsize=9)
        ax.set_xlabel("Component", color="#57606a", fontsize=8)
        ax.set_ylabel("% variance", color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="white", edgecolor="#d0d7de")

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_paths(self, source, recursive) -> List[Path]:
        if isinstance(source, (list, tuple)):
            return [Path(p) for p in source]
        source = Path(source)
        if source.is_file():
            return [source]
        pattern = "**/*" if recursive else "*"
        return sorted(
            p for p in source.glob(pattern)
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[viseda] {msg}")

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() or .load_arrays() first.")

    @staticmethod
    def _finalise(fig, save_path, dpi):
        plt = _plt()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        else:
            plt.show()
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _stat_dict(arr) -> Dict[str, float]:
    arr = np.asarray([x for x in np.asarray(arr).ravel()
                      if x is not None and np.isfinite(x)])
    if len(arr) == 0:
        return {}
    return {
        "min":    round(float(np.min(arr)),    4),
        "max":    round(float(np.max(arr)),    4),
        "mean":   round(float(np.mean(arr)),   4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(np.std(arr)),    4),
        "p25":    round(float(np.percentile(arr, 25)), 4),
        "p75":    round(float(np.percentile(arr, 75)), 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═════════════════════════════════════════════════════════════════════════════

def _generate_html_report(summary: Dict[str, Any], output_path: str) -> None:
    inv = summary.get("inventory", {})
    sp  = summary.get("spatial",   {})
    ss  = summary.get("spectral_stats", {})
    sq  = summary.get("spectral_quality", {})
    si  = summary.get("spectral_indices", {})
    tx  = summary.get("texture",   {})
    pc  = summary.get("pca",       {})
    lb  = summary.get("labels",    {})

    def card(title, stats):
        if not stats:
            return ""
        rows = "".join(
            f'<div class="stat"><span>{k}</span>'
            f'<span class="val">{_fmt(v)}</span></div>'
            for k, v in stats.items() if not isinstance(v, (list, dict))
        )
        return f'<div class="card"><h3>{title}</h3>{rows}</div>'

    def badge(text, cls="blue"):
        return f'<span class="badge badge-{cls}">{text}</span>'

    def bar_chart(title, dist, span=1):
        if not dist:
            return ""
        mx = max(dist.values()) or 1
        rows = ""
        for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1])[:25]:
            pct = cnt / mx * 100
            rows += (f'<div class="bar-row">'
                     f'<span class="bar-label">{lbl}</span>'
                     f'<div class="bar">'
                     f'<div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
                     f'<span class="bar-count">{cnt:,}</span></div>')
        sp_style = f'grid-column:span {span};' if span > 1 else ''
        return (f'<div class="card" style="{sp_style}">'
                f'<h3>{title}</h3><div class="bar-wrap">{rows}</div></div>')

    badges_html = (
        badge(f"{inv.get('total_cubes',0):,} cubes") +
        badge(f"{inv.get('valid_cubes',0):,} valid", "green") +
        (badge(f"{inv.get('corrupt_cubes',0):,} corrupt", "red")
         if inv.get("corrupt_cubes") else "")
    )

    counts_card = card("Counts", {
        "Total cubes":   inv.get("total_cubes"),
        "Valid cubes":   inv.get("valid_cubes"),
        "Corrupt cubes": inv.get("corrupt_cubes"),
    })
    imbalance_card = card("Class Imbalance", {
        "Imbalance ratio": lb.get("class_imbalance_ratio"),
    })

    index_html = ""
    for name, data in si.items():
        index_html += card(f"{name.upper()} (per-cube mean)",
                           data.get("per_cube_mean", {}))

    texture_html = "".join(
        card(k.replace("_"," ").title(), v) for k, v in tx.items())

    pca_html = card("PCA Summary", {
        "Components for 95% variance": pc.get("n_components_95pct_mean"),
    })

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VisEDA — Hyperspectral Report</title>
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
       padding:.18rem 0;border-bottom:1px solid var(--border)}}
.stat:last-child{{border-bottom:none}}
.val{{color:var(--accent);font-variant-numeric:tabular-nums}}
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
<h1>🌈 VisEDA — Hyperspectral EDA Report</h1>
<p class="sub">Generated by <strong>VisEDA</strong></p>
<p style="margin-bottom:1rem">{badges_html}</p>

<h2>📦 Inventory</h2>
<div class="grid">
  {counts_card}
  {bar_chart("Label Distribution", inv.get("label_distribution") or {}, span=2)}
  {bar_chart("Band Count Distribution", {str(k): v for k, v in inv.get("band_distribution", {}).items()})}
  {bar_chart("Format Distribution", inv.get("format_distribution", {}))}
</div>

<h2>📐 Spatial</h2>
<div class="grid">
  {card("Height (px)",    sp.get("height",  {}))}
  {card("Width (px)",     sp.get("width",   {}))}
  {card("Bands",          sp.get("bands",   {}))}
  {card("File Size (KB)", sp.get("file_size_kb", {}))}
</div>

<h2>〰️ Spectral Statistics</h2>
<div class="grid">
  {card("Global Mean (per cube)",    ss.get("global_mean",   {}))}
  {card("Global Std (per cube)",     ss.get("global_std",    {}))}
  {card("Dynamic Range (per cube)",  ss.get("dynamic_range", {}))}
</div>

<h2>🔬 Spectral Quality</h2>
<div class="grid">
  {card("SNR (per cube)",             sq.get("snr_mean",            {}))}
  {card("Spectral Smoothness",        sq.get("spectral_smoothness", {}))}
  {card("Inter-band Correlation",     sq.get("inter_band_corr",     {}))}
  {card("Dropout Bands (per cube)",   sq.get("n_dropout_bands",     {}))}
</div>

<h2>🌿 Spectral Indices</h2>
<div class="grid">{index_html}</div>

<h2>🧱 Texture (GLCM on PC1)</h2>
<div class="grid">{texture_html}</div>

<h2>📊 PCA</h2>
<div class="grid">{pca_html}</div>

<h2>🏷️ Labels</h2>
<div class="grid">
  {bar_chart("Label Distribution", lb.get("label_distribution") or {}, span=2)}
  {imbalance_card}
</div>

<footer>Generated by VisEDA — Visual Exploratory Data Analysis</footer>
</body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")


def _fmt(v) -> str:
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:,.4f}"
    if isinstance(v, int):   return f"{v:,}"
    return str(v)