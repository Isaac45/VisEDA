"""
viseda.image.eda
----------------
Comprehensive EDA for image datasets.

Analyses
~~~~~~~~
INVENTORY
  - Total / valid / corrupt count
  - File format & extension distribution
  - File size distribution (KB)
  - Colour-mode distribution (RGB / grayscale / RGBA)
  - Bit-depth distribution

SPATIAL
  - Height, width, aspect-ratio distributions
  - Resolution (megapixels) distribution
  - Portrait / landscape / square breakdown
  - Spatial resolution consistency check

PIXEL STATISTICS  (per image, then aggregated across dataset)
  - Mean, std, min, max per channel
  - Per-channel histograms aggregated across the dataset
  - Global dataset-level pixel mean and std ("dataset statistics" for
    normalisation — the same numbers used in torchvision transforms)

COLOUR ANALYSIS
  - RGB, HSV and Lab colour space distributions
  - Dominant colour palette extraction (K-Means across sample)
  - Colour temperature estimate (warm / neutral / cool)
  - Colour cast detection (channel imbalance)
  - Greyscale-like detection (low colour saturation)

QUALITY METRICS  (per image, then aggregated)
  - Brightness (mean luminance)
  - Contrast (std of luminance)
  - Sharpness (Laplacian variance)
  - Noise estimate (high-freq energy via Laplacian on smooth image)
  - Exposure (over- / under-exposed pixel fraction)
  - Blurriness flag (sharpness below threshold)
  - JPEG compression artefact score (blockiness)

TEXTURE & FREQUENCY
  - GLCM texture features: contrast, dissimilarity, homogeneity,
    energy, correlation, ASM (per image → aggregated)
  - FFT frequency energy distribution (low / mid / high freq ratio)

DUPLICATE DETECTION
  - Perceptual hash (pHash, 64-bit DCT)
  - Average hash (aHash)
  - Near-duplicate grouping (Hamming distance ≤ threshold)
  - Exact-duplicate detection (MD5)

DATASET-LEVEL
  - Class / label distribution (from folder names or dict)
  - Per-class pixel statistics
  - Class imbalance ratio
  - Outlier image detection (images far from dataset mean embedding)

VISUALISATIONS
  - Full EDA dashboard (light theme, 6×4 grid)
  - Sample image grid
  - Per-class sample grid
  - Channel correlation matrix
  - Pixel-value heatmap (average image across dataset)
  - UMAP / t-SNE embedding of image features (optional)
  - Duplicate group viewer
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ── lazy heavy imports ────────────────────────────────────────────────────
def _cv2():
    import cv2; return cv2

def _plt():
    import matplotlib.pyplot as plt; return plt

def _mpl():
    import matplotlib as mpl; return mpl

def _ski_feature():
    from skimage.feature import graycomatrix, graycoprops
    return graycomatrix, graycoprops


# ═════════════════════════════════════════════════════════════════════════════
# Per-image record
# ═════════════════════════════════════════════════════════════════════════════

class ImageRecord:
    """All per-image statistics stored in one lightweight object."""

    __slots__ = (
        # identity
        "path", "label", "file_ext", "file_size_kb",
        # spatial
        "height", "width", "channels", "dtype",
        "aspect_ratio", "megapixels",
        # pixel stats
        "mean_rgb", "std_rgb", "min_rgb", "max_rgb",
        # quality
        "brightness", "contrast", "sharpness",
        "noise_estimate",
        "overexposed_frac", "underexposed_frac",
        "is_blurry",
        "compression_score",
        # colour
        "mean_hsv", "mean_lab",
        "saturation_mean",
        "color_temp",          # "warm" | "neutral" | "cool"
        "is_grayscale_like",
        # texture
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "glcm_asm",
        # frequency
        "freq_low", "freq_mid", "freq_high",
        # entropy
        "entropy_val",
        # hashes
        "phash", "ahash", "md5",
        # status
        "is_corrupt",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.is_corrupt = False
        self.is_blurry = False
        self.is_grayscale_like = False


# ═════════════════════════════════════════════════════════════════════════════
# Main class
# ═════════════════════════════════════════════════════════════════════════════

class ImageEDA:
    """
    Comprehensive exploratory data analysis for image datasets.

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    max_images : int | None
        Analyse at most *max_images* (useful for quick previews).
    n_colors : int
        Dominant-colour clusters to extract (default 8).
    phash_threshold : int
        Hamming distance threshold for near-duplicate detection (default 10).
    blur_threshold : float
        Laplacian variance below this flags an image as blurry (default 50).
    compute_glcm : bool
        Compute GLCM texture features — requires scikit-image (default True).
    compute_freq : bool
        Compute FFT frequency band energies (default True).

    Examples
    --------
    >>> from viseda import ImageEDA
    >>> eda = ImageEDA()
    >>> eda.load("path/to/dataset/", label_from_parent=True)
    >>> print(eda.summary())
    >>> eda.plot()
    >>> eda.plot_samples()
    >>> eda.plot_average_image()
    >>> eda.report("report.html")
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp",
                      ".tif", ".tiff", ".webp", ".gif"}

    def __init__(
        self,
        verbose: bool = True,
        max_images: Optional[int] = None,
        n_colors: int = 8,
        phash_threshold: int = 10,
        blur_threshold: float = 50.0,
        compute_glcm: bool = True,
        compute_freq: bool = True,
    ):
        self.verbose = verbose
        self.max_images = max_images
        self.n_colors = n_colors
        self.phash_threshold = phash_threshold
        self.blur_threshold = blur_threshold
        self.compute_glcm = compute_glcm
        self.compute_freq = compute_freq

        self._records: List[ImageRecord] = []
        self._labels_map: Dict[str, str] = {}
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
    ) -> "ImageEDA":
        """
        Load images from a directory, file, or list of paths.

        Parameters
        ----------
        source
            Directory path, single image path, or list of paths.
        labels
            ``{path: label}`` mapping.
        label_from_parent
            Infer label from the parent folder name
            (e.g. ``dataset/cats/img.jpg`` → label ``"cats"``).
        recursive
            Recurse into sub-directories (default True).
        """
        paths = self._resolve_paths(source, recursive)
        if self.max_images:
            paths = paths[: self.max_images]

        if labels:
            self._labels_map = {str(Path(k).resolve()): v
                                for k, v in labels.items()}

        self._log(f"Found {len(paths)} images — computing statistics …")
        self._records = []

        for i, p in enumerate(paths):
            if self.verbose and i % max(1, len(paths) // 20) == 0:
                self._log(f"  [{i:>{len(str(len(paths)))}}/{len(paths)}] {p.name}")
            rec = self._analyse_single(p, label_from_parent)
            self._records.append(rec)

        self._loaded = True
        n_corrupt = sum(r.is_corrupt for r in self._records)
        self._log(
            f"Done. {len(self._records):,} images loaded "
            f"({n_corrupt} corrupt)."
        )
        return self

    def load_arrays(
        self,
        arrays: List[np.ndarray],
        labels: Optional[List[str]] = None,
    ) -> "ImageEDA":
        """Load images directly as NumPy arrays (HxWx3 uint8 or float)."""
        self._log(f"Loading {len(arrays)} arrays …")
        self._records = []
        for i, arr in enumerate(arrays):
            rec = ImageRecord()
            rec.path = f"<array_{i}>"
            rec.label = labels[i] if labels and i < len(labels) else None
            rec.file_ext = "array"
            try:
                self._fill_stats(rec, self._to_uint8_rgb(arr))
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

        Sections
        --------
        inventory, spatial, pixel_stats, quality, colour,
        texture, frequency, duplicates, dataset_stats, labels
        """
        self._check_loaded()
        valid   = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]

        if not valid:
            return {"error": "No valid images found."}

        def arr(attr):
            return np.array([getattr(r, attr) for r in valid
                             if getattr(r, attr) is not None])

        # ── inventory ───────────────────────────────────────────────
        ext_dist = dict(Counter(r.file_ext for r in valid))
        mode_dist = dict(Counter(
            {1: "grayscale", 3: "RGB", 4: "RGBA"}.get(r.channels, str(r.channels))
            for r in valid
        ))
        dtype_dist = dict(Counter(r.dtype for r in valid))

        # ── spatial ─────────────────────────────────────────────────
        orientations = Counter()
        for r in valid:
            if r.aspect_ratio > 1.05:   orientations["landscape"] += 1
            elif r.aspect_ratio < 0.95: orientations["portrait"]  += 1
            else:                        orientations["square"]    += 1

        # ── pixel / colour ───────────────────────────────────────────
        mean_rgb_matrix = np.array([r.mean_rgb for r in valid
                                    if r.mean_rgb is not None])
        dataset_mean = mean_rgb_matrix.mean(axis=0).tolist() \
            if len(mean_rgb_matrix) else []
        dataset_std_across = mean_rgb_matrix.std(axis=0).tolist() \
            if len(mean_rgb_matrix) else []

        std_rgb_matrix = np.array([r.std_rgb for r in valid
                                   if r.std_rgb is not None])
        dataset_pixel_std = std_rgb_matrix.mean(axis=0).tolist() \
            if len(std_rgb_matrix) else []

        color_temp_dist = dict(Counter(r.color_temp for r in valid
                                       if r.color_temp))
        grayscale_like_count = sum(r.is_grayscale_like for r in valid)

        # ── quality ──────────────────────────────────────────────────
        blurry_count = sum(r.is_blurry for r in valid)
        overexp  = arr("overexposed_frac")
        underexp = arr("underexposed_frac")

        # ── texture ──────────────────────────────────────────────────
        texture_summary = {}
        for feat in ("glcm_contrast", "glcm_dissimilarity",
                     "glcm_homogeneity", "glcm_energy",
                     "glcm_correlation", "glcm_asm"):
            a = arr(feat)
            if len(a):
                texture_summary[feat] = _stat_dict(a)

        # ── frequency ────────────────────────────────────────────────
        freq_summary = {}
        for band in ("freq_low", "freq_mid", "freq_high"):
            a = arr(band)
            if len(a):
                freq_summary[band] = _stat_dict(a)

        # ── duplicates ───────────────────────────────────────────────
        exact_dupe_groups   = self._find_exact_duplicates(valid)
        near_dupe_groups    = self._find_near_duplicates(valid)

        # ── label info ───────────────────────────────────────────────
        label_dist = None
        class_imbalance_ratio = None
        if any(r.label for r in valid):
            lc = Counter(r.label for r in valid)
            label_dist = dict(lc)
            mx = max(lc.values()); mn = min(lc.values())
            class_imbalance_ratio = round(mx / mn, 3) if mn else None

        result = {
            # ── inventory
            "inventory": {
                "total":        len(self._records),
                "valid":        len(valid),
                "corrupt":      len(corrupt),
                "corrupt_paths": [r.path for r in corrupt],
                "format_distribution": ext_dist,
                "colour_mode_distribution": mode_dist,
                "dtype_distribution": dtype_dist,
            },
            # ── spatial
            "spatial": {
                "height":      _stat_dict(arr("height")),
                "width":       _stat_dict(arr("width")),
                "aspect_ratio": _stat_dict(arr("aspect_ratio")),
                "megapixels":  _stat_dict(arr("megapixels")),
                "file_size_kb": _stat_dict(arr("file_size_kb")),
                "orientation_distribution": dict(orientations),
            },
            # ── pixel stats
            "pixel_stats": {
                "dataset_mean_rgb":         dataset_mean,
                "dataset_std_rgb_across_images": dataset_std_across,
                "dataset_pixel_std_rgb":    dataset_pixel_std,
                "per_image_mean_rgb":       _stat_dict(mean_rgb_matrix.mean(axis=1))
                                             if len(mean_rgb_matrix) else {},
            },
            # ── quality
            "quality": {
                "brightness":         _stat_dict(arr("brightness")),
                "contrast":           _stat_dict(arr("contrast")),
                "sharpness":          _stat_dict(arr("sharpness")),
                "noise_estimate":     _stat_dict(arr("noise_estimate")),
                "entropy":            _stat_dict(arr("entropy_val")),
                "compression_score":  _stat_dict(arr("compression_score")),
                "blurry_count":       blurry_count,
                "blurry_fraction":    round(blurry_count / len(valid), 4),
                "overexposed_frac":   _stat_dict(overexp),
                "underexposed_frac":  _stat_dict(underexp),
            },
            # ── colour
            "colour": {
                "saturation":           _stat_dict(arr("saturation_mean")),
                "colour_temp_distribution": color_temp_dist,
                "grayscale_like_count": grayscale_like_count,
                "grayscale_like_fraction": round(grayscale_like_count / len(valid), 4),
            },
            # ── texture
            "texture": texture_summary,
            # ── frequency
            "frequency": freq_summary,
            # ── duplicates
            "duplicates": {
                "exact_duplicate_groups":      exact_dupe_groups,
                "n_exact_duplicate_groups":    len(exact_dupe_groups),
                "near_duplicate_groups":       near_dupe_groups,
                "n_near_duplicate_groups":     len(near_dupe_groups),
            },
            # ── labels
            "labels": {
                "label_distribution":     label_dist,
                "class_imbalance_ratio":  class_imbalance_ratio,
            },
        }
        self._results["summary"] = result
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Dataset-level normalisation stats
    # ─────────────────────────────────────────────────────────────────────

    def normalization_stats(self) -> Dict[str, List[float]]:
        """
        Compute per-channel mean and std for use in dataset normalisation
        (e.g. ``torchvision.transforms.Normalize``).

        Returns
        -------
        dict with keys ``mean`` and ``std``, each a list of 3 floats
        in [0, 1] (channel order: R, G, B).

        Example
        -------
        >>> stats = eda.normalization_stats()
        >>> # use in torchvision:
        >>> transforms.Normalize(mean=stats["mean"], std=stats["std"])
        """
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt
                 and r.mean_rgb is not None]
        if not valid:
            raise RuntimeError("No valid images to compute statistics from.")

        means = np.array([r.mean_rgb for r in valid]) / 255.0  # (N, 3)
        stds  = np.array([r.std_rgb  for r in valid]) / 255.0  # (N, 3)

        return {
            "mean": means.mean(axis=0).tolist(),
            "std":  stds.mean(axis=0).tolist(),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — main dashboard
    # ─────────────────────────────────────────────────────────────────────

    def plot(
        self,
        figsize: Tuple[int, int] = (24, 28),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Render the full EDA dashboard (6-row × 4-col grid)."""
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        valid = [r for r in self._records if not r.is_corrupt]

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("VisEDA — Image Dataset Analysis",
                     fontsize=24, color="#1f2328", y=0.99, fontweight="bold")

        gs = mpl.gridspec.GridSpec(
            6, 4, figure=fig,
            hspace=0.55, wspace=0.35,
            left=0.06, right=0.97, top=0.97, bottom=0.02,
        )

        def ax(*args, **kw):
            a = fig.add_subplot(*args, **kw)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        # ── Row 0: inventory overview ─────────────────────────────────
        self._plot_info_card(ax(gs[0, :2]), valid)
        self._plot_label_dist(ax(gs[0, 2:]), valid)

        # ── Row 1: spatial distributions ─────────────────────────────
        self._plot_hist(ax(gs[1, 0]), [r.height for r in valid],
                        "Heights (px)", "#58a6ff")
        self._plot_hist(ax(gs[1, 1]), [r.width  for r in valid],
                        "Widths (px)",  "#3fb950")
        self._plot_hist(ax(gs[1, 2]), [r.aspect_ratio for r in valid],
                        "Aspect Ratios", "#d2a8ff")
        self._plot_hist(ax(gs[1, 3]), [r.megapixels for r in valid],
                        "Megapixels", "#ffa657")

        # ── Row 2: quality metrics ────────────────────────────────────
        self._plot_hist(ax(gs[2, 0]), [r.brightness for r in valid],
                        "Brightness", "#79c0ff")
        self._plot_hist(ax(gs[2, 1]), [r.contrast   for r in valid],
                        "Contrast (std)", "#56d364")
        self._plot_hist(ax(gs[2, 2]), [r.sharpness  for r in valid],
                        "Sharpness (Laplacian var)", "#e3b341",
                        log_x=True)
        self._plot_hist(ax(gs[2, 3]), [r.noise_estimate for r in valid],
                        "Noise Estimate", "#f78166")

        # ── Row 3: exposure & colour quality ─────────────────────────
        self._plot_hist(ax(gs[3, 0]),
                        [r.overexposed_frac * 100 for r in valid],
                        "Overexposed Pixels (%)", "#ff6b6b")
        self._plot_hist(ax(gs[3, 1]),
                        [r.underexposed_frac * 100 for r in valid],
                        "Underexposed Pixels (%)", "#a5d8ff")
        self._plot_hist(ax(gs[3, 2]), [r.saturation_mean for r in valid],
                        "Colour Saturation (HSV-S)", "#f9c74f")
        self._plot_hist(ax(gs[3, 3]), [r.entropy_val for r in valid],
                        "Pixel Entropy (bits)", "#90be6d")

        # ── Row 4: channel histograms + texture ──────────────────────
        self._plot_channel_histograms(ax(gs[4, :2]), valid)
        self._plot_texture_radar(ax(gs[4, 2:]), valid)

        # ── Row 5: frequency + scatter + colour temp ──────────────────
        self._plot_frequency_bands(ax(gs[5, 0]), valid)
        self._plot_brightness_sharpness_scatter(ax(gs[5, 1]), valid)
        self._plot_color_temp_pie(ax(gs[5, 2]), valid)
        self._plot_format_dist(ax(gs[5, 3]), valid)

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — colour dashboard
    # ─────────────────────────────────────────────────────────────────────

    def plot_colour(
        self,
        figsize: Tuple[int, int] = (22, 14),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Deep-dive colour analysis dashboard."""
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        valid = [r for r in self._records if not r.is_corrupt]

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("VisEDA — Colour Analysis",
                     fontsize=20, color="#1f2328", y=0.98, fontweight="bold")

        gs = mpl.gridspec.GridSpec(2, 4, figure=fig,
                                   hspace=0.45, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.93, bottom=0.06)

        def ax(*args):
            a = fig.add_subplot(*args)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        self._plot_channel_histograms(ax(gs[0, :2]), valid)
        self._plot_dominant_colours(ax(gs[0, 2:]), valid)
        self._plot_hsv_hue_wheel(ax(gs[1, 0]), valid)
        self._plot_saturation_value(ax(gs[1, 1]), valid)
        self._plot_lab_ab_scatter(ax(gs[1, 2]), valid)
        self._plot_color_temp_pie(ax(gs[1, 3]), valid)

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — quality dashboard
    # ─────────────────────────────────────────────────────────────────────

    def plot_quality(
        self,
        figsize: Tuple[int, int] = (22, 14),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Deep-dive quality metrics dashboard."""
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        valid = [r for r in self._records if not r.is_corrupt]

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("VisEDA — Quality Analysis",
                     fontsize=20, color="#1f2328", y=0.98, fontweight="bold")

        gs = mpl.gridspec.GridSpec(2, 4, figure=fig,
                                   hspace=0.45, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.93, bottom=0.06)

        def ax(*args):
            a = fig.add_subplot(*args)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        self._plot_hist(ax(gs[0, 0]), [r.sharpness for r in valid],
                        "Sharpness", "#e3b341", log_x=True)
        self._plot_hist(ax(gs[0, 1]), [r.noise_estimate for r in valid],
                        "Noise Estimate", "#f78166")
        self._plot_hist(ax(gs[0, 2]),
                        [r.compression_score for r in valid
                         if r.compression_score is not None],
                        "JPEG Blockiness Score", "#79c0ff")
        self._plot_blurry_breakdown(ax(gs[0, 3]), valid)
        self._plot_exposure_scatter(ax(gs[1, :2]), valid)
        self._plot_texture_radar(ax(gs[1, 2:]), valid)

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — texture dashboard
    # ─────────────────────────────────────────────────────────────────────

    def plot_texture(
        self,
        figsize: Tuple[int, int] = (22, 10),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """GLCM texture features and frequency analysis dashboard."""
        self._check_loaded()
        plt = _plt(); mpl = _mpl()
        valid = [r for r in self._records if not r.is_corrupt]

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("VisEDA — Texture & Frequency Analysis",
                     fontsize=20, color="#1f2328", y=0.98, fontweight="bold")

        gs = mpl.gridspec.GridSpec(1, 4, figure=fig,
                                   hspace=0.4, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.88, bottom=0.1)

        def ax(*args):
            a = fig.add_subplot(*args)
            a.set_facecolor("#f6f8fa")
            a.tick_params(colors="#57606a", labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor("#d0d7de")
            return a

        for i, (feat, color) in enumerate([
            ("glcm_contrast",     "#58a6ff"),
            ("glcm_homogeneity",  "#3fb950"),
            ("glcm_energy",       "#e3b341"),
            ("glcm_correlation",  "#d2a8ff"),
        ]):
            self._plot_hist(ax(gs[0, i]),
                            [getattr(r, feat) for r in valid
                             if getattr(r, feat) is not None],
                            feat.replace("_", " ").title(), color)

        self._finalise(fig, save_path, dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — sample grid
    # ─────────────────────────────────────────────────────────────────────

    def plot_samples(
        self,
        n: int = 25,
        cols: int = 5,
        label: Optional[str] = None,
        random_seed: int = 42,
        figsize: Optional[Tuple] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Display a grid of sample images.

        Parameters
        ----------
        label
            If supplied, only show images with this label.
        """
        self._check_loaded()
        plt = _plt(); cv2 = _cv2()

        pool = [r for r in self._records
                if not r.is_corrupt and not r.path.startswith("<array")]
        if label:
            pool = [r for r in pool if r.label == label]

        rng = np.random.default_rng(random_seed)
        sample = rng.choice(pool, size=min(n, len(pool)), replace=False)
        rows = int(np.ceil(len(sample) / cols))
        figsize = figsize or (cols * 3, rows * 3 + 0.5)

        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                 facecolor="white")
        axes = np.array(axes).flatten()

        for i, rec in enumerate(sample):
            img = cv2.imread(str(rec.path))
            if img is None:
                axes[i].axis("off"); continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = _resize_display(img, 224)
            axes[i].imshow(img)
            title = (rec.label or Path(rec.path).stem)[:20]
            axes[i].set_title(title, fontsize=7, color="#1f2328")
            axes[i].axis("off")

        for j in range(len(sample), len(axes)):
            axes[j].axis("off")

        title_str = f"Sample Images" + (f" — {label}" if label else "")
        fig.suptitle(title_str, color="#1f2328", fontsize=13)
        plt.tight_layout()
        self._finalise(fig, save_path, dpi=120)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — per-class sample grids
    # ─────────────────────────────────────────────────────────────────────

    def plot_class_samples(
        self,
        n_per_class: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """One row of sample images per class label."""
        self._check_loaded()
        plt = _plt(); cv2 = _cv2()

        classes = sorted(set(r.label for r in self._records
                             if r.label and not r.is_corrupt))
        if not classes:
            self._log("No labels found — use label_from_parent=True when loading.")
            return

        n_classes = len(classes)
        fig, axes = plt.subplots(
            n_classes, n_per_class,
            figsize=(n_per_class * 2.5, n_classes * 2.5),
            facecolor="white",
        )
        if n_classes == 1:
            axes = axes[np.newaxis, :]

        rng = np.random.default_rng(0)
        for row, cls in enumerate(classes):
            pool = [r for r in self._records
                    if r.label == cls and not r.is_corrupt
                    and not r.path.startswith("<array")]
            sample = rng.choice(pool, size=min(n_per_class, len(pool)),
                                replace=False)
            for col in range(n_per_class):
                ax = axes[row, col]
                ax.axis("off")
                ax.set_facecolor("white")
                if col >= len(sample):
                    continue
                img = cv2.imread(str(sample[col].path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = _resize_display(img, 128)
                ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(cls, color="#1f2328", fontsize=8,
                                  rotation=0, labelpad=50, va="center")

        fig.suptitle("Per-class Sample Images", color="#1f2328", fontsize=14)
        plt.tight_layout()
        self._finalise(fig, save_path, dpi=120)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — average image
    # ─────────────────────────────────────────────────────────────────────

    def plot_average_image(
        self,
        target_size: Tuple[int, int] = (224, 224),
        label: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compute and display the pixel-wise average image across the dataset.

        This reveals systematic dataset biases (e.g. sky always at top,
        ground at bottom, objects centred).
        """
        self._check_loaded()
        plt = _plt(); cv2 = _cv2()

        pool = [r for r in self._records
                if not r.is_corrupt and not r.path.startswith("<array")]
        if label:
            pool = [r for r in pool if r.label == label]

        if not pool:
            self._log("No images available for average-image computation.")
            return

        self._log(f"Computing average image over {len(pool)} images …")
        acc = np.zeros((*target_size, 3), dtype=np.float64)
        count = 0
        for rec in pool:
            img = cv2.imread(str(rec.path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_size[1], target_size[0]))
            acc += img.astype(np.float64)
            count += 1

        if count == 0:
            self._log("No images could be read.")
            return

        avg = (acc / count).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="white")
        axes[0].imshow(avg)
        axes[0].set_title("Average Image", color="#1f2328", fontsize=12)
        axes[0].axis("off")

        # standard deviation image
        self._log("Computing std-dev image …")
        sq_acc = np.zeros_like(acc)
        for rec in pool:
            img = cv2.imread(str(rec.path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_size[1], target_size[0]))
            sq_acc += (img.astype(np.float64) - acc / count) ** 2

        std_img = np.sqrt(sq_acc / count).astype(np.uint8)
        axes[1].imshow(std_img)
        axes[1].set_title("Std-Dev Image\n(high = high variance across dataset)",
                          color="#1f2328", fontsize=12)
        axes[1].axis("off")

        title = f"Dataset Average & Variance" + (f" — {label}" if label else "")
        fig.suptitle(title, color="#1f2328", fontsize=14)
        plt.tight_layout()
        self._finalise(fig, save_path, dpi=150)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — channel correlation
    # ─────────────────────────────────────────────────────────────────────

    def plot_channel_correlation(
        self,
        max_pixels: int = 50_000,
        save_path: Optional[str] = None,
    ) -> None:
        """R/G/B pairwise scatter matrix across a random pixel sample."""
        self._check_loaded()
        plt = _plt(); cv2 = _cv2()

        pool = [r for r in self._records if not r.is_corrupt
                and not r.path.startswith("<array")][:80]

        rng = np.random.default_rng(0)
        pixels = []
        for rec in pool:
            img = cv2.imread(str(rec.path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            flat = img.reshape(-1, 3)
            n = max_pixels // len(pool)
            if len(flat) > n:
                flat = flat[rng.choice(len(flat), n, replace=False)]
            pixels.append(flat)

        if not pixels:
            return
        pixels = np.vstack(pixels).astype(float)
        ch_names = ["Red", "Green", "Blue"]
        ch_colors = ["#ff6b6b", "#51cf66", "#339af0"]

        fig, axes = plt.subplots(3, 3, figsize=(11, 11), facecolor="white")
        for i in range(3):
            for j in range(3):
                a = axes[i, j]
                a.set_facecolor("#f6f8fa")
                for sp in a.spines.values():
                    sp.set_edgecolor("#d0d7de")
                a.tick_params(colors="#57606a", labelsize=7)
                if i == j:
                    a.hist(pixels[:, i], bins=60, color=ch_colors[i], alpha=0.85)
                    a.set_title(ch_names[i], color="#1f2328", fontsize=9)
                else:
                    a.scatter(pixels[:, j], pixels[:, i],
                              alpha=0.04, s=1, color=ch_colors[i])
                    corr = float(np.corrcoef(pixels[:, i], pixels[:, j])[0, 1])
                    a.set_title(f"r = {corr:.3f}", color="#57606a", fontsize=8)
                if i == 2:
                    a.set_xlabel(ch_names[j], color="#57606a", fontsize=8)
                if j == 0:
                    a.set_ylabel(ch_names[i], color="#57606a", fontsize=8)

        fig.suptitle("Channel Correlation Matrix", color="#1f2328", fontsize=14)
        plt.tight_layout()
        self._finalise(fig, save_path, dpi=130)

    # ─────────────────────────────────────────────────────────────────────
    # Plotting — duplicate viewer
    # ─────────────────────────────────────────────────────────────────────

    def plot_duplicates(
        self,
        mode: str = "near",
        max_groups: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualise duplicate / near-duplicate groups.

        Parameters
        ----------
        mode : ``"exact"`` | ``"near"``
        """
        self._check_loaded()
        plt = _plt(); cv2 = _cv2()

        valid = [r for r in self._records if not r.is_corrupt]
        groups = (self._find_exact_duplicates(valid) if mode == "exact"
                  else self._find_near_duplicates(valid))
        groups = groups[:max_groups]

        if not groups:
            self._log(f"No {mode} duplicates found.")
            return

        max_cols = max(len(g) for g in groups)
        fig, axes = plt.subplots(
            len(groups), max_cols,
            figsize=(max_cols * 2.5, len(groups) * 2.5),
            facecolor="white",
            squeeze=False,
        )

        for row, group in enumerate(groups):
            for col in range(max_cols):
                ax = axes[row, col]
                ax.axis("off")
                ax.set_facecolor("white")
                if col >= len(group):
                    continue
                img = cv2.imread(str(group[col]))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = _resize_display(img, 128)
                ax.imshow(img)
                ax.set_title(Path(group[col]).name[:18],
                             fontsize=6, color="#57606a")

        fig.suptitle(f"{mode.capitalize()} Duplicate Groups",
                     color="#1f2328", fontsize=13)
        plt.tight_layout()
        self._finalise(fig, save_path, dpi=120)

    # ─────────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────────

    def report(self, output_path: str = "viseda_report.html") -> str:
        """Generate a self-contained HTML report."""
        self._check_loaded()
        s = self.summary()
        norm = self.normalization_stats()
        s["normalization_stats"] = norm
        _generate_html_report(s, output_path)
        self._log(f"Report saved → {output_path}")
        return output_path

    # ─────────────────────────────────────────────────────────────────────
    # Per-image analysis
    # ─────────────────────────────────────────────────────────────────────

    def _analyse_single(self, path: Path, label_from_parent: bool) -> ImageRecord:
        cv2 = _cv2()
        rec = ImageRecord()
        rec.path = str(path)
        rec.file_ext = path.suffix.lower()
        rec.file_size_kb = path.stat().st_size / 1024

        if label_from_parent:
            rec.label = path.parent.name
        else:
            rec.label = self._labels_map.get(str(path.resolve()))

        img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            rec.is_corrupt = True
            return rec

        # Normalise to HxWx3 uint8
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._fill_stats(rec, img_rgb)
        return rec

    def _fill_stats(self, rec: ImageRecord, img: np.ndarray) -> None:
        cv2 = _cv2()

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img.astype(np.uint8)

        h, w = img.shape[:2]
        rec.height   = h
        rec.width    = w
        rec.channels = img.shape[2] if img.ndim == 3 else 1
        rec.dtype    = str(img.dtype)
        rec.aspect_ratio = round(w / h, 4)
        rec.megapixels   = round(h * w / 1_000_000, 4)

        f = img.astype(np.float32)

        # ── pixel stats ─────────────────────────────────────────────
        rec.mean_rgb = f.mean(axis=(0, 1)).tolist()
        rec.std_rgb  = f.std(axis=(0, 1)).tolist()
        rec.min_rgb  = f.min(axis=(0, 1)).tolist()
        rec.max_rgb  = f.max(axis=(0, 1)).tolist()

        # ── brightness & contrast (luminance) ────────────────────────
        gray = (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1]
                + 0.114 * f[:, :, 2])
        rec.brightness = float(gray.mean())
        rec.contrast   = float(gray.std())

        # ── sharpness: Laplacian variance ────────────────────────────
        gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
        rec.sharpness = float(lap.var())
        rec.is_blurry = rec.sharpness < self.blur_threshold

        # ── noise estimate: high-freq residual ───────────────────────
        blurred  = cv2.GaussianBlur(gray_u8, (5, 5), 0).astype(np.float32)
        residual = gray_u8.astype(np.float32) - blurred
        rec.noise_estimate = float(residual.std())

        # ── exposure ─────────────────────────────────────────────────
        rec.overexposed_frac  = float((gray > 245).mean())
        rec.underexposed_frac = float((gray < 10).mean())

        # ── entropy ──────────────────────────────────────────────────
        hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        rec.entropy_val = float(-np.sum(p * np.log2(p)))

        # ── JPEG blockiness score ────────────────────────────────────
        rec.compression_score = self._blockiness(gray_u8)

        # ── colour spaces ────────────────────────────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        rec.mean_hsv        = hsv.mean(axis=(0, 1)).tolist()
        rec.saturation_mean = float(hsv[:, :, 1].mean())
        rec.is_grayscale_like = rec.saturation_mean < 15.0

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
        rec.mean_lab = lab.mean(axis=(0, 1)).tolist()

        # colour temperature from R/B ratio
        r_mean = rec.mean_rgb[0]; b_mean = rec.mean_rgb[2]
        ratio = r_mean / (b_mean + 1e-6)
        if ratio > 1.1:   rec.color_temp = "warm"
        elif ratio < 0.9: rec.color_temp = "cool"
        else:             rec.color_temp = "neutral"

        # ── perceptual hash (pHash) ──────────────────────────────────
        small = cv2.resize(gray_u8, (32, 32),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)
        dct_low = dct[:8, :8].flatten()
        med = np.median(dct_low)
        rec.phash = int("".join("1" if v > med else "0"
                                for v in dct_low), 2)

        # average hash (aHash)
        tiny = cv2.resize(gray_u8, (8, 8),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        mean_tiny = tiny.mean()
        rec.ahash = int("".join("1" if v > mean_tiny else "0"
                                for v in tiny.flatten()), 2)

        # MD5 (for exact duplicates)
        raw = img.tobytes()
        rec.md5 = hashlib.md5(raw).hexdigest()

        # ── GLCM texture ─────────────────────────────────────────────
        if self.compute_glcm:
            self._fill_glcm(rec, gray_u8)

        # ── FFT frequency bands ───────────────────────────────────────
        if self.compute_freq:
            self._fill_frequency(rec, gray_u8)

    def _fill_glcm(self, rec: ImageRecord, gray_u8: np.ndarray) -> None:
        try:
            graycomatrix, graycoprops = _ski_feature()
            # Downsample for speed
            small = _cv2().resize(gray_u8, (128, 128))
            # Quantise to 64 levels
            quant = (small // 4).astype(np.uint8)
            glcm = graycomatrix(quant, distances=[1],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=64, symmetric=True, normed=True)
            rec.glcm_contrast      = float(graycoprops(glcm, "contrast").mean())
            rec.glcm_dissimilarity = float(graycoprops(glcm, "dissimilarity").mean())
            rec.glcm_homogeneity   = float(graycoprops(glcm, "homogeneity").mean())
            rec.glcm_energy        = float(graycoprops(glcm, "energy").mean())
            rec.glcm_correlation   = float(graycoprops(glcm, "correlation").mean())
            rec.glcm_asm           = float(graycoprops(glcm, "ASM").mean())
        except Exception:
            pass

    def _fill_frequency(self, rec: ImageRecord, gray_u8: np.ndarray) -> None:
        try:
            small = _cv2().resize(gray_u8, (128, 128)).astype(np.float32)
            fft   = np.fft.fft2(small)
            fft_s = np.fft.fftshift(np.abs(fft))
            h, w  = fft_s.shape
            cy, cx = h // 2, w // 2
            total = fft_s.sum() + 1e-9

            # Low = inner 10 %, Mid = 10–40 %, High = outer 40–50 %
            Y, X  = np.ogrid[:h, :w]
            dist  = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
            max_d = min(cy, cx)

            rec.freq_low  = float(fft_s[dist < max_d * 0.10].sum() / total)
            rec.freq_mid  = float(fft_s[(dist >= max_d * 0.10)
                                        & (dist < max_d * 0.40)].sum() / total)
            rec.freq_high = float(fft_s[dist >= max_d * 0.40].sum() / total)
        except Exception:
            pass

    @staticmethod
    def _blockiness(gray_u8: np.ndarray) -> float:
        """Estimate JPEG blockiness as mean absolute difference across 8-px boundaries."""
        h, w = gray_u8.shape
        g = gray_u8.astype(np.float32)
        scores = []
        for y in range(8, h, 8):
            scores.append(np.abs(g[y, :] - g[y - 1, :]).mean())
        for x in range(8, w, 8):
            scores.append(np.abs(g[:, x] - g[:, x - 1]).mean())
        return float(np.mean(scores)) if scores else 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Duplicate detection
    # ─────────────────────────────────────────────────────────────────────

    def _find_exact_duplicates(
        self, records: List[ImageRecord]
    ) -> List[List[str]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for r in records:
            if r.md5:
                groups[r.md5].append(r.path)
        return [g for g in groups.values() if len(g) > 1]

    def _find_near_duplicates(
        self, records: List[ImageRecord]
    ) -> List[List[str]]:
        groups: List[List[str]] = []
        visited: set = set()
        hashes = [(r.phash, r.path) for r in records if r.phash is not None]
        for i, (h1, p1) in enumerate(hashes):
            if p1 in visited:
                continue
            grp = [p1]
            for h2, p2 in hashes[i + 1:]:
                if p2 in visited:
                    continue
                if bin(h1 ^ h2).count("1") <= self.phash_threshold:
                    grp.append(p2)
                    visited.add(p2)
            if len(grp) > 1:
                groups.append(grp)
                visited.add(p1)
        return groups

    # ─────────────────────────────────────────────────────────────────────
    # Plot helpers
    # ─────────────────────────────────────────────────────────────────────

    def _plot_info_card(self, ax, valid):
        ax.axis("off")
        corrupt = len(self._records) - len(valid)
        blurry  = sum(r.is_blurry for r in valid)
        grey_like = sum(r.is_grayscale_like for r in valid)
        n_exact = len(self._find_exact_duplicates(valid))
        n_near  = len(self._find_near_duplicates(valid))
        lines = [
            f"Total images:       {len(self._records):,}",
            f"Valid / Corrupt:    {len(valid):,} / {corrupt:,}",
            f"Unique labels:      "
            f"{len(set(r.label for r in valid if r.label)):,}",
            f"Blurry images:      {blurry:,}  "
            f"({blurry/max(len(valid),1)*100:.1f}%)",
            f"Greyscale-like:     {grey_like:,}  "
            f"({grey_like/max(len(valid),1)*100:.1f}%)",
            f"Exact dupes groups: {n_exact:,}",
            f"Near dupes groups:  {n_near:,}",
            f"Median H × W:       "
            f"{int(np.median([r.height for r in valid]))} × "
            f"{int(np.median([r.width  for r in valid]))}",
        ]
        ax.text(0.04, 0.96, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color="#1f2328", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaeef2",
                          edgecolor="#d0d7de"))
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
        bars = ax.barh(y, counts, color="#58a6ff", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Label Distribution (top 25)", color="#1f2328", fontsize=11)
        ax.set_xlabel("Count", color="#57606a", fontsize=8)
        # imbalance ratio
        if len(counts) > 1:
            ratio = max(counts) / min(counts)
            ax.text(0.97, 0.02, f"imbalance ratio: {ratio:.1f}×",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#e3b341")

    def _plot_hist(self, ax, data, title, color, log_x=False):
        """Histogram with explicit axis metrics for interpretation."""
        vals = np.array([v for v in data if v is not None and np.isfinite(v)])
        if log_x:
            vals = vals[vals > 0]
            vals = np.log10(vals + 1e-9)
            x_label = f"log10({title})"
        else:
            x_label = title

        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Number of images")
            return

        ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(vals.mean(), color="#1f2328", ls="--", lw=1,
                   label=f"Mean = {vals.mean():.2f}")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Number of images")
        ax.legend(fontsize=7)

    def _plot_channel_histograms(self, ax, valid):
        means = np.array([r.mean_rgb for r in valid if r.mean_rgb])
        if not len(means):
            return
        colors = ["#ff6b6b", "#51cf66", "#339af0"]
        names  = ["Red", "Green", "Blue"]
        for i, (c, n) in enumerate(zip(colors, names)):
            ax.hist(means[:, i], bins=50, color=c, alpha=0.6,
                    label=n, edgecolor="none")
        ax.set_title("Per-channel Mean Distribution", color="#1f2328", fontsize=9)
        ax.set_xlabel("Pixel value (0–255)", color="#57606a", fontsize=8)
        ax.legend(fontsize=7, labelcolor="#1f2328",
                  facecolor="#eaeef2", edgecolor="#d0d7de")

    def _plot_dominant_colours(self, ax, valid):
        cv2 = _cv2()
        pool = [r for r in valid if not r.path.startswith("<array")][:60]
        all_pix = []
        for rec in pool:
            img = cv2.imread(str(rec.path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32))
            all_pix.append(img.reshape(-1, 3))
        if not all_pix:
            return
        pixels = np.vstack(all_pix)
        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=self.n_colors,
                                 random_state=42, n_init=3)
            if len(pixels) > 50_000:
                idx = np.random.choice(len(pixels), 50_000, replace=False)
                pixels = pixels[idx]
            labels = km.fit_predict(pixels)
            centers = km.cluster_centers_.astype(np.uint8)
            counts  = np.bincount(labels, minlength=self.n_colors)
            pct     = counts / counts.sum()
            order   = np.argsort(-pct)
            centers, pct = centers[order], pct[order]
            hex_cols = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
            ax.barh(np.arange(len(pct)), pct * 100,
                    color=hex_cols, edgecolor="#f6f8fa", height=0.7)
            ax.set_yticks(np.arange(len(pct)))
            ax.set_yticklabels(hex_cols, fontsize=8)
            ax.set_xlabel("% of pixels (sample)", color="#57606a", fontsize=8)
            ax.set_title("Dominant Colour Palette", color="#1f2328", fontsize=9)
        except Exception:
            pass

    def _plot_hsv_hue_wheel(self, ax, valid):
        cv2 = _cv2()
        sample = [r for r in valid if not r.path.startswith("<array")][:100]
        hues = []
        for rec in sample:
            img = cv2.imread(str(rec.path))
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            small = cv2.resize(hsv, (32, 32))
            hues.extend(small[:, :, 0].ravel().tolist())
        if not hues:
            return
        hues = np.array(hues) * 2  # OpenCV hue: 0-180 → 0-360
        counts, edges = np.histogram(hues, bins=36, range=(0, 360))
        centres = (edges[:-1] + edges[1:]) / 2
        colors_h = [_plt().cm.hsv(c / 360) for c in centres]
        ax.bar(centres, counts, width=10, color=colors_h, alpha=0.9)
        ax.set_title("Hue Distribution", color="#1f2328", fontsize=9)
        ax.set_xlabel("Hue (°)", color="#57606a", fontsize=8)

    def _plot_saturation_value(self, ax, valid):
        sats = [r.mean_hsv[1] if r.mean_hsv else None for r in valid]
        vals = [r.mean_hsv[2] if r.mean_hsv else None for r in valid]
        sats = [s for s in sats if s is not None]
        vals = [v for v in vals if v is not None]
        ax.scatter(sats, vals, alpha=0.3, s=6, color="#f9c74f", linewidths=0)
        ax.set_xlabel("Saturation", color="#57606a", fontsize=8)
        ax.set_ylabel("Value (brightness)", color="#57606a", fontsize=8)
        ax.set_title("HSV Saturation vs Value", color="#1f2328", fontsize=9)

    def _plot_lab_ab_scatter(self, ax, valid):
        labs = np.array([r.mean_lab for r in valid if r.mean_lab])
        if not len(labs):
            return
        ax.scatter(labs[:, 1], labs[:, 2], alpha=0.3, s=6,
                   c=labs[:, 0], cmap="viridis", linewidths=0)
        ax.axhline(0, color="#d0d7de", lw=0.8)
        ax.axvline(0, color="#d0d7de", lw=0.8)
        ax.set_xlabel("a* (green–red)", color="#57606a", fontsize=8)
        ax.set_ylabel("b* (blue–yellow)", color="#57606a", fontsize=8)
        ax.set_title("Lab Colour Space (a* vs b*)", color="#1f2328", fontsize=9)

    def _plot_color_temp_pie(self, ax, valid):
        ct = Counter(r.color_temp for r in valid if r.color_temp)
        if not ct:
            return
        colors = {"warm": "#ff6b6b", "neutral": "#57606a", "cool": "#339af0"}
        lbls = list(ct.keys())
        vals = [ct[l] for l in lbls]
        clrs = [colors.get(l, "#white") for l in lbls]
        wedges, _, autotexts = ax.pie(
            vals, labels=lbls, colors=clrs,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "#1f2328", "fontsize": 8},
        )
        ax.set_title("Colour Temperature", color="#1f2328", fontsize=9)

    def _plot_texture_radar(self, ax, valid):
        """Draw GLCM radar chart. ax is a plain Axes used only for position;
        we replace it with a polar subplot in the same grid slot."""
        feats = ["glcm_contrast", "glcm_homogeneity",
                 "glcm_energy", "glcm_correlation", "glcm_asm"]
        labels = ["Contrast", "Homogeneity", "Energy", "Correlation", "ASM"]
        vals = []
        for f in feats:
            a = [getattr(r, f) for r in valid if getattr(r, f) is not None]
            vals.append(np.mean(a) if a else 0.0)

        # Remove the plain axes and replace with a polar one at the same position
        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()

        if all(v == 0 for v in vals):
            ax2 = fig.add_axes(pos)
            ax2.set_facecolor("#f6f8fa")
            for sp in ax2.spines.values():
                sp.set_edgecolor("#d0d7de")
            ax2.text(0.5, 0.5, "GLCM not computed\n(install scikit-image)",
                     ha="center", va="center", color="#57606a",
                     transform=ax2.transAxes, fontsize=9)
            ax2.set_title("Texture Features (GLCM)", color="#1f2328", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
            return

        ax_polar = fig.add_axes(pos, projection="polar")
        ax_polar.set_facecolor("#f6f8fa")

        # normalise
        mx = max(vals) or 1.0
        norm_vals = [v / mx for v in vals]
        norm_vals += norm_vals[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        ax_polar.set_theta_offset(np.pi / 2)
        ax_polar.set_theta_direction(-1)
        ax_polar.plot(angles, norm_vals, color="#58a6ff", lw=2)
        ax_polar.fill(angles, norm_vals, color="#58a6ff", alpha=0.25)
        ax_polar.set_xticks(angles[:-1])
        ax_polar.set_xticklabels(labels, color="#1f2328", fontsize=7)
        ax_polar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_polar.set_yticklabels(["25%", "50%", "75%", "100%"],
                                  color="#57606a", fontsize=6)
        ax_polar.spines["polar"].set_edgecolor("#d0d7de")
        ax_polar.tick_params(colors="#57606a", labelsize=7)
        ax_polar.set_title("Texture Features (GLCM, normalised)",
                            color="#1f2328", fontsize=9, pad=12)

    def _plot_frequency_bands(self, ax, valid):
        low  = [r.freq_low  for r in valid if r.freq_low  is not None]
        mid  = [r.freq_mid  for r in valid if r.freq_mid  is not None]
        high = [r.freq_high for r in valid if r.freq_high is not None]
        if not low:
            ax.text(0.5, 0.5, "Frequency analysis\nnot computed",
                    ha="center", va="center", color="#57606a",
                    transform=ax.transAxes)
            ax.set_title("FFT Frequency Bands", color="#1f2328", fontsize=9)
            return
        x = np.arange(3)
        means = [np.mean(low), np.mean(mid), np.mean(high)]
        stds  = [np.std(low),  np.std(mid),  np.std(high)]
        colors = ["#58a6ff", "#3fb950", "#e3b341"]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
               edgecolor="none", capsize=4,
               error_kw=dict(ecolor="#1f2328", elinewidth=1))
        ax.set_xticks(x)
        ax.set_xticklabels(["Low\n(<10%)", "Mid\n(10–40%)", "High\n(>40%)"],
                           color="#57606a", fontsize=8)
        ax.set_title("FFT Frequency Band Energy\n(mean ± std)",
                     color="#1f2328", fontsize=9)
        ax.set_ylabel("Fraction of total energy", color="#57606a", fontsize=8)

    def _plot_brightness_sharpness_scatter(self, ax, valid):
        b = [r.brightness for r in valid]
        s = [r.sharpness  for r in valid]
        c = [r.contrast   for r in valid]
        sc = ax.scatter(b, s, c=c, cmap="plasma", s=6,
                        alpha=0.4, linewidths=0)
        ax.set_yscale("log")
        ax.set_xlabel("Brightness", color="#57606a", fontsize=8)
        ax.set_ylabel("Sharpness (log)", color="#57606a", fontsize=8)
        ax.set_title("Brightness vs Sharpness\n(colour = contrast)",
                     color="#1f2328", fontsize=9)
        _plt().colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                        label="Contrast").ax.tick_params(
            labelcolor="#57606a", labelsize=6)

    def _plot_exposure_scatter(self, ax, valid):
        over  = [r.overexposed_frac  * 100 for r in valid]
        under = [r.underexposed_frac * 100 for r in valid]
        ax.scatter(under, over, alpha=0.3, s=6,
                   color="#79c0ff", linewidths=0)
        ax.set_xlabel("Underexposed pixels (%)", color="#57606a", fontsize=8)
        ax.set_ylabel("Overexposed pixels (%)",  color="#57606a", fontsize=8)
        ax.set_title("Exposure Map", color="#1f2328", fontsize=9)

    def _plot_blurry_breakdown(self, ax, valid):
        sharp  = sum(not r.is_blurry for r in valid)
        blurry = sum(r.is_blurry     for r in valid)
        ax.bar(["Sharp", "Blurry"], [sharp, blurry],
               color=["#3fb950", "#f78166"], alpha=0.85, edgecolor="none")
        ax.set_title(f"Blurry Detection\n(threshold={self.blur_threshold})",
                     color="#1f2328", fontsize=9)
        for i, v in enumerate([sharp, blurry]):
            ax.text(i, v + max(sharp, blurry) * 0.02, str(v),
                    ha="center", va="bottom", color="#1f2328", fontsize=9)

    def _plot_format_dist(self, ax, valid):
        cnt = Counter(r.file_ext for r in valid)
        if not cnt:
            return
        names, counts = zip(*cnt.most_common())
        colors = ["#58a6ff", "#3fb950", "#e3b341", "#f78166",
                  "#d2a8ff", "#ffa657", "#79c0ff", "#56d364"]
        ax.bar(names, counts,
               color=colors[:len(names)], alpha=0.85, edgecolor="none")
        ax.set_title("File Format Distribution", color="#1f2328", fontsize=9)
        ax.set_xlabel("Extension", color="#57606a", fontsize=8)
        ax.set_ylabel("Count", color="#57606a", fontsize=8)

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

    @staticmethod
    def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[viseda] {msg}")

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() or .load_arrays() before analysing.")

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
    arr = np.asarray([x for x in arr if x is not None and np.isfinite(x)])
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


def _resize_display(img: np.ndarray, max_side: int) -> np.ndarray:
    cv2 = _cv2()
    h, w = img.shape[:2]
    scale = min(max_side / h, max_side / w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═════════════════════════════════════════════════════════════════════════════

def _generate_html_report(summary: Dict[str, Any], output_path: str) -> None:
    inv = summary.get("inventory", {})
    sp  = summary.get("spatial",   {})
    px  = summary.get("pixel_stats", {})
    qu  = summary.get("quality",   {})
    co  = summary.get("colour",    {})
    tx  = summary.get("texture",   {})
    fr  = summary.get("frequency", {})
    du  = summary.get("duplicates",{})
    lb  = summary.get("labels",    {})
    nm  = summary.get("normalization_stats", {})

    def card(title, stats):
        if not stats:
            return ""
        rows = "".join(
            f'<div class="stat"><span>{k}</span>'
            f'<span class="val">{_fmt(v)}</span></div>'
            for k, v in stats.items()
        )
        return f'<div class="card"><h3>{title}</h3>{rows}</div>'

    def badge(text, cls="blue"):
        return f'<span class="badge badge-{cls}">{text}</span>'

    def bar_chart(title, dist, span=1):
        if not dist:
            return ""
        total = sum(dist.values()) or 1
        mx = max(dist.values()) or 1
        rows = ""
        for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1])[:25]:
            pct = cnt / mx * 100
            rows += (
                f'<div class="bar-row">'
                f'<span class="bar-label" title="{lbl}">{lbl}</span>'
                f'<div class="bar">'
                f'<div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
                f'<span class="bar-count">{cnt:,}</span></div>'
            )
        span_style = f'grid-column: span {span};' if span > 1 else ''
        return (f'<div class="card" style="{span_style}">'
                f'<h3>{title}</h3><div class="bar-wrap">{rows}</div></div>')

    badges_html = (
        badge(f"{inv.get('total', 0):,} images") +
        badge(f"{inv.get('valid', 0):,} valid", "green") +
        (badge(f"{inv.get('corrupt', 0):,} corrupt", "red")
         if inv.get("corrupt") else "") +
        badge(f"{du.get('n_exact_duplicate_groups', 0)} exact dupes", "yellow") +
        badge(f"{du.get('n_near_duplicate_groups', 0)} near dupes", "yellow")
    )

    norm_html = ""
    if nm:
        m = [f"{v:.4f}" for v in nm.get("mean", [])]
        s = [f"{v:.4f}" for v in nm.get("std",  [])]
        norm_html = f"""
        <h2>📐 Normalisation Stats (for torchvision / transforms)</h2>
        <div class="card"><pre style="color:#58a6ff;font-size:0.85rem;">
transforms.Normalize(
    mean={m},
    std ={s}
)</pre></div>"""

    corrupt_html = ""
    if inv.get("corrupt_paths"):
        items = "".join(f"<li>{p}</li>"
                        for p in inv["corrupt_paths"][:50])
        corrupt_html = (f'<h2>⚠️ Corrupt Files</h2>'
                        f'<div class="card corrupt"><ul>{items}</ul></div>')

    # Pre-build dicts that can't go inside f-string {{ }} literals
    blurry_card     = card("Blurry Summary", {
        "Blurry count":    qu.get("blurry_count"),
        "Blurry fraction": qu.get("blurry_fraction"),
    })
    greyscale_card  = card("Greyscale-like", {
        "Count":    co.get("grayscale_like_count"),
        "Fraction": co.get("grayscale_like_fraction"),
    })
    duplicate_card  = card("Duplicate Summary", {
        "Exact duplicate groups": du.get("n_exact_duplicate_groups"),
        "Near duplicate groups":  du.get("n_near_duplicate_groups"),
    })
    class_imb_card  = card("Class Imbalance", {
        "Imbalance ratio (max/min)": lb.get("class_imbalance_ratio"),
    })
    counts_card     = card("Counts", {
        "Total":   inv.get("total"),
        "Valid":   inv.get("valid"),
        "Corrupt": inv.get("corrupt"),
    })
    mean_rgb_card   = card("Dataset Mean RGB (0-255)",
        dict(zip(["R mean", "G mean", "B mean"],
                 [round(v, 3) for v in px.get("dataset_mean_rgb", [])])))
    std_rgb_card    = card("Dataset Pixel Std RGB",
        dict(zip(["R std", "G std", "B std"],
                 [round(v, 3) for v in px.get("dataset_pixel_std_rgb", [])])))

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VisEDA Report</title>
<style>
:root{{--bg:white;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;
      --muted:#57606a;--accent:#58a6ff;--green:#3fb950;--red:#f78166;
      --yellow:#e3b341;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     padding:2rem;}}
h1{{font-size:1.9rem;margin-bottom:.25rem}}
h2{{font-size:1.05rem;color:var(--accent);margin:1.8rem 0 .6rem;}}
h3{{font-size:.78rem;color:var(--muted);text-transform:uppercase;
    letter-spacing:.05em;margin-bottom:.5rem}}
.sub{{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
       gap:.9rem}}
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
.badge-yellow{{background:rgba(227,179,65,.15);color:var(--yellow)}}
.bar-wrap{{margin-top:.4rem}}
.bar-row{{display:flex;align-items:center;gap:.4rem;margin:.18rem 0;
          font-size:.76rem}}
.bar-label{{width:120px;overflow:hidden;text-overflow:ellipsis;
            white-space:nowrap;color:var(--muted)}}
.bar{{flex:1;background:var(--border);border-radius:3px;height:9px}}
.bar-fill{{height:100%;border-radius:3px;background:var(--accent)}}
.bar-count{{width:55px;text-align:right;color:var(--accent)}}
.corrupt{{max-height:200px;overflow-y:auto;font-size:.78rem;
          color:var(--red)}}
pre{{background:#eaeef2;padding:.8rem;border-radius:6px;
     overflow-x:auto;font-size:.82rem}}
footer{{margin-top:3rem;color:var(--muted);font-size:.72rem;
        border-top:1px solid var(--border);padding-top:1rem}}
</style></head><body>
<h1>🔬 VisEDA — Image EDA Report</h1>
<p class="sub">Generated by <strong>VisEDA</strong></p>
<p style="margin-bottom:1rem">{badges_html}</p>

<h2>📦 Inventory</h2>
<div class="grid">
  {counts_card}
  {bar_chart("Format Distribution",  inv.get("format_distribution",  {}))}
  {bar_chart("Colour Mode",          inv.get("colour_mode_distribution", {}))}
</div>

<h2>📐 Spatial</h2>
<div class="grid">
  {card("Height (px)",      sp.get("height",       {}))}
  {card("Width (px)",       sp.get("width",        {}))}
  {card("Aspect Ratio",     sp.get("aspect_ratio", {}))}
  {card("Megapixels",       sp.get("megapixels",   {}))}
  {card("File Size (KB)",   sp.get("file_size_kb", {}))}
  {bar_chart("Orientation", sp.get("orientation_distribution", {}))}
</div>

<h2>🎨 Pixel Statistics</h2>
<div class="grid">
  {mean_rgb_card}
  {std_rgb_card}
</div>

{norm_html}

<h2>🔍 Quality Metrics</h2>
<div class="grid">
  {card("Brightness",        qu.get("brightness",       {}))}
  {card("Contrast",          qu.get("contrast",         {}))}
  {card("Sharpness",         qu.get("sharpness",        {}))}
  {card("Noise Estimate",    qu.get("noise_estimate",   {}))}
  {card("Pixel Entropy",     qu.get("entropy",          {}))}
  {card("Compression Score", qu.get("compression_score",{}))}
  {card("Overexposed Frac",  qu.get("overexposed_frac", {}))}
  {card("Underexposed Frac", qu.get("underexposed_frac",{}))}
  {blurry_card}
</div>

<h2>🌈 Colour</h2>
<div class="grid">
  {card("Saturation (HSV-S)", co.get("saturation", {}))}
  {bar_chart("Colour Temperature", co.get("colour_temp_distribution", {}))}
  {greyscale_card}
</div>

<h2>🧱 Texture (GLCM)</h2>
<div class="grid">
  {"".join(card(k.replace("_"," ").title(), v) for k, v in tx.items())}
</div>

<h2>〰️ Frequency (FFT)</h2>
<div class="grid">
  {"".join(card(k.replace("_"," ").title(), v) for k, v in fr.items())}
</div>

<h2>🔁 Duplicates</h2>
<div class="grid">
  {duplicate_card}
</div>

<h2>🏷️ Labels</h2>
<div class="grid">
  {bar_chart("Label Distribution",
             lb.get("label_distribution") or {}, span=2)}
  {class_imb_card}
</div>

{corrupt_html}

<footer>Generated by VisEDA — Visual Exploratory Data Analysis</footer>
</body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:,.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
