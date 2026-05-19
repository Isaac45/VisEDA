"""
viseda.utils
------------
Shared utility helpers used across all EDA modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HYPERSPECTRAL_EXTENSIONS = {".hdr", ".bil", ".bip", ".bsq", ".envi", ".tif", ".tiff"}
POINTCLOUD_EXTENSIONS = {".las", ".laz", ".ply", ".pcd", ".xyz", ".txt", ".npy", ".npz"}


def discover_files(
    root: Union[str, Path],
    extensions: set,
    recursive: bool = True,
) -> List[Path]:
    """Walk *root* and return all files matching *extensions*."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in extensions else []
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in extensions
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise division that replaces zero-denominator with *fill*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b != 0, a / b, fill)
    return out


def percentile_clip(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Clip array to [lo, hi] percentiles and normalise to [0, 1]."""
    lo_val = np.percentile(arr, lo)
    hi_val = np.percentile(arr, hi)
    clipped = np.clip(arr, lo_val, hi_val)
    return safe_divide(clipped - lo_val, hi_val - lo_val, fill=0.0)


def entropy(hist: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert HxWx3 float/uint8 RGB to HxW grayscale (ITU-R BT.601)."""
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def dominant_colors(img: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means dominant colour extraction.
    Returns (centers, percentages) both sorted by dominance descending.
    """
    from sklearn.cluster import MiniBatchKMeans

    pixels = img.reshape(-1, img.shape[-1]).astype(np.float32)
    # subsample for speed on large images
    if len(pixels) > 50_000:
        idx = np.random.choice(len(pixels), 50_000, replace=False)
        pixels = pixels[idx]

    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.uint8)
    counts = np.bincount(labels, minlength=k)
    pct = counts / counts.sum()
    order = np.argsort(-pct)
    return centers[order], pct[order]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float images in [0,1] to uint8 [0,255]."""
    if img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def resize_for_display(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Downscale large images for display purposes."""
    import cv2  # lazy import

    h, w = img.shape[:2]
    scale = min(max_side / h, max_side / w, 1.0)
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img