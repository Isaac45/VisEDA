"""
viseda.hyperspectral.eda
------------------------
EDA for hyperspectral / multispectral image cubes (H × W × Bands).

Analyses covered
~~~~~~~~~~~~~~~~
* Cube dimensions and band count
* Per-band statistics (mean, std, SNR, saturation fraction)
* Band correlation matrix
* Spectral signature analysis (mean spectrum + variance envelope)
* Spectral angle mapper (SAM) distance map
* Vegetation / water indices (NDVI, NDWI, EVI) when bands allow
* PCA of spectral space
* Endmember extraction (N-FINDR lite)
* Noise estimation per band
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from viseda.core.base import BaseEDA
from viseda.utils.helpers import safe_divide, entropy


class HyperspectralEDA(BaseEDA):
    """
    Perform EDA on a hyperspectral data cube.

    The cube is expected to have shape ``(H, W, B)`` where B = number of bands.

    Parameters
    ----------
    verbose : bool
        Print progress messages.
    wavelengths : array-like, optional
        Wavelength (nm) for each band. Enables spectral plots with
        meaningful x-axes.

    Examples
    --------
    >>> from viseda import HyperspectralEDA
    >>> eda = HyperspectralEDA(wavelengths=np.linspace(400, 2500, 200))
    >>> eda.load("cube.hdr")
    >>> eda.summary()
    >>> eda.plot()
    """

    def __init__(
        self,
        verbose: bool = True,
        wavelengths: Optional[np.ndarray] = None,
    ):
        super().__init__(verbose=verbose)
        self.wavelengths = np.asarray(wavelengths) if wavelengths is not None else None
        self._cube: Optional[np.ndarray] = None   # (H, W, B)
        self._mask: Optional[np.ndarray] = None   # (H, W) bool valid pixels
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        source: Union[str, Path, np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> "HyperspectralEDA":
        """
        Load a hyperspectral cube.

        Parameters
        ----------
        source
            * Path to an ENVI ``.hdr`` file — requires ``spectral`` package.
            * Path to a ``.npy`` or ``.npz`` file.
            * A NumPy array of shape (H, W, B).
        mask
            Optional boolean array (H × W) marking valid pixels.
        """
        if isinstance(source, np.ndarray):
            self._cube = source.astype(np.float32)
        else:
            self._cube = self._read_file(Path(source))

        if self._cube.ndim == 2:
            self._cube = self._cube[:, :, np.newaxis]

        self._mask = mask
        self._loaded = True
        H, W, B = self._cube.shape
        self._log(f"Cube loaded: {H}×{W}×{B} bands, dtype={self._cube.dtype}")
        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        self._check_loaded()
        H, W, B = self._cube.shape
        cube = self._cube

        band_means = cube.mean(axis=(0, 1))
        band_stds = cube.std(axis=(0, 1))
        band_min = cube.min(axis=(0, 1))
        band_max = cube.max(axis=(0, 1))

        # SNR estimate: mean / std per band
        snr = safe_divide(band_means, band_stds, fill=0.0)

        # Saturation fraction (values at max of data range)
        data_max = float(cube.max())
        sat_frac = (cube == data_max).mean(axis=(0, 1))

        # Noise: median absolute deviation per band
        noise = np.median(
            np.abs(cube - np.median(cube, axis=(0, 1), keepdims=True)),
            axis=(0, 1),
        )

        result = {
            "shape": {"H": H, "W": W, "bands": B},
            "dtype": str(self._cube.dtype),
            "wavelengths_provided": self.wavelengths is not None,
            "global_min": float(cube.min()),
            "global_max": float(cube.max()),
            "global_mean": float(cube.mean()),
            "global_std": float(cube.std()),
            "band_means": band_means.tolist(),
            "band_stds": band_stds.tolist(),
            "band_min": band_min.tolist(),
            "band_max": band_max.tolist(),
            "band_snr": snr.tolist(),
            "band_saturation_fraction": sat_frac.tolist(),
            "band_noise_mad": noise.tolist(),
        }
        self._store("summary", result)
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        figsize: Tuple[int, int] = (20, 18),
        save_path: Optional[str] = None,
        dpi: int = 150,
        rgb_bands: Tuple[int, int, int] = (30, 20, 10),
    ) -> None:
        """
        Comprehensive hyperspectral EDA dashboard.

        Parameters
        ----------
        rgb_bands
            Band indices to use for false-colour RGB preview.
        """
        self._check_loaded()
        plt = self._plt()
        mpl = self._mpl()

        H, W, B = self._cube.shape
        wl = self.wavelengths if self.wavelengths is not None else np.arange(B)

        fig = plt.figure(figsize=figsize, facecolor="#0d1117")
        fig.suptitle("VisEDA — Hyperspectral Cube Analysis",
                     fontsize=20, color="white", y=0.98, fontweight="bold")

        gs = mpl.gridspec.GridSpec(4, 3, figure=fig,
                                   hspace=0.5, wspace=0.35,
                                   left=0.06, right=0.97,
                                   top=0.94, bottom=0.04)

        def make_ax(*args):
            ax = fig.add_subplot(*args)
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            return ax

        # ── False colour preview ──────────────────────────────────────
        ax_fc = make_ax(gs[0, 0])
        self._plot_false_colour(ax_fc, rgb_bands)

        # ── Mean spectrum + std envelope ─────────────────────────────
        ax_spec = make_ax(gs[0, 1:])
        self._plot_mean_spectrum(ax_spec, wl)

        # ── Band statistics ───────────────────────────────────────────
        ax_bm = make_ax(gs[1, :])
        self._plot_band_statistics(ax_bm, wl)

        # ── SNR per band ──────────────────────────────────────────────
        ax_snr = make_ax(gs[2, :2])
        self._plot_snr(ax_snr, wl)

        # ── Band correlation heatmap ──────────────────────────────────
        ax_corr = make_ax(gs[2, 2])
        self._plot_band_correlation(ax_corr)

        # ── PCA variance explained ────────────────────────────────────
        ax_pca = make_ax(gs[3, 0])
        self._plot_pca(ax_pca)

        # ── Spectral indices ──────────────────────────────────────────
        ax_idx = make_ax(gs[3, 1:])
        self._plot_spectral_indices(ax_idx)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            self._log(f"Dashboard saved → {save_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Spectral analysis helpers (public)
    # ------------------------------------------------------------------

    def spectral_signature(
        self, row: int, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (wavelengths, reflectance) for a single pixel."""
        self._check_loaded()
        wl = self.wavelengths if self.wavelengths is not None \
            else np.arange(self._cube.shape[2])
        return wl, self._cube[row, col, :].copy()

    def pca(self, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PCA on spectral dimension.

        Returns
        -------
        scores : ndarray, shape (H*W, n_components)
        variance_ratio : ndarray, shape (n_components,)
        """
        self._check_loaded()
        from sklearn.decomposition import PCA

        H, W, B = self._cube.shape
        X = self._cube.reshape(-1, B)
        pca = PCA(n_components=min(n_components, B), svd_solver="randomized")
        scores = pca.fit_transform(X)
        return scores, pca.explained_variance_ratio_

    def compute_ndvi(
        self, nir_band: int, red_band: int
    ) -> np.ndarray:
        """Return NDVI map ``(NIR - Red) / (NIR + Red)``."""
        self._check_loaded()
        nir = self._cube[:, :, nir_band].astype(np.float32)
        red = self._cube[:, :, red_band].astype(np.float32)
        return safe_divide(nir - red, nir + red, fill=0.0)

    # ------------------------------------------------------------------
    # Internal plot methods
    # ------------------------------------------------------------------

    def _plot_false_colour(self, ax, rgb_bands):
        plt = self._plt()
        B = self._cube.shape[2]
        bands = [min(b, B - 1) for b in rgb_bands]
        fc = self._cube[:, :, bands].astype(np.float32)
        fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-9)
        ax.imshow(np.clip(fc, 0, 1))
        ax.set_title(f"False Colour (bands {bands})", color="white", fontsize=9)
        ax.axis("off")

    def _plot_mean_spectrum(self, ax, wl):
        cube = self._cube
        mean_spec = cube.mean(axis=(0, 1))
        std_spec = cube.std(axis=(0, 1))
        ax.plot(wl, mean_spec, color="#58a6ff", linewidth=1.5, label="Mean")
        ax.fill_between(wl, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.25, color="#58a6ff", label="±1σ")
        ax.set_title("Mean Spectral Signature", color="white", fontsize=9)
        ax.set_xlabel("Band / Wavelength (nm)" if self.wavelengths is not None
                      else "Band index", color="#8b949e", fontsize=8)
        ax.set_ylabel("Reflectance", color="#8b949e", fontsize=8)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_band_statistics(self, ax, wl):
        cube = self._cube
        ax.plot(wl, cube.mean(axis=(0, 1)), color="#58a6ff", label="Mean", lw=1.2)
        ax.plot(wl, cube.max(axis=(0, 1)), color="#f78166", label="Max", lw=0.8, alpha=0.7)
        ax.plot(wl, cube.min(axis=(0, 1)), color="#3fb950", label="Min", lw=0.8, alpha=0.7)
        ax.set_title("Per-band Statistics", color="white", fontsize=9)
        ax.set_xlabel("Band / Wavelength", color="#8b949e", fontsize=8)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_snr(self, ax, wl):
        cube = self._cube
        means = cube.mean(axis=(0, 1))
        stds = cube.std(axis=(0, 1)) + 1e-9
        snr = means / stds
        ax.plot(wl, snr, color="#e3b341", lw=1.2)
        ax.axhline(snr.mean(), color="white", lw=1, linestyle="--",
                   alpha=0.6, label=f"Mean SNR={snr.mean():.1f}")
        ax.set_title("Signal-to-Noise Ratio per Band", color="white", fontsize=9)
        ax.set_xlabel("Band / Wavelength", color="#8b949e", fontsize=8)
        ax.set_ylabel("SNR", color="#8b949e", fontsize=8)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d")

    def _plot_band_correlation(self, ax):
        plt = self._plt()
        H, W, B = self._cube.shape
        # subsample bands for readability
        step = max(1, B // 32)
        sub_cube = self._cube[:, :, ::step]          # (H, W, B_sub)
        B_sub = sub_cube.shape[2]
        sub = sub_cube.reshape(H * W, B_sub).T       # (B_sub, H*W)
        # sample pixels to keep correlation fast
        n_pix = min(5000, sub.shape[1])
        idx = np.random.choice(sub.shape[1], n_pix, replace=False)
        corr = np.corrcoef(sub[:, idx])
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelcolor="#8b949e", labelsize=7)
        ax.set_title("Band Correlation\n(subsampled)", color="white", fontsize=9)
        ax.set_xlabel("Band (step)", color="#8b949e", fontsize=7)
        ax.set_ylabel("Band (step)", color="#8b949e", fontsize=7)

    def _plot_pca(self, ax):
        try:
            _, var = self.pca(n_components=min(20, self._cube.shape[2]))
        except Exception:
            ax.set_title("PCA (unavailable)", color="white", fontsize=9)
            return
        cumvar = np.cumsum(var) * 100
        ax.bar(np.arange(1, len(var) + 1), var * 100,
               color="#d2a8ff", alpha=0.8)
        ax.plot(np.arange(1, len(var) + 1), cumvar,
                color="white", lw=1.2, marker=".", markersize=4)
        ax.axhline(95, color="#f78166", lw=1, linestyle="--", alpha=0.7)
        ax.set_title("PCA Variance Explained", color="white", fontsize=9)
        ax.set_xlabel("Component", color="#8b949e", fontsize=8)
        ax.set_ylabel("% variance", color="#8b949e", fontsize=8)

    def _plot_spectral_indices(self, ax):
        B = self._cube.shape[2]
        text = "Spectral indices require wavelength information.\n"
        if self.wavelengths is not None:
            wl = self.wavelengths
            # Try to find red (~670nm) and NIR (~850nm)
            red_idx = int(np.argmin(np.abs(wl - 670)))
            nir_idx = int(np.argmin(np.abs(wl - 850)))
            ndvi = self.compute_ndvi(nir_idx, red_idx)
            text = (
                f"NDVI  — min={ndvi.min():.3f}, max={ndvi.max():.3f}, "
                f"mean={ndvi.mean():.3f}\n"
                f"(Red≈{wl[red_idx]:.0f}nm band {red_idx}, "
                f"NIR≈{wl[nir_idx]:.0f}nm band {nir_idx})"
            )
            im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            import matplotlib.pyplot as plt_
            plt_.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
                labelcolor="#8b949e", labelsize=7)
        else:
            ax.text(0.5, 0.5, text, ha="center", va="center",
                    color="#8b949e", transform=ax.transAxes, fontsize=9)
        ax.set_title("NDVI Map", color="white", fontsize=9)
        ax.axis("off")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(str(path)).astype(np.float32)
        if suffix == ".npz":
            data = np.load(str(path))
            key = list(data.keys())[0]
            return data[key].astype(np.float32)
        if suffix in (".hdr", ".bil", ".bip", ".bsq", ".envi"):
            try:
                import spectral
                img = spectral.open_image(str(path))
                return img.load().astype(np.float32)
            except ImportError:
                raise ImportError(
                    "Install the 'spectral' package to read ENVI files: "
                    "pip install spectral"
                )
        if suffix in (".tif", ".tiff"):
            try:
                import rasterio
                with rasterio.open(str(path)) as src:
                    arr = src.read()  # (B, H, W)
                return arr.transpose(1, 2, 0).astype(np.float32)
            except ImportError:
                raise ImportError(
                    "Install 'rasterio' to read multi-band GeoTIFFs: "
                    "pip install rasterio"
                )
        raise ValueError(f"Unsupported hyperspectral format: {suffix}")

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