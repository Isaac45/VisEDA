"""
examples/hyperspectral_demo.py
------------------------------
Demonstrates HyperspectralEDA on a synthetic cube.
"""

import numpy as np
from viseda import HyperspectralEDA

# ── Synthetic 128×128×200 cube ───────────────────────────────────────
rng = np.random.default_rng(42)
H, W, B = 128, 128, 200
cube = rng.random((H, W, B)).astype(np.float32)

# Add a simulated "vegetation" region with higher NIR reflectance
cube[40:80, 40:80, 140:] *= 3.0     # bright NIR patch

wavelengths = np.linspace(400, 2500, B)   # VNIR-SWIR range

# ── Load & analyse ───────────────────────────────────────────────────
eda = HyperspectralEDA(verbose=True, wavelengths=wavelengths)
eda.load(cube)

summary = eda.summary()
print(f"Shape:       {summary['shape']}")
print(f"Bands:       {summary['shape']['bands']}")
print(f"Global mean: {summary['global_mean']:.4f}")
print(f"Global std:  {summary['global_std']:.4f}")

# Spectral signature of a single pixel
wl, sig = eda.spectral_signature(60, 60)
print(f"\nSpectral sig at (60,60):  min={sig.min():.3f}  max={sig.max():.3f}")

# NDVI
ndvi = eda.compute_ndvi(nir_band=140, red_band=60)
print(f"NDVI:  min={ndvi.min():.3f}  max={ndvi.max():.3f}  mean={ndvi.mean():.3f}")

# PCA
scores, var = eda.pca(n_components=10)
print(f"PCA:   top-3 components explain "
      f"{var[:3].sum()*100:.1f}% variance")

# eda.plot()   # uncomment to view dashboard
