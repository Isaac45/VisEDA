import numpy as np
from viseda import HyperspectralDatasetEDA, PointCloudDatasetEDA

# ── Collection of hyperspectral cubes ──────────────────────────────
eda = HyperspectralDatasetEDA(
    wavelengths=np.linspace(400, 2500, 200)  # optional but recommended
)

# From a directory of .npy / .hdr / .mat / .tif files
eda.load("C:/Users/Ike/Desktop/testhyper/", label_from_parent=True)

# Or from a list of paths
eda.load(["Indian_pines_corrected.mat", "PaviaU.mat"])
Indian_pines_corrected = np.random.rand(145, 145, 200)
PaviaU = np.random.rand(610, 340, 200)
# Or directly from NumPy arrays
eda.load_arrays([Indian_pines_corrected, PaviaU], labels=["farmland", "Bridge"])

eda.summary()   # cross-cube stats, spectral diversity, band distributions
eda.plot()      # dataset-level dashboard


# # ── Collection of point clouds ─────────────────────────────────────
# eda2 = PointCloudDatasetEDA(max_points_per_cloud=200_000)
#
# # From a directory of .las / .ply / .xyz / .npy files
# eda2.load("path/to/clouds/", label_from_parent=True)
#
# eda2.summary()  # point counts, density distributions, aggregated class labels
# eda2.plot()     # dataset-level dashboard