import h5py, numpy as np
from viseda import PointCloudEDA

# Each .h5 file has keys: 'data' (B, 2048, 3) and 'label' (B, 1)
with h5py.File("modelnet40_ply_hdf5_2048/ply_data_train0.h5", "r") as f:
    points = f["data"][:]    # shape (2048_samples, 2048_pts, 3)
    labels = f["label"][:]   # shape (2048_samples, 1)

# Analyse all points from the first 100 shapes
xyz = points[:100].reshape(-1, 3).astype(np.float32)
lbl = np.repeat(labels[:100].ravel(), 2048).astype(np.int32)

eda = PointCloudEDA(verbose=True, max_points=500_000)
eda.load(xyz, labels=lbl)
eda.summary()
eda.plot()
eda.plot_3d(color_by="label")