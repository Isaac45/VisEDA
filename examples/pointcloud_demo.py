"""
examples/pointcloud_demo.py
----------------------------
Demonstrates PointCloudEDA on a synthetic point cloud.
"""

import numpy as np
from viseda import PointCloudEDA

rng = np.random.default_rng(99)

# ── Simulate a building scene ────────────────────────────────────────
# Ground plane
ground = np.hstack([
    rng.uniform(-50, 50, (30_000, 2)),
    rng.normal(0, 0.05, (30_000, 1)),
])
# Building walls
wall = np.hstack([
    rng.uniform(5, 15, (10_000, 1)),
    rng.uniform(5, 15, (10_000, 1)),
    rng.uniform(0, 20, (10_000, 1)),
])
# Some noise / outliers
noise = rng.uniform(-60, 60, (500, 3))

xyz = np.vstack([ground, wall, noise]).astype(np.float32)
intensity = rng.random(len(xyz)).astype(np.float32) * 255
rgb = rng.integers(100, 220, (len(xyz), 3), dtype=np.uint8)
labels = np.concatenate([
    np.zeros(30_000, dtype=np.int32),   # ground
    np.ones(10_000, dtype=np.int32),    # building
    np.full(500, 2, dtype=np.int32),    # noise
])

# ── Load and summarise ───────────────────────────────────────────────
eda = PointCloudEDA(verbose=True, max_points=500_000)
eda.load(xyz, intensity=intensity, rgb=rgb, labels=labels)

s = eda.summary()
print(f"Points:           {s['n_points']:,}")
print(f"Z range:          {s['z']['min']:.2f} – {s['z']['max']:.2f} m")
print(f"Density:          {s['density_pts_per_unit3']:.4f} pts/m³")
print(f"Estimated outliers: {s['estimated_outliers']:,} ({s['outlier_fraction']*100:.1f}%)")
print(f"Label distribution: {s['label_distribution']}")

# eda.plot()      # full 2-D dashboard
# eda.plot_3d()   # interactive 3-D scatter
