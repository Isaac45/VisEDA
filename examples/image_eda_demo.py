"""
examples/image_eda_demo.py
--------------------------
Demonstrates full ImageEDA workflow on a synthetic dataset.
Run with:  python examples/image_eda_demo.py
"""

import numpy as np
from viseda import ImageEDA

# ── 1. Create synthetic images in memory ────────────────────────────
rng = np.random.default_rng(0)
N = 200
classes = ["cats", "dogs", "birds", "cars"]
arrays = [rng.integers(0, 255, (rng.integers(64, 256), rng.integers(64, 256), 3),
                        dtype=np.uint8) for _ in range(N)]
labels = [classes[i % len(classes)] for i in range(N)]

# Inject a few "corrupt" arrays (wrong shapes / empty)
arrays[5] = np.array([])   # simulate corrupt
arrays[12] = np.array([])

# ── 2. Initialise and load ───────────────────────────────────────────
eda = ImageEDA(verbose=True, n_colors=8, phash_threshold=10)
eda.load_arrays(arrays, labels=labels)

# ── 3. Print summary ─────────────────────────────────────────────────
summary = eda.summary()
print("\n=== SUMMARY ===")
print(f"Total:          {summary['total_images']}")
print(f"Valid:          {summary['valid_images']}")
print(f"Corrupt:        {summary['corrupt_images']}")
print(f"Label dist:     {summary['label_distribution']}")
print(f"Brightness:     mean={summary['brightness']['mean']:.1f}  "
      f"std={summary['brightness']['std']:.1f}")
print(f"Contrast:       mean={summary['contrast']['mean']:.1f}")
print(f"Sharpness:      mean={summary['sharpness']['mean']:.1f}")
print(f"Duplicate grps: {summary['n_duplicate_groups']}")

# ── 4. Generate HTML report ──────────────────────────────────────────
eda.report("image_eda_report.html")
print("\nHTML report → image_eda_report.html")

# ── 5. Plot dashboard ────────────────────────────────────────────────
# eda.plot(save_path="image_dashboard.png")   # uncomment to save
# eda.plot()                                   # uncomment to show
print("Call eda.plot() to view the dashboard.")
