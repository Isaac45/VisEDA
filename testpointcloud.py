"""
testpointcloud.py
=================
Complete test script for PointCloudEDA.

Usage
-----
Synthetic data only:
    python testpointcloud.py

With a directory of point clouds:
    python testpointcloud.py --dir C:/path/to/pointclouds --save-plots --quick

With a single file:
    python testpointcloud.py --file C:/path/to/cloud.npy --save-plots
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viseda.pointcloud.eda import PointCloudEDA


def section(title: str) -> None:
    bar = "═" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def ok(msg: str) -> None:
    print(f"  ✔  {msg}")


def info(msg: str) -> None:
    print(f"  ℹ  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def make_synthetic_cloud(n: int = 8000, seed: int = 0, scene: str = "sphere", with_rgb: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if scene == "sphere":
        phi = rng.uniform(0, 2 * np.pi, n)
        costheta = rng.uniform(-1, 1, n)
        theta = np.arccos(costheta)
        r = 1 + rng.normal(0, 0.03, n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    elif scene == "plane":
        x = rng.uniform(-5, 5, n)
        y = rng.uniform(-5, 5, n)
        z = 0.15 * x + 0.05 * y + rng.normal(0, 0.05, n)
    elif scene == "building":
        x = rng.uniform(-3, 3, n)
        y = rng.uniform(-3, 3, n)
        z = rng.choice([0, 2, 4, 6], n) + rng.normal(0, 0.04, n)
    elif scene == "linear":
        t = rng.uniform(-5, 5, n)
        x = t
        y = 0.15 * t + rng.normal(0, 0.04, n)
        z = -0.05 * t + rng.normal(0, 0.04, n)
    else:
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        z = rng.normal(0, 1, n)

    xyz = np.column_stack([x, y, z]).astype(np.float32)
    intensity = ((z - z.min()) / (z.max() - z.min() + 1e-9)).reshape(-1, 1).astype(np.float32)
    cloud = np.hstack([xyz, intensity])
    if with_rgb:
        rgb = np.column_stack([
            ((x - x.min()) / (x.max() - x.min() + 1e-9)) * 255,
            ((y - y.min()) / (y.max() - y.min() + 1e-9)) * 255,
            ((z - z.min()) / (z.max() - z.min() + 1e-9)) * 255,
        ]).astype(np.float32)
        cloud = np.hstack([cloud, rgb])
    return cloud.astype(np.float32)


def write_dataset(root: Path, n: int = 6) -> None:
    classes = ["bridge", "road", "building"]
    scenes = ["linear", "plane", "building"]
    for i in range(n):
        cls = classes[i % len(classes)]
        (root / cls).mkdir(parents=True, exist_ok=True)
        cloud = make_synthetic_cloud(6000 + i * 500, seed=i, scene=scenes[i % len(scenes)], with_rgb=i % 2 == 0)
        np.save(root / cls / f"cloud_{i:03d}.npy", cloud)


def write_text_cloud(path: Path, cloud: np.ndarray) -> None:
    np.savetxt(path, cloud[:, :4], fmt="%.6f")


def write_ascii_ply(path: Path, cloud: np.ndarray) -> None:
    xyz = cloud[:, :3]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for row in xyz:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")


def save_or_show(eda: PointCloudEDA, method: str, save: bool, out_dir: Path, **kwargs) -> None:
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{method}.png")
        kwargs["save_path"] = path
        getattr(eda, method)(**kwargs)
        ok(f"Saved → {path}")
    else:
        getattr(eda, method)(**kwargs)


def test_load_arrays(quick: bool) -> PointCloudEDA:
    section("1 · LOAD — NumPy arrays")
    scenes = ["sphere", "plane", "building", "linear"]
    labels = ["object", "terrain", "urban", "road"]
    clouds = [make_synthetic_cloud(7000, seed=i, scene=scenes[i % len(scenes)], with_rgb=i % 2 == 0) for i in range(8)]
    cloud_labels = [labels[i % len(labels)] for i in range(8)]
    eda = PointCloudEDA(verbose=True, compute_neighbors=not quick, compute_geometry=True)
    t0 = time.perf_counter()
    eda.load_arrays(clouds, labels=cloud_labels)
    elapsed = time.perf_counter() - t0
    assert len(eda._records) == 8
    assert sum(not r.is_corrupt for r in eda._records) == 8
    ok(f"Loaded 8 point clouds in {elapsed:.2f}s")
    ok(f"Labels: {sorted(set(cloud_labels))}")
    return eda


def test_summary(eda: PointCloudEDA) -> None:
    section("2 · SUMMARY")
    s = eda.summary()
    inv = s["inventory"]
    assert inv["valid_clouds"] > 0
    assert inv["point_count"]["mean"] > 0
    ok(f"Total: {inv['total_clouds']}  Valid: {inv['valid_clouds']}  Corrupt: {inv['corrupt_clouds']}")
    ok(f"Format distribution: {inv['format_distribution']}")
    ok(f"Label distribution: {inv['label_distribution']}")
    ok(f"Mean points: {inv['point_count']['mean']}")
    geo = s["geometry"]
    ok(f"Mean bbox volume: {geo['bbox_volume'].get('mean', 'N/A')}")
    ok(f"Mean density: {geo['density'].get('mean', 'N/A')}")
    q = s["quality"]
    ok(f"Duplicate fraction: {q['duplicate_fraction'].get('mean', 'N/A')}")
    ok(f"Outlier fraction: {q['outlier_fraction'].get('mean', 'N/A')}")
    shape = s["shape_descriptors"]
    ok(f"Linearity: {shape['linearity'].get('mean', 'N/A')}")
    ok(f"Planarity: {shape['planarity'].get('mean', 'N/A')}")


def test_per_record_fields(eda: PointCloudEDA) -> None:
    section("3 · PER-RECORD FIELD VALIDATION")
    required = [
        "n_points", "n_dims", "xyz_min", "xyz_max", "xyz_mean", "xyz_std",
        "bbox_size", "bbox_volume", "density", "height_mean", "height_std",
        "finite_fraction", "duplicate_fraction", "outlier_fraction", "linearity",
        "planarity", "scattering", "curvature",
    ]
    valid = [r for r in eda._records if not r.is_corrupt]
    for rec in valid[:5]:
        for field in required:
            assert getattr(rec, field) is not None, f"{field} missing for {rec.path}"
        assert rec.n_points > 0
        assert len(rec.bbox_size) == 3
    ok(f"All {len(required)} fields populated on {min(5, len(valid))} sampled records")


def test_load_directory(tmpdir: Path, quick: bool) -> PointCloudEDA:
    section("4 · LOAD — Directory of .npy files")
    write_dataset(tmpdir, n=6)
    eda = PointCloudEDA(verbose=True, compute_neighbors=not quick, compute_geometry=True)
    eda.load(tmpdir, label_from_parent=True)
    valid = [r for r in eda._records if not r.is_corrupt]
    assert len(valid) == 6
    assert len(set(r.label for r in valid)) >= 2
    ok(f"Loaded {len(valid)} clouds from directory")
    ok(f"Labels: {sorted(set(r.label for r in valid))}")
    return eda


def test_file_formats(tmpdir: Path, quick: bool) -> None:
    section("5 · FILE FORMATS — npy, npz, txt, xyz, ply")
    # Ensure the temporary formats directory exists before writing files.
    tmpdir.mkdir(parents=True, exist_ok=True)
    cloud = make_synthetic_cloud(2500, seed=42, scene="plane", with_rgb=True)
    np.save(tmpdir / "sample.npy", cloud)
    np.savez(tmpdir / "sample_npz.npz", points=cloud)
    write_text_cloud(tmpdir / "sample.txt", cloud)
    write_text_cloud(tmpdir / "sample.xyz", cloud[:, :3])
    write_ascii_ply(tmpdir / "sample.ply", cloud)
    paths = [tmpdir / "sample.npy", tmpdir / "sample_npz.npz", tmpdir / "sample.txt", tmpdir / "sample.xyz", tmpdir / "sample.ply"]
    eda = PointCloudEDA(verbose=False, compute_neighbors=not quick, compute_geometry=True)
    eda.load(paths)
    valid = [r for r in eda._records if not r.is_corrupt]
    assert len(valid) == len(paths), [(r.path, r.error) for r in eda._records]
    ok(f"Loaded {len(valid)} supported point cloud file formats")


def test_get_cloud_and_record(eda: PointCloudEDA) -> None:
    section("6 · GET CLOUD / RECORD")
    rec = eda.get_record(0)
    cloud = eda.get_cloud(0)
    assert cloud.shape[0] == rec.n_points
    assert cloud.shape[1] == rec.n_dims
    ok(f"Record and cloud retrieved: shape={cloud.shape}")


def test_pairwise_distances(eda: PointCloudEDA) -> None:
    section("7 · PAIRWISE CLOUD DISTANCES")
    dist, names = eda.pairwise_cloud_distances()
    assert dist.shape[0] == dist.shape[1] == len(names)
    assert np.allclose(np.diag(dist), 0)
    ok(f"Distance matrix computed: {dist.shape}")


def test_plots(eda: PointCloudEDA, save: bool, out_dir: Path) -> None:
    section("8 · PLOTS")
    save_or_show(eda, "plot_dataset", save, out_dir)
    ok("Dataset dashboard rendered")
    save_or_show(eda, "plot", save, out_dir, cloud_index=0)
    ok("Single-cloud dashboard rendered")
    save_or_show(eda, "plot_clouds_grid", save, out_dir, n=6)
    ok("Cloud preview grid rendered")
    save_or_show(eda, "plot_height_distribution", save, out_dir)
    ok("Height distribution rendered")
    save_or_show(eda, "plot_pairwise_cloud_distances", save, out_dir)
    ok("Pairwise distance heatmap rendered")


def test_report(eda: PointCloudEDA, report_path: str) -> None:
    section("9 · HTML REPORT")
    path = eda.report(report_path)
    assert Path(path).exists()
    ok(f"Report saved → {path} ({Path(path).stat().st_size / 1024:.1f} KB)")


def test_edge_cases() -> None:
    section("10 · EDGE CASES")
    fresh = PointCloudEDA(verbose=False)
    try:
        fresh.summary()
        assert False, "summary() should fail before load"
    except RuntimeError:
        ok("RuntimeError raised before load()")

    eda = PointCloudEDA(verbose=False, max_clouds=2, compute_neighbors=False, compute_geometry=False)
    clouds = [make_synthetic_cloud(500, seed=i) for i in range(4)]
    eda.load_arrays(clouds)
    assert len(eda._records) == 2
    ok("max_clouds respected for arrays")

    bad = np.array([[np.nan, 1, 2], [np.inf, 2, 3]], dtype=np.float32)
    eda2 = PointCloudEDA(verbose=False)
    eda2.load_arrays([bad])
    assert eda2._records[0].is_corrupt
    ok("Invalid all-nonfinite cloud marked corrupt")

    one_point = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    eda3 = PointCloudEDA(verbose=False)
    eda3.load_arrays([one_point])
    s = eda3.summary()
    assert s["inventory"]["valid_clouds"] == 1
    ok("Single-point cloud handled correctly")


def test_user_file(file_path: str, quick: bool, save: bool, out_dir: Path, report_path: str) -> None:
    section("11 · USER-SUPPLIED FILE")
    eda = PointCloudEDA(verbose=True, compute_neighbors=not quick, compute_geometry=True)
    eda.load(file_path, label_from_parent=True)
    valid = [r for r in eda._records if not r.is_corrupt]
    if not valid:
        warn("No valid point clouds loaded from the supplied file. "
             "If this is a LAS/LAZ file, install laspy first: pip install laspy")
        return
    test_summary(eda)
    # Generate the final report from the user-supplied file, not the synthetic test data.
    eda.report(report_path)
    ok(f"User file report saved → {report_path}")
    if save:
        save_or_show(eda, "plot", save, out_dir, cloud_index=eda._records.index(valid[0]))


def test_user_directory(dir_path: str, quick: bool, save: bool, out_dir: Path, report_path: str) -> None:
    section("12 · USER-SUPPLIED DIRECTORY")
    eda = PointCloudEDA(verbose=True, compute_neighbors=not quick, compute_geometry=True)
    eda.load(dir_path, label_from_parent=True, recursive=True)
    valid = [r for r in eda._records if not r.is_corrupt]
    if not valid:
        warn("No valid point clouds loaded from the supplied directory. "
             "If the directory contains LAS/LAZ files, install laspy first: pip install laspy")
        return
    test_summary(eda)
    # Generate the final report from the user-supplied directory, not the synthetic test data.
    eda.report(report_path)
    ok(f"User directory report saved → {report_path}")
    if save:
        save_or_show(eda, "plot_dataset", save, out_dir)
        save_or_show(eda, "plot_clouds_grid", save, out_dir, n=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete PointCloudEDA test suite")
    parser.add_argument("--file", type=str, default=None, help="Optional single point cloud file")
    parser.add_argument("--dir", type=str, default=None, help="Optional directory of point cloud files")
    parser.add_argument("--save-plots", action="store_true", help="Save plots instead of displaying")
    parser.add_argument("--quick", action="store_true", help="Skip slower nearest-neighbour calculations")
    parser.add_argument("--report", type=str, default="viseda_pointcloud_report.html")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  VisEDA — PointCloudEDA Complete Test Suite                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\n  Save plots : {args.save_plots}")
    print(f"  Quick mode : {args.quick}")
    print(f"  File       : {args.file}")
    print(f"  Directory  : {args.dir}")

    out_dir = Path("outputs_pointcloud")
    passed = 0
    failed = 0
    t0 = time.perf_counter()

    def run(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"\n  ✘  TEST FAILED: {name}")
            import traceback
            traceback.print_exc()

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        eda_arrays_holder = {}
        eda_dir_holder = {}

        run("load_arrays", lambda: eda_arrays_holder.setdefault("eda", test_load_arrays(args.quick)))
        eda_arrays = eda_arrays_holder.get("eda")
        if eda_arrays:
            run("summary_arrays", lambda: test_summary(eda_arrays))
            run("record_fields", lambda: test_per_record_fields(eda_arrays))
            run("get_cloud", lambda: test_get_cloud_and_record(eda_arrays))
            run("pairwise", lambda: test_pairwise_distances(eda_arrays))
            run("plots_arrays", lambda: test_plots(eda_arrays, args.save_plots, out_dir))
            run("report", lambda: test_report(eda_arrays, args.report))

        run("load_directory", lambda: eda_dir_holder.setdefault("eda", test_load_directory(tmpdir / "dataset", args.quick)))
        eda_dir = eda_dir_holder.get("eda")
        if eda_dir:
            run("summary_directory", lambda: test_summary(eda_dir))
            run("pairwise_directory", lambda: test_pairwise_distances(eda_dir))

        run("file_formats", lambda: test_file_formats(tmpdir / "formats", args.quick))
        run("edge_cases", test_edge_cases)

    if args.file:
        run("user_file", lambda: test_user_file(args.file, args.quick, args.save_plots, out_dir, args.report))
    if args.dir:
        run("user_directory", lambda: test_user_directory(args.dir, args.quick, args.save_plots, out_dir, args.report))

    elapsed = time.perf_counter() - t0
    print("\n" + "═" * 62)
    print(f"  Results:  {passed} passed  |  {failed} failed  |  {elapsed:.1f}s total")
    if args.save_plots:
        print(f"  Plots saved to: {out_dir.resolve()}/")
    print(f"  Report: {args.report}")
    print("═" * 62)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
