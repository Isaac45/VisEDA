"""
tests/test_image_full.py
========================
Complete test script for ImageEDA using a cats-and-dogs dataset.

Expected directory structure
-----------------------------
    dataset/
        cats/
            cat.001.jpg
            cat.002.jpg
            ...
        dogs/
            dog.001.jpg
            dog.002.jpg
            ...

Usage
-----
    # Run all tests and generate all outputs
    python tests/test_image_full.py --data path/to/dataset

    # Quick run (limit images, skip slow tests)
    python tests/test_image_full.py --data path/to/dataset --max 200 --quick

    # Save all plots instead of showing them
    python tests/test_image_full.py --data path/to/dataset --save-plots

Arguments
---------
    --data          Path to the dataset directory            [required]
    --max           Max images to load (default: all)
    --save-plots    Save plots to ./outputs/ instead of showing
    --quick         Skip slow tests (average image, UMAP)
    --report        Path for the HTML report (default: viseda_report.html)
    --blur-thresh   Sharpness threshold for blurry detection (default: 50)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ── Add project root to path so we can import viseda without installing ──────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viseda.image.eda import ImageEDA


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    """Print a clearly visible section header."""
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def ok(msg: str) -> None:
    print(f"  ✔  {msg}")


def info(msg: str) -> None:
    print(f"  ℹ  {msg}")


def save_or_show(eda, method: str, save: bool, out_dir: Path, **kwargs):
    """Call an EDA plot method, either saving or showing."""
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{method}.png")
        kwargs["save_path"] = path
        getattr(eda, method)(**kwargs)
        ok(f"Saved → {path}")
    else:
        getattr(eda, method)(**kwargs)


# ════════════════════════════════════════════════════════════════════════════
# Test functions
# ════════════════════════════════════════════════════════════════════════════

def test_load(eda: ImageEDA, data_path: str) -> None:
    section("1 · LOADING")

    t0 = time.perf_counter()
    eda.load(data_path, label_from_parent=True, recursive=True)
    elapsed = time.perf_counter() - t0

    valid   = [r for r in eda._records if not r.is_corrupt]
    corrupt = [r for r in eda._records if r.is_corrupt]

    assert len(eda._records) > 0, "No images found — check --data path."
    ok(f"Loaded {len(eda._records):,} images in {elapsed:.1f}s")
    ok(f"Valid: {len(valid):,}   Corrupt: {len(corrupt):,}")

    # Labels
    labels = set(r.label for r in valid if r.label)
    if len(labels) < 2:
        info(f"Only {len(labels)} class(es) found: {sorted(labels)}")
        info("Tip: remove --max or increase it so all sub-folders are loaded.")
    else:
        ok(f"Classes found: {sorted(labels)}")


def test_summary(eda: ImageEDA) -> None:
    section("2 · SUMMARY")

    s = eda.summary()

    # ── inventory ──────────────────────────────────────────────────────
    inv = s["inventory"]
    assert inv["valid"] > 0
    ok(f"Total: {inv['total']:,}  Valid: {inv['valid']:,}  "
       f"Corrupt: {inv['corrupt']:,}")
    ok(f"Formats: {inv['format_distribution']}")
    ok(f"Colour modes: {inv['colour_mode_distribution']}")

    # ── spatial ────────────────────────────────────────────────────────
    sp = s["spatial"]
    assert sp["height"]["mean"] > 0
    assert sp["width"]["mean"]  > 0
    ok(f"Height  — mean: {sp['height']['mean']:.0f}  "
       f"min: {sp['height']['min']:.0f}  max: {sp['height']['max']:.0f}")
    ok(f"Width   — mean: {sp['width']['mean']:.0f}  "
       f"min: {sp['width']['min']:.0f}  max: {sp['width']['max']:.0f}")
    ok(f"Aspect  — mean: {sp['aspect_ratio']['mean']:.3f}")
    ok(f"MP      — mean: {sp['megapixels']['mean']:.3f}")
    ok(f"Orientations: {sp['orientation_distribution']}")

    # ── quality ────────────────────────────────────────────────────────
    qu = s["quality"]
    assert 0 <= qu["brightness"]["mean"] <= 255
    ok(f"Brightness — mean: {qu['brightness']['mean']:.1f}  "
       f"std: {qu['brightness']['std']:.1f}")
    ok(f"Contrast   — mean: {qu['contrast']['mean']:.1f}")
    ok(f"Sharpness  — mean: {qu['sharpness']['mean']:.1f}  "
       f"(log scale recommended)")
    ok(f"Noise est  — mean: {qu['noise_estimate']['mean']:.3f}")
    ok(f"Entropy    — mean: {qu['entropy']['mean']:.3f} bits")
    ok(f"Blurry:    {qu['blurry_count']:,} images "
       f"({qu['blurry_fraction']*100:.1f}%)")
    ok(f"Overexp:   mean {qu['overexposed_frac']['mean']*100:.2f}%")
    ok(f"Underexp:  mean {qu['underexposed_frac']['mean']*100:.2f}%")

    # ── colour ─────────────────────────────────────────────────────────
    co = s["colour"]
    ok(f"Saturation — mean: {co['saturation']['mean']:.1f}")
    ok(f"Colour temp: {co['colour_temp_distribution']}")
    ok(f"Greyscale-like: {co['grayscale_like_count']:,} "
       f"({co['grayscale_like_fraction']*100:.1f}%)")

    # ── pixel stats ────────────────────────────────────────────────────
    px = s["pixel_stats"]
    assert len(px["dataset_mean_rgb"]) == 3
    ok(f"Dataset mean RGB: "
       f"R={px['dataset_mean_rgb'][0]:.1f}  "
       f"G={px['dataset_mean_rgb'][1]:.1f}  "
       f"B={px['dataset_mean_rgb'][2]:.1f}")

    # ── texture ────────────────────────────────────────────────────────
    tx = s["texture"]
    if tx:
        ok(f"GLCM contrast   — mean: "
           f"{tx.get('glcm_contrast',   {}).get('mean', 'N/A')}")
        ok(f"GLCM homogeneity— mean: "
           f"{tx.get('glcm_homogeneity',{}).get('mean', 'N/A')}")
        ok(f"GLCM energy     — mean: "
           f"{tx.get('glcm_energy',     {}).get('mean', 'N/A')}")
    else:
        info("GLCM not computed (install scikit-image for texture features)")

    # ── frequency ──────────────────────────────────────────────────────
    fr = s["frequency"]
    if fr:
        ok(f"FFT low  — mean: {fr.get('freq_low',  {}).get('mean', 'N/A'):.4f}")
        ok(f"FFT mid  — mean: {fr.get('freq_mid',  {}).get('mean', 'N/A'):.4f}")
        ok(f"FFT high — mean: {fr.get('freq_high', {}).get('mean', 'N/A'):.4f}")

    # ── duplicates ─────────────────────────────────────────────────────
    du = s["duplicates"]
    ok(f"Exact duplicate groups: {du['n_exact_duplicate_groups']:,}")
    ok(f"Near  duplicate groups: {du['n_near_duplicate_groups']:,}")
    if du["n_near_duplicate_groups"] > 0:
        grp = du["near_duplicate_groups"][0]
        info(f"  Example near-dupe group ({len(grp)} images):")
        for p in grp[:3]:
            info(f"    {p}")

    # ── labels ─────────────────────────────────────────────────────────
    lb = s["labels"]
    assert lb["label_distribution"] is not None
    ok(f"Label distribution: {lb['label_distribution']}")
    ok(f"Class imbalance ratio: {lb['class_imbalance_ratio']}")


def test_normalization_stats(eda: ImageEDA) -> None:
    section("3 · NORMALISATION STATS")

    stats = eda.normalization_stats()
    assert "mean" in stats and "std" in stats
    assert len(stats["mean"]) == 3
    assert len(stats["std"])  == 3
    assert all(0.0 <= v <= 1.0 for v in stats["mean"])
    assert all(0.0 <= v <= 1.0 for v in stats["std"])

    ok(f"mean = {[round(v, 4) for v in stats['mean']]}")
    ok(f"std  = {[round(v, 4) for v in stats['std']]}")
    info("Use these values in torchvision.transforms.Normalize()")


def test_main_dashboard(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("4 · MAIN EDA DASHBOARD  (plot)")
    save_or_show(eda, "plot", save, out_dir)
    ok("Main dashboard rendered")


def test_colour_dashboard(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("5 · COLOUR DASHBOARD  (plot_colour)")
    save_or_show(eda, "plot_colour", save, out_dir)
    ok("Colour dashboard rendered")


def test_quality_dashboard(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("6 · QUALITY DASHBOARD  (plot_quality)")
    save_or_show(eda, "plot_quality", save, out_dir)
    ok("Quality dashboard rendered")


def test_texture_dashboard(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("7 · TEXTURE DASHBOARD  (plot_texture)")
    save_or_show(eda, "plot_texture", save, out_dir)
    ok("Texture dashboard rendered")


def test_sample_grid(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("8 · SAMPLE GRID  (plot_samples)")

    # All classes
    save_or_show(eda, "plot_samples", save, out_dir,
                 n=25, cols=5)
    ok("All-class sample grid rendered")

    # Per class
    labels = sorted(set(r.label for r in eda._records
                        if r.label and not r.is_corrupt))
    for lbl in labels[:3]:
        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = str(out_dir / f"plot_samples_{lbl}.png")
            eda.plot_samples(n=10, cols=5, label=lbl, save_path=path)
            ok(f"Saved samples for '{lbl}' → {path}")
        else:
            eda.plot_samples(n=10, cols=5, label=lbl)
            ok(f"Samples for '{lbl}' rendered")


def test_class_samples(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("9 · PER-CLASS SAMPLE GRID  (plot_class_samples)")
    save_or_show(eda, "plot_class_samples", save, out_dir,
                 n_per_class=6)
    ok("Per-class sample grid rendered")


def test_average_image(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("10 · AVERAGE & STD-DEV IMAGE  (plot_average_image)")

    # Full dataset
    save_or_show(eda, "plot_average_image", save, out_dir,
                 target_size=(224, 224))
    ok("Full-dataset average image rendered")

    # Per class
    labels = sorted(set(r.label for r in eda._records
                        if r.label and not r.is_corrupt))
    for lbl in labels:
        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = str(out_dir / f"plot_average_image_{lbl}.png")
            eda.plot_average_image(target_size=(224, 224),
                                   label=lbl, save_path=path)
            ok(f"Average image for '{lbl}' → {path}")
        else:
            eda.plot_average_image(target_size=(224, 224), label=lbl)
            ok(f"Average image for '{lbl}' rendered")


def test_channel_correlation(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("11 · CHANNEL CORRELATION MATRIX  (plot_channel_correlation)")
    save_or_show(eda, "plot_channel_correlation", save, out_dir)
    ok("Channel correlation matrix rendered")


def test_duplicates(eda: ImageEDA, save: bool, out_dir: Path) -> None:
    section("12 · DUPLICATE VIEWER  (plot_duplicates)")

    du = eda.summary()["duplicates"]

    if du["n_exact_duplicate_groups"] > 0:
        save_or_show(eda, "plot_duplicates", save, out_dir,
                     mode="exact", max_groups=5)
        ok("Exact duplicate viewer rendered")
    else:
        info("No exact duplicates found in this dataset")

    if du["n_near_duplicate_groups"] > 0:
        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = str(out_dir / "plot_duplicates_near.png")
            eda.plot_duplicates(mode="near", max_groups=5, save_path=path)
            ok(f"Near-duplicate viewer → {path}")
        else:
            eda.plot_duplicates(mode="near", max_groups=5)
            ok("Near-duplicate viewer rendered")
    else:
        info("No near-duplicates found in this dataset")


def test_html_report(eda: ImageEDA, report_path: str) -> None:
    section("13 · HTML REPORT  (report)")
    path = eda.report(report_path)
    assert Path(path).exists(), f"Report file not created at {path}"
    size_kb = Path(path).stat().st_size / 1024
    ok(f"Report saved → {path}  ({size_kb:.1f} KB)")
    info("Open in a browser to explore the interactive report")


def test_per_image_records(eda: ImageEDA) -> None:
    """Validate that all expected fields are populated on ImageRecord objects."""
    section("14 · PER-IMAGE RECORD VALIDATION")

    valid = [r for r in eda._records if not r.is_corrupt]
    sample = valid[:min(20, len(valid))]

    required_fields = [
        "height", "width", "aspect_ratio", "megapixels",
        "brightness", "contrast", "sharpness", "noise_estimate",
        "entropy_val", "overexposed_frac", "underexposed_frac",
        "mean_rgb", "std_rgb", "mean_hsv", "mean_lab",
        "saturation_mean", "color_temp", "phash", "ahash", "md5",
    ]

    missing_counts = {f: 0 for f in required_fields}
    for rec in sample:
        for f in required_fields:
            if getattr(rec, f) is None:
                missing_counts[f] += 1

    all_ok = True
    for f, cnt in missing_counts.items():
        if cnt > 0:
            info(f"Field '{f}' missing on {cnt}/{len(sample)} sampled records")
            all_ok = False

    if all_ok:
        ok(f"All {len(required_fields)} fields populated on "
           f"{len(sample)} sampled records")

    # Type checks
    for rec in sample[:3]:
        assert isinstance(rec.height, int)
        assert isinstance(rec.width,  int)
        assert isinstance(rec.mean_rgb, list) and len(rec.mean_rgb) == 3
        assert rec.color_temp in ("warm", "neutral", "cool")
        assert isinstance(rec.is_blurry, bool)

    ok("Type checks passed")


def test_edge_cases(eda: ImageEDA) -> None:
    """Test edge cases and guard conditions."""
    section("15 · EDGE CASES")

    # Summary is idempotent
    s1 = eda.summary()
    s2 = eda.summary()
    assert s1["inventory"]["total"] == s2["inventory"]["total"]
    ok("Summary is idempotent (multiple calls return same result)")

    # Not-loaded guard
    fresh = ImageEDA(verbose=False)
    try:
        fresh.summary()
        assert False, "Should have raised"
    except RuntimeError:
        ok("RuntimeError raised when calling summary() before load()")

    # Empty label list still works
    no_label = ImageEDA(verbose=False)
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
              for _ in range(5)]
    no_label.load_arrays(arrays)
    s = no_label.summary()
    assert s["labels"]["label_distribution"] is None
    ok("load_arrays() without labels works correctly")

    # Normalisation stats shape
    norm = eda.normalization_stats()
    assert len(norm["mean"]) == 3 and len(norm["std"]) == 3
    ok("normalization_stats() returns correct shape")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="VisEDA ImageEDA — complete test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", required=True,
                   help="Path to dataset directory (e.g. path/to/cats_dogs/)")
    p.add_argument("--max", type=int, default=None,
                   help="Max images to load (default: all)")
    p.add_argument("--save-plots", action="store_true",
                   help="Save plots to ./outputs_img/ instead of displaying")
    p.add_argument("--quick", action="store_true",
                   help="Skip slow tests (average image computation)")
    p.add_argument("--report", default="viseda_report.html",
                   help="Output path for HTML report")
    p.add_argument("--blur-thresh", type=float, default=50.0,
                   help="Laplacian variance threshold for blurry detection")
    p.add_argument("--no-glcm", action="store_true",
                   help="Skip GLCM texture computation (faster)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path("outputs_img")

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  VisEDA — ImageEDA Complete Test Suite" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Dataset : {args.data}")
    print(f"  Max imgs: {args.max or 'all'}")
    print(f"  Save    : {args.save_plots}")
    print(f"  Quick   : {args.quick}")

    # ── Initialise EDA ─────────────────────────────────────────────────
    eda = ImageEDA(
        verbose=True,
        max_images=args.max,
        n_colors=8,
        phash_threshold=10,
        blur_threshold=args.blur_thresh,
        compute_glcm=not args.no_glcm,
        compute_freq=True,
    )

    total_start = time.perf_counter()
    passed = 0
    failed = 0

    tests = [
        ("load",               lambda: test_load(eda, args.data)),
        ("summary",            lambda: test_summary(eda)),
        ("normalization_stats",lambda: test_normalization_stats(eda)),
        ("main_dashboard",     lambda: test_main_dashboard(eda, args.save_plots, out_dir)),
        ("colour_dashboard",   lambda: test_colour_dashboard(eda, args.save_plots, out_dir)),
        ("quality_dashboard",  lambda: test_quality_dashboard(eda, args.save_plots, out_dir)),
        ("texture_dashboard",  lambda: test_texture_dashboard(eda, args.save_plots, out_dir)),
        ("sample_grid",        lambda: test_sample_grid(eda, args.save_plots, out_dir)),
        ("class_samples",      lambda: test_class_samples(eda, args.save_plots, out_dir)),
        ("average_image",      lambda: test_average_image(eda, args.save_plots, out_dir)),
        ("channel_correlation",lambda: test_channel_correlation(eda, args.save_plots, out_dir)),
        ("duplicates",         lambda: test_duplicates(eda, args.save_plots, out_dir)),
        ("html_report",        lambda: test_html_report(eda, args.report)),
        ("per_image_records",  lambda: test_per_image_records(eda)),
        ("edge_cases",         lambda: test_edge_cases(eda)),
    ]

    # Skip slow tests if --quick
    slow_tests = {"average_image"}
    if args.quick:
        tests = [(n, f) for n, f in tests if n not in slow_tests]
        info("Quick mode: skipping slow tests " + str(slow_tests))

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✘  TEST FAILED: {name}")
            print(f"     {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ── Final summary ──────────────────────────────────────────────────
    elapsed = time.perf_counter() - total_start
    print("\n" + "═" * 60)
    print(f"  Results:  {passed} passed  |  {failed} failed  "
          f"|  {elapsed:.1f}s total")
    if args.save_plots:
        print(f"  Plots saved to: {out_dir.resolve()}/")
    print(f"  Report:   {args.report}")
    print("═" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()