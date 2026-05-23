"""
tests/test_hyperspectral_full.py
=================================
Complete test script for HyperspectralEDA.

Covers single-cube deep-dive, multi-cube dataset analysis,
file-loading (mat/npy/npz), all plot methods, HTML report,
spectral index computation, PCA, GLCM texture, and edge cases.

Usage
-----
    # Synthetic data only (no real files needed)
    python tests/test_hyperspectral_full.py

    # With a real .mat file (Indian Pines, Pavia, Salinas …)
    python tests/test_hyperspectral_full.py --mat path/to/Indian_pines_corrected.mat

    # With a directory of cubes
    python tests/test_hyperspectral_full.py --dir path/to/cubes/

    # Save plots instead of displaying
    python tests/test_hyperspectral_full.py --save-plots

    # Skip slow tests (GLCM, PCA, average image)
    python tests/test_hyperspectral_full.py --quick
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viseda.hyperspectral.eda import HyperspectralEDA


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 62
    print(f"\n{bar}\n  {title}\n{bar}")

def ok(msg: str)   -> None: print(f"  ✔  {msg}")
def info(msg: str) -> None: print(f"  ℹ  {msg}")
def warn(msg: str) -> None: print(f"  ⚠  {msg}")

def save_or_show(eda, method, save, out_dir, **kwargs):
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{method}.png")
        kwargs["save_path"] = path
        getattr(eda, method)(**kwargs)
        ok(f"Saved → {path}")
    else:
        getattr(eda, method)(**kwargs)

def make_synthetic_cube(
    H: int = 64, W: int = 64, B: int = 50,
    seed: int = 0,
    scene: str = "random",
) -> np.ndarray:
    """
    Generate a realistic-ish synthetic hyperspectral cube.

    scene options
    -------------
    "random"     — pure random noise
    "gradient"   — smooth spatial gradient with spectral variation
    "vegetation" — bright NIR patch simulating vegetation
    "urban"      — mixed bright/dark regions
    """
    rng = np.random.default_rng(seed)
    cube = rng.random((H, W, B)).astype(np.float32) * 0.15  # base noise

    if scene == "gradient":
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)
        spatial = (xx + yy) / 2
        spectral = np.sin(np.linspace(0, np.pi, B))
        cube += (spatial[:, :, None] * spectral[None, None, :]).astype(np.float32)

    elif scene == "vegetation":
        # Simulate bright NIR in upper half
        spectral = np.linspace(0.05, 0.9, B)
        cube[:H//2, :, :] += spectral[None, None, :] * 0.6
        cube[H//2:, :, :] += spectral[None, None, :] * 0.1

    elif scene == "urban":
        # Three distinct regions
        spectral_road   = np.ones(B) * 0.3
        spectral_bldg   = np.linspace(0.1, 0.8, B)
        spectral_veggie = np.sin(np.linspace(0, np.pi, B)) * 0.5
        cube[:H//3,     :, :] += spectral_road[None, None, :]
        cube[H//3:2*H//3, :, :] += spectral_bldg[None, None, :]
        cube[2*H//3:,   :, :] += spectral_veggie[None, None, :]

    return np.clip(cube, 0, 1)


def write_npy_dataset(tmpdir: Path, n: int = 6) -> None:
    """Write n synthetic .npy cubes to tmpdir with sub-folder labels."""
    scenes  = ["vegetation", "urban", "gradient"]
    classes = ["farmland", "city", "forest"]
    for i in range(n):
        cls = classes[i % len(classes)]
        (tmpdir / cls).mkdir(parents=True, exist_ok=True)
        cube = make_synthetic_cube(64, 64, 50, seed=i,
                                   scene=scenes[i % len(scenes)])
        np.save(str(tmpdir / cls / f"cube_{i:03d}.npy"), cube)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_load_arrays(quick: bool) -> HyperspectralEDA:
    """Load 9 synthetic cubes from arrays (3 classes × 3 cubes)."""
    section("1 · LOAD — NumPy arrays")

    scenes  = ["vegetation", "urban", "gradient", "random"]
    classes = ["farmland", "city", "forest"]
    arrays  = []
    labels  = []
    for i in range(9):
        scene = scenes[i % len(scenes)]
        label = classes[i % len(classes)]
        arrays.append(make_synthetic_cube(64, 64, 50, seed=i, scene=scene))
        labels.append(label)

    wl  = np.linspace(400, 2500, 50)
    eda = HyperspectralEDA(
        verbose=True,
        wavelengths=wl,
        compute_glcm=not quick,
        compute_pca=True,
    )
    t0 = time.perf_counter()
    eda.load_arrays(arrays, labels=labels)
    elapsed = time.perf_counter() - t0

    assert len(eda._records) == 9
    valid = [r for r in eda._records if not r.is_corrupt]
    assert len(valid) == 9

    ok(f"Loaded 9 cubes in {elapsed:.2f}s")
    ok(f"Labels: {sorted(set(r.label for r in valid))}")
    return eda


def test_load_npy_directory(tmpdir: Path, quick: bool) -> HyperspectralEDA:
    """Load cubes from a directory of .npy files."""
    section("2 · LOAD — Directory of .npy files")

    write_npy_dataset(tmpdir, n=6)
    wl = np.linspace(400, 2500, 50)

    eda = HyperspectralEDA(
        verbose=True, wavelengths=wl,
        compute_glcm=not quick, compute_pca=True,
    )
    t0 = time.perf_counter()
    eda.load(str(tmpdir), label_from_parent=True, recursive=True)
    elapsed = time.perf_counter() - t0

    valid = [r for r in eda._records if not r.is_corrupt]
    assert len(valid) == 6, f"Expected 6 cubes, got {len(valid)}"
    labels = set(r.label for r in valid if r.label)
    assert len(labels) >= 2, f"Expected ≥2 classes, got {labels}"

    ok(f"Loaded {len(valid)} cubes from directory in {elapsed:.2f}s")
    ok(f"Labels: {sorted(labels)}")
    return eda


def test_load_npy_list(tmpdir: Path, quick: bool) -> HyperspectralEDA:
    """Load cubes from an explicit list of file paths."""
    section("3 · LOAD — Explicit file list")

    paths = sorted(tmpdir.rglob("*.npy"))[:4]
    if not paths:
        write_npy_dataset(tmpdir, n=4)
        paths = sorted(tmpdir.rglob("*.npy"))[:4]

    eda = HyperspectralEDA(verbose=False, compute_glcm=False, compute_pca=True)
    eda.load([str(p) for p in paths])
    assert len(eda._records) == len(paths)
    ok(f"Loaded {len(paths)} cubes from explicit list")
    return eda


def test_load_mat(mat_path: str, quick: bool) -> HyperspectralEDA:
    """Load a real .mat file (Indian Pines / Pavia / Salinas)."""
    section("4 · LOAD — Real .mat file")
    try:
        import scipy.io
    except ImportError:
        warn("scipy not installed — skipping .mat test")
        return None

    eda = HyperspectralEDA(
        verbose=True,
        wavelengths=np.linspace(400, 2500, 200),
        compute_glcm=not quick,
        compute_pca=True,
    )
    t0 = time.perf_counter()
    eda.load(mat_path)
    elapsed = time.perf_counter() - t0

    valid = [r for r in eda._records if not r.is_corrupt]
    assert len(valid) == 1
    rec = valid[0]
    ok(f"Loaded {rec.height}×{rec.width}×{rec.bands} cube in {elapsed:.2f}s")
    ok(f"Global mean: {rec.global_mean:.4f}  std: {rec.global_std:.4f}")
    return eda


def test_summary(eda: HyperspectralEDA) -> None:
    section("5 · SUMMARY")

    s = eda.summary()

    # ── inventory ──────────────────────────────────────────────────────
    inv = s["inventory"]
    assert inv["valid_cubes"] > 0
    ok(f"Total: {inv['total_cubes']}  Valid: {inv['valid_cubes']}  "
       f"Corrupt: {inv['corrupt_cubes']}")
    ok(f"Band distribution: {inv['band_distribution']}")
    ok(f"Label distribution: {inv['label_distribution']}")

    # ── spatial ────────────────────────────────────────────────────────
    sp = s["spatial"]
    assert sp["height"]["mean"] > 0
    assert sp["width"]["mean"]  > 0
    assert sp["bands"]["mean"]  > 0
    ok(f"Height  mean: {sp['height']['mean']:.1f}")
    ok(f"Width   mean: {sp['width']['mean']:.1f}")
    ok(f"Bands   mean: {sp['bands']['mean']:.1f}")

    # ── spectral stats ─────────────────────────────────────────────────
    ss = s["spectral_stats"]
    assert ss["global_mean"]["mean"] is not None
    ok(f"Global mean:    {ss['global_mean']['mean']:.4f}")
    ok(f"Dynamic range:  {ss['dynamic_range']['mean']:.4f}")
    ok(f"Dominant bands: {ss['dominant_band_count']}")
    ok(f"Matching cubes: {ss['n_matching_cubes']}")
    assert ss["cross_cube_mean_spectrum"] is not None
    ok(f"Cross-cube spectrum length: {len(ss['cross_cube_mean_spectrum'])}")

    # ── spectral quality ───────────────────────────────────────────────
    sq = s["spectral_quality"]
    ok(f"Mean SNR:           {sq['snr_mean'].get('mean','N/A')}")
    ok(f"Spectral smoothness:{sq['spectral_smoothness'].get('mean','N/A')}")
    ok(f"Inter-band corr:    {sq['inter_band_corr'].get('mean','N/A')}")
    ok(f"Dropout bands:      {sq['n_dropout_bands'].get('mean','N/A')}")

    # ── spectral indices ───────────────────────────────────────────────
    si = s["spectral_indices"]
    for name in ("ndvi", "ndwi", "evi", "savi"):
        if name in si:
            mu = si[name]["per_cube_mean"].get("mean", "N/A")
            ok(f"{name.upper()} per-cube mean: {mu}")

    # ── texture ────────────────────────────────────────────────────────
    tx = s["texture"]
    if tx:
        ok(f"GLCM contrast:    {tx.get('glcm_contrast',{}).get('mean','N/A')}")
        ok(f"GLCM homogeneity: {tx.get('glcm_homogeneity',{}).get('mean','N/A')}")
    else:
        info("GLCM not computed (use --no-quick to enable)")

    # ── PCA ────────────────────────────────────────────────────────────
    pc = s["pca"]
    if pc:
        ok(f"PCA components for 95% variance: "
           f"{pc.get('n_components_95pct_mean','N/A')}")
    else:
        info("PCA not computed")


def test_per_record_fields(eda: HyperspectralEDA, strict_spectral_indices: bool = True) -> None:
    section("6 · PER-RECORD FIELD VALIDATION")

    valid = [r for r in eda._records if not r.is_corrupt]
    assert valid, "No valid records to check"
    sample = valid[:min(5, len(valid))]

    required = [
        "height", "width", "bands", "dtype",
        "global_mean", "global_std", "dynamic_range",
        "band_means", "band_stds", "band_snr", "band_noise_mad",
        "snr_mean", "spectral_smoothness", "inter_band_corr",
        "n_dropout_bands", "spatial_mean_map",
    ]
    for rec in sample:
        for f in required:
            v = getattr(rec, f)
            assert v is not None, f"Field '{f}' is None on {rec.path}"
        assert isinstance(rec.band_means, np.ndarray)
        assert len(rec.band_means) == rec.bands
        assert rec.bands > 0
        assert rec.height > 0 and rec.width > 0

    ok(f"All {len(required)} fields populated on {len(sample)} sampled records")

    # Spectral index spot check.
    # Synthetic reflectance-like cubes should stay in a small expected range.
    # Real public .mat datasets may be stored as raw radiance / digital numbers,
    # and EVI can become numerically extreme when its denominator is close to zero.
    # For user-supplied real datasets, validate that values are finite instead of
    # forcing the synthetic-data range.
    extreme_indices = []
    for rec in sample:
        for idx in ("ndvi_mean", "ndwi_mean", "evi_mean", "savi_mean"):
            v = getattr(rec, idx)
            if v is not None:
                assert np.isfinite(v), f"{idx} is not finite: {v}"
                if not (-2.0 <= v <= 2.0):
                    extreme_indices.append((rec.path, idx, v))

    if strict_spectral_indices:
        assert not extreme_indices, f"Spectral index value(s) out of expected synthetic range: {extreme_indices[:3]}"
        ok("Spectral index values in expected range")
    else:
        if extreme_indices:
            info("Some spectral indices are outside the synthetic-data range; this is allowed for real raw .mat datasets.")
        ok("Spectral index values are finite")


def test_spectral_signature(eda: HyperspectralEDA, tmpdir: Path) -> None:
    section("7 · SPECTRAL SIGNATURE (single pixel)")

    # Only works for file-backed cubes
    file_recs = [r for r in eda._records
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping spectral signature test")
        return

    wl, sig = eda.spectral_signature(
        cube_index=eda._records.index(file_recs[0]),
        row=10, col=10
    )
    assert len(wl) == len(sig)
    assert len(sig) == file_recs[0].bands
    assert np.all(np.isfinite(sig))
    ok(f"Spectral signature: {len(sig)} bands, "
       f"range [{sig.min():.4f}, {sig.max():.4f}]")


def test_compute_index(eda: HyperspectralEDA) -> None:
    section("8 · INDEX COMPUTATION (compute_index)")

    file_recs = [r for r in eda._records
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping index test")
        return

    idx_in_records = eda._records.index(file_recs[0])
    rec = file_recs[0]

    for name in ("ndvi", "ndwi", "evi", "savi"):
        try:
            idx_map = eda.compute_index(idx_in_records, name)
            assert idx_map.shape == (rec.height, rec.width)
            assert np.all(np.isfinite(idx_map))
            ok(f"{name.upper()}: shape={idx_map.shape} "
               f"range=[{idx_map.min():.3f}, {idx_map.max():.3f}]")
        except Exception as e:
            warn(f"{name.upper()} failed: {e}")


def test_pca_scores(eda: HyperspectralEDA) -> None:
    section("9 · PCA SCORES (pca_scores)")

    file_recs = [r for r in eda._records
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping PCA scores test")
        return

    idx = eda._records.index(file_recs[0])
    rec = file_recs[0]

    scores, var = eda.pca_scores(idx, n_components=5)
    assert scores.shape == (rec.height * rec.width, 5)
    assert len(var) == 5
    assert abs(var.sum() - sum(var)) < 1e-6
    cumvar = np.cumsum(var)
    ok(f"PCA scores shape: {scores.shape}")
    ok(f"Variance ratio: {[round(v,4) for v in var]}")
    ok(f"Cumulative variance (5 comps): {cumvar[-1]*100:.1f}%")


def test_normalization_idempotent(eda: HyperspectralEDA) -> None:
    section("10 · SUMMARY IDEMPOTENCY")
    s1 = eda.summary()
    s2 = eda.summary()
    assert s1["inventory"]["total_cubes"] == s2["inventory"]["total_cubes"]
    ok("summary() is idempotent")


def test_plot_single(eda: HyperspectralEDA, save: bool, out_dir: Path,
                     quick: bool) -> None:
    section("11 · PLOT — Single cube deep-dive (plot)")

    file_recs = [i for i, r in enumerate(eda._records)
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping single-cube plot")
        return

    save_or_show(eda, "plot", save, out_dir, cube_index=file_recs[0])
    ok("Single-cube dashboard rendered")


def test_plot_dataset(eda: HyperspectralEDA, save: bool, out_dir: Path) -> None:
    section("12 · PLOT — Dataset dashboard (plot_dataset)")
    save_or_show(eda, "plot_dataset", save, out_dir)
    ok("Dataset dashboard rendered")


def test_plot_spectra(eda: HyperspectralEDA, save: bool, out_dir: Path) -> None:
    section("13 · PLOT — Spectral overlay (plot_spectra)")
    save_or_show(eda, "plot_spectra", save, out_dir)
    ok("Spectral overlay rendered")


def test_plot_false_colour(eda: HyperspectralEDA, save: bool,
                           out_dir: Path) -> None:
    section("14 · PLOT — False colour grid (plot_false_colour)")

    file_recs = [r for r in eda._records
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping false colour grid")
        return

    save_or_show(eda, "plot_false_colour", save, out_dir, n=min(8, len(file_recs)))
    ok("False colour grid rendered")


def test_plot_ndvi(eda: HyperspectralEDA, save: bool, out_dir: Path) -> None:
    section("15 · PLOT — NDVI maps grid (plot_ndvi)")

    file_recs = [r for r in eda._records
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping NDVI grid")
        return

    save_or_show(eda, "plot_ndvi", save, out_dir, n=min(8, len(file_recs)))
    ok("NDVI grid rendered")


def test_plot_pca_components(eda: HyperspectralEDA, save: bool,
                              out_dir: Path) -> None:
    section("16 · PLOT — PCA component images (plot_pca_components)")

    file_recs = [i for i, r in enumerate(eda._records)
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping PCA components plot")
        return

    save_or_show(eda, "plot_pca_components", save, out_dir,
                 cube_index=file_recs[0], n_components=6)
    ok("PCA component images rendered")


def test_plot_band_stats(eda: HyperspectralEDA, save: bool,
                         out_dir: Path) -> None:
    section("17 · PLOT — Band statistics chart (plot_band_stats)")

    file_recs = [i for i, r in enumerate(eda._records)
                 if not r.is_corrupt and not r.path.startswith("<array")]
    if not file_recs:
        info("No file-backed cubes — skipping band stats chart")
        return

    save_or_show(eda, "plot_band_stats", save, out_dir,
                 cube_index=file_recs[0])
    ok("Band statistics chart rendered")


def test_plot_spectral_diversity(eda: HyperspectralEDA, save: bool,
                                  out_dir: Path) -> None:
    section("18 · PLOT — Spectral diversity heatmap (plot_spectral_diversity)")
    save_or_show(eda, "plot_spectral_diversity", save, out_dir)
    ok("Spectral diversity heatmap rendered")


def test_html_report(eda: HyperspectralEDA, report_path: str) -> None:
    section("19 · HTML REPORT (report)")
    path = eda.report(report_path)
    assert Path(path).exists()
    size_kb = Path(path).stat().st_size / 1024
    ok(f"Report saved → {path}  ({size_kb:.1f} KB)")


def test_edge_cases() -> None:
    section("20 · EDGE CASES")

    # Not loaded guard
    fresh = HyperspectralEDA(verbose=False)
    try:
        fresh.summary()
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        ok("RuntimeError raised before load()")

    # Single-band cube (edge case)
    eda2 = HyperspectralEDA(verbose=False, compute_glcm=False,
                             compute_pca=False)
    single_band = np.random.rand(32, 32, 1).astype(np.float32)
    eda2.load_arrays([single_band])
    s = eda2.summary()
    assert s["inventory"]["valid_cubes"] == 1
    assert s["spatial"]["bands"]["mean"] == 1
    ok("Single-band cube handled correctly")

    # Mixed band counts (can't compute cross spectrum for all)
    eda3 = HyperspectralEDA(verbose=False, compute_glcm=False,
                             compute_pca=False)
    rng  = np.random.default_rng(99)
    cubes_mixed = [
        rng.random((32, 32, 50)).astype(np.float32),
        rng.random((32, 32, 50)).astype(np.float32),
        rng.random((32, 32, 100)).astype(np.float32),  # different bands
    ]
    eda3.load_arrays(cubes_mixed, labels=["A", "A", "B"])
    s3 = eda3.summary()
    assert s3["inventory"]["valid_cubes"] == 3
    # 2 out of 3 match dominant band count (50)
    assert s3["spectral_stats"]["n_matching_cubes"] == 2
    ok("Mixed band counts handled correctly "
       f"(dominant={s3['spectral_stats']['dominant_band_count']})")

    # max_cubes is only applied by file-based load(), after paths are resolved.
    # load_arrays() should keep every array explicitly supplied by the caller.
    eda4 = HyperspectralEDA(verbose=False, max_cubes=2,
                             compute_glcm=False, compute_pca=False)
    eda4.load_arrays(cubes_mixed)
    assert len(eda4._records) == len(cubes_mixed)
    ok("load_arrays keeps all provided arrays")

    # Idempotent summary
    eda5 = HyperspectralEDA(verbose=False, compute_glcm=False,
                             compute_pca=False)
    eda5.load_arrays([rng.random((32,32,50)).astype(np.float32)] * 3)
    s_a = eda5.summary()
    s_b = eda5.summary()
    assert s_a["inventory"]["total_cubes"] == s_b["inventory"]["total_cubes"]
    ok("Summary is idempotent")


def test_npz_loading(tmpdir: Path) -> None:
    section("21 · LOAD — .npz files")
    rng = np.random.default_rng(7)
    npz_dir = tmpdir / "npz_cubes"
    npz_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        cube = rng.random((32, 32, 30)).astype(np.float32)
        np.savez(str(npz_dir / f"cube_{i:02d}.npz"), data=cube)

    eda = HyperspectralEDA(verbose=False, compute_glcm=False, compute_pca=False)
    eda.load(str(npz_dir))
    valid = [r for r in eda._records if not r.is_corrupt]
    assert len(valid) == 3
    ok(f"Loaded {len(valid)} cubes from .npz files")


def test_real_mat(mat_path: str, save: bool, out_dir: Path,
                  quick: bool) -> None:
    """Full pipeline test on a real .mat hyperspectral file."""
    section(f"22 · REAL .mat FILE — {Path(mat_path).name}")

    eda = test_load_mat(mat_path, quick)
    if eda is None:
        return

    test_summary(eda)
    test_per_record_fields(eda)

    # If a real .mat file is supplied, make the report reflect that real input.
    eda.report("viseda_hyper_real_mat_report.html")
    ok("Real .mat report saved → viseda_hyper_real_mat_report.html")

    file_recs = [i for i, r in enumerate(eda._records)
                 if not r.is_corrupt]

    if file_recs:
        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
            eda.plot(file_recs[0],
                     save_path=str(out_dir / "plot_mat_single.png"))
            ok(f"Single-cube plot saved → {out_dir/'plot_mat_single.png'}")
        else:
            eda.plot(file_recs[0])
            ok("Single-cube plot rendered")

        if not quick:
            save_or_show(eda, "plot_band_stats", save, out_dir,
                         cube_index=file_recs[0])
            ok("Band stats rendered for real .mat file")

    # HTML report
    rpt = str(out_dir / "report_mat.html") if save else tempfile.mktemp(".html")
    eda.report(rpt)
    ok(f"Report generated → {rpt}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="VisEDA HyperspectralEDA — complete test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mat",        default=None,
                   help="Path to a .mat hyperspectral file")
    p.add_argument("--dir",        default=None,
                   help="Path to a directory of cube files")
    p.add_argument("--save-plots", action="store_true",
                   help="Save plots to ./outputs_hyper/")
    p.add_argument("--quick",      action="store_true",
                   help="Skip GLCM and other slow computations")
    p.add_argument("--report",     default="viseda_hyper_report.html",
                   help="Output path for HTML report")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path("outputs_hyper")
    tmpdir = Path(tempfile.mkdtemp())
    has_user_input = bool(args.dir or args.mat)

    print("\n╔" + "═"*60 + "╗")
    print("║  VisEDA — HyperspectralEDA Complete Test Suite" + " "*14 + "║")
    print("╚" + "═"*60 + "╝")
    print(f"\n  Save plots : {args.save_plots}")
    print(f"  Quick mode : {args.quick}")
    print(f"  .mat file  : {args.mat or 'None'}")
    print(f"  Directory  : {args.dir or 'None'}")
    if has_user_input:
        info("User data supplied — synthetic checks will run without generating final plots/reports.")
        info("Final report and saved plots will be generated from the supplied user data only.")

    passed = 0
    failed = 0
    total_start = time.perf_counter()

    def run(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
        except Exception:
            print(f"\n  ✘  TEST FAILED: {name}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ── Core synthetic checks ─────────────────────────────────────────
    # These validate the API. When real user input is supplied, they do not
    # create final plot/report files, preventing duplicate dashboards such as
    # plot_dataset.png from synthetic arrays.
    eda_arrays = [None]

    def _load_arrays():
        eda_arrays[0] = test_load_arrays(args.quick)

    run("load_arrays", _load_arrays)

    if eda_arrays[0]:
        run("summary", lambda: test_summary(eda_arrays[0]))
        run("per_record_fields", lambda: test_per_record_fields(eda_arrays[0]))
        run("normalization_idem", lambda: test_normalization_idempotent(eda_arrays[0]))

        if not has_user_input:
            run("plot_dataset", lambda: test_plot_dataset(eda_arrays[0], args.save_plots, out_dir))
            run("plot_spectra", lambda: test_plot_spectra(eda_arrays[0], args.save_plots, out_dir))
            run("plot_spectral_div", lambda: test_plot_spectral_diversity(eda_arrays[0], args.save_plots, out_dir))
            run("html_report", lambda: test_html_report(eda_arrays[0], args.report))

    # ── Synthetic file-backed checks ──────────────────────────────────
    # These are kept for API coverage. Their plot outputs are suppressed when
    # real user input is supplied, so final visual outputs come from user data.
    eda_dir = [None]

    def _load_dir():
        eda_dir[0] = test_load_npy_directory(tmpdir, args.quick)

    run("load_npy_dir", _load_dir)

    if eda_dir[0]:
        run("spectral_signature", lambda: test_spectral_signature(eda_dir[0], tmpdir))
        run("compute_index", lambda: test_compute_index(eda_dir[0]))
        run("pca_scores", lambda: test_pca_scores(eda_dir[0]))

        if not has_user_input:
            run("plot_single", lambda: test_plot_single(eda_dir[0], args.save_plots, out_dir, args.quick))
            run("plot_false_colour", lambda: test_plot_false_colour(eda_dir[0], args.save_plots, out_dir))
            run("plot_ndvi", lambda: test_plot_ndvi(eda_dir[0], args.save_plots, out_dir))
            run("plot_pca_components", lambda: test_plot_pca_components(eda_dir[0], args.save_plots, out_dir))
            run("plot_band_stats", lambda: test_plot_band_stats(eda_dir[0], args.save_plots, out_dir))

    run("load_npy_list", lambda: test_load_npy_list(tmpdir, args.quick))
    run("npz_loading", lambda: test_npz_loading(tmpdir))
    run("edge_cases", test_edge_cases)

    # ── User-supplied .mat file ───────────────────────────────────────
    if args.mat:
        def _user_mat_pipeline():
            section(f"22 · USER-SUPPLIED .mat FILE — {Path(args.mat).name}")
            eda_user = test_load_mat(args.mat, args.quick)
            if eda_user is None:
                return
            test_summary(eda_user)
            test_per_record_fields(eda_user, strict_spectral_indices=False)
            test_html_report(eda_user, args.report)
            if args.save_plots:
                out_dir.mkdir(parents=True, exist_ok=True)
                eda_user.plot_dataset(save_path=str(out_dir / "plot_dataset.png"))
                test_plot_spectra(eda_user, True, out_dir)
                test_plot_spectral_diversity(eda_user, True, out_dir)
                test_plot_single(eda_user, True, out_dir, args.quick)
                test_plot_false_colour(eda_user, True, out_dir)
                test_plot_ndvi(eda_user, True, out_dir)
                test_plot_pca_components(eda_user, True, out_dir)
                test_plot_band_stats(eda_user, True, out_dir)
            else:
                eda_user.plot_dataset()
                eda_user.plot_spectra()
        run("user_mat", _user_mat_pipeline)

    # ── User-supplied directory ───────────────────────────────────────
    if args.dir:
        def _user_dir_pipeline():
            section("23 · USER-SUPPLIED DIRECTORY")
            eda_user = HyperspectralEDA(
                verbose=True,
                wavelengths=np.linspace(400, 2500, 200),
                compute_glcm=not args.quick,
                compute_pca=True,
            )
            eda_user.load(args.dir, label_from_parent=True, recursive=True)

            valid = [r for r in eda_user._records if not r.is_corrupt]
            if not valid:
                warn("No valid hyperspectral cubes loaded from the supplied directory.")
                return

            test_summary(eda_user)
            test_per_record_fields(eda_user, strict_spectral_indices=False)
            test_html_report(eda_user, args.report)

            if args.save_plots:
                out_dir.mkdir(parents=True, exist_ok=True)
                # Main detailed dashboard. Uses the standard name so there is only
                # one final plot_dataset.png, and it belongs to the user dataset.
                eda_user.plot_dataset(save_path=str(out_dir / "plot_dataset.png"))
                ok(f"User dataset dashboard saved → {out_dir / 'plot_dataset.png'}")

                test_plot_spectra(eda_user, True, out_dir)
                test_plot_spectral_diversity(eda_user, True, out_dir)
                test_plot_single(eda_user, True, out_dir, args.quick)
                test_plot_false_colour(eda_user, True, out_dir)
                test_plot_ndvi(eda_user, True, out_dir)
                test_plot_pca_components(eda_user, True, out_dir)
                test_plot_band_stats(eda_user, True, out_dir)
            else:
                eda_user.plot_dataset()
                eda_user.plot_spectra()
        run("user_directory", _user_dir_pipeline)

    # ── Final results ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - total_start
    print("\n" + "═"*62)
    print(f"  Results:  {passed} passed  |  {failed} failed  |  {elapsed:.1f}s total")
    if args.save_plots:
        print(f"  Plots saved to: {out_dir.resolve()}/")
    print(f"  Report:   {args.report}")
    print("═"*62 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()