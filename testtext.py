"""
testtext.py
===========
User-dataset-only test script for VisEDA TextEDA.

No synthetic dataset is generated or analysed.

You MUST provide one of:
    --dir   directory of text documents
    --file  single text/structured data file

Examples
--------
Directory organised by class folders:
    python testtext.py \
        --dir "C:/Users/Ike/Desktop/test_dataset/folder_dataset" \
        --save-plots

CSV:
    python testtext.py \
        --file "C:/Users/Ike/Desktop/test_dataset/reviews.csv" \
        --text-field text \
        --label-field label \
        --save-plots

JSONL:
    python testtext.py \
        --file "C:/Users/Ike/Desktop/test_dataset/reviews.jsonl" \
        --text-field text \
        --label-field label \
        --save-plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make project root importable when this file sits in the VisionEDA root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from viseda.text.eda import TextEDA


# ════════════════════════════════════════════════════════════════════════════
# Console helpers
# ════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "═" * 66
    print(f"\n{bar}\n  {title}\n{bar}")


def ok(msg: str) -> None:
    print(f"  ✔  {msg}")


def info(msg: str) -> None:
    print(f"  ℹ  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def save_or_show(
    eda: TextEDA,
    method: str,
    save: bool,
    out_dir: Path,
    **kwargs,
) -> None:
    """Run a TextEDA plotting method and optionally save it."""
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{method}.png")
        kwargs["save_path"] = path
        getattr(eda, method)(**kwargs)
        ok(f"Saved → {path}")
    else:
        getattr(eda, method)(**kwargs)


# ════════════════════════════════════════════════════════════════════════════
# User dataset loading
# ════════════════════════════════════════════════════════════════════════════

def load_user_dataset(
    *,
    file_path: str | None,
    dir_path: str | None,
    text_field: str | None,
    label_field: str | None,
    label_from_parent: bool,
    recursive: bool,
    max_documents: int | None,
) -> TextEDA:
    section("1 · LOAD — User dataset")

    eda = TextEDA(
        verbose=True,
        max_documents=max_documents,
    )

    t0 = time.perf_counter()

    if dir_path:
        source = Path(dir_path)
        if not source.exists():
            raise FileNotFoundError(f"Directory does not exist: {source}")

        eda.load(
            source,
            text_field=text_field,
            label_field=label_field,
            label_from_parent=label_from_parent,
            recursive=recursive,
        )

    elif file_path:
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File does not exist: {source}")

        eda.load(
            source,
            text_field=text_field,
            label_field=label_field,
            label_from_parent=label_from_parent,
            recursive=False,
        )

    else:
        raise ValueError("Provide either --dir or --file.")

    elapsed = time.perf_counter() - t0
    valid = [r for r in eda._records if not r.is_corrupt]

    if not valid:
        raise RuntimeError("No valid documents were loaded from the supplied dataset.")

    ok(
        f"Loaded {len(eda._records)} document(s) "
        f"({len(valid)} valid) in {elapsed:.3f}s"
    )
    return eda


# ════════════════════════════════════════════════════════════════════════════
# Validation and analysis
# ════════════════════════════════════════════════════════════════════════════

def test_summary(eda: TextEDA) -> dict:
    section("2 · SUMMARY")

    s = eda.summary()

    if s.get("error"):
        raise RuntimeError(s["error"])

    inv = s["inventory"]
    assert inv["total_documents"] > 0
    assert inv["valid_documents"] > 0

    ok(
        f"Total: {inv['total_documents']}  "
        f"Valid: {inv['valid_documents']}  "
        f"Corrupt: {inv['corrupt_documents']}"
    )
    ok(f"Format distribution: {inv.get('format_distribution')}")
    ok(f"Label distribution: {inv.get('label_distribution')}")

    length = s.get("length", {})
    lexical = s.get("lexical", {})
    quality = s.get("quality", {})

    # The TextEDA summary schema stores word-count statistics as
    # summary()["length"]["words"].
    word_stats = length.get("words", {})
    if word_stats.get("mean") is not None:
        ok(f"Mean words: {word_stats['mean']:.2f}")

    # Dataset vocabulary statistics live under the "lexical" section.
    vocabulary_size = lexical.get("dataset_unique_tokens", 0)
    ok(f"Vocabulary size: {vocabulary_size}")

    ttr = lexical.get("dataset_type_token_ratio")
    if ttr is not None:
        ok(f"Type-token ratio: {ttr:.4f}")

    hapax = lexical.get("dataset_hapax_count")
    if hapax is not None:
        ok(f"Hapax tokens: {hapax}")

    duplicate_groups = quality.get("exact_duplicate_groups", [])
    ok(f"Exact duplicate groups: {len(duplicate_groups)}")

    # User datasets are NOT required to contain duplicate documents.
    return s


def test_per_record_fields(eda: TextEDA) -> None:
    section("3 · PER-RECORD FIELD VALIDATION")

    valid = [r for r in eda._records if not r.is_corrupt]
    if not valid:
        raise AssertionError("No valid records available.")

    required = [
        "char_count",
        "word_count",
        "sentence_count",
        "paragraph_count",
        "line_count",
        "unique_word_count",
        "lexical_diversity",
        "avg_word_length",
        "avg_sentence_length",
        "stopword_ratio",
        "punctuation_rate",
        "digit_rate",
        "uppercase_rate",
        "non_ascii_rate",
        "url_count",
        "email_count",
        "repeated_line_fraction",
        "dominant_script",
    ]

    checked = valid[: min(10, len(valid))]

    for rec in checked:
        for field in required:
            if not hasattr(rec, field):
                raise AssertionError(
                    f"TextRecord has no field '{field}' for {rec.path}"
                )
            value = getattr(rec, field)
            if value is None:
                raise AssertionError(
                    f"Field '{field}' is None for {rec.path}"
                )

        assert rec.char_count >= 0
        assert rec.word_count >= 0
        assert rec.unique_word_count >= 0

    ok(
        f"All {len(required)} required fields populated on "
        f"{len(checked)} sampled record(s)"
    )


def test_vocabulary_and_distances(eda: TextEDA) -> None:
    section("4 · VOCABULARY / DOCUMENT DISTANCES")

    # TextEDA exposes vocabulary(), which returns an ordered dict-like mapping
    # of token -> count.
    vocab = eda.vocabulary(top_n=20)
    top_vocab = list(vocab.items())
    ok(f"Top vocabulary: {top_vocab[:5]}")

    valid_count = sum(not r.is_corrupt for r in eda._records)

    if valid_count >= 2:
        dist, names = eda.pairwise_document_distances()
        assert dist.shape[0] == dist.shape[1] == len(names)
        assert np.allclose(np.diag(dist), 0.0)
        ok(f"Distance matrix: {dist.shape}")

        try:
            pairs = eda.near_duplicate_pairs()
            ok(f"Near-duplicate pairs: {len(pairs)}")
        except Exception as exc:
            info(f"Near-duplicate pair analysis skipped: {exc}")
    else:
        info("Only one valid document — pairwise distance analysis skipped.")


def test_plots(
    eda: TextEDA,
    save: bool,
    out_dir: Path,
) -> None:
    section("5 · PLOTS")

    save_or_show(eda, "plot_dataset", save, out_dir)
    ok("Dataset dashboard rendered")

    valid = [r for r in eda._records if not r.is_corrupt]

    if valid:
        valid_index = eda._records.index(valid[0])
        save_or_show(
            eda,
            "plot",
            save,
            out_dir,
            document_index=valid_index,
        )
        ok("Single-document dashboard rendered")

    save_or_show(
        eda,
        "plot_word_frequency",
        save,
        out_dir,
    )
    ok("Word-frequency plot rendered")

    save_or_show(
        eda,
        "plot_ngram_frequency",
        save,
        out_dir,
    )
    ok("N-gram plot rendered")

    save_or_show(
        eda,
        "plot_length_distribution",
        save,
        out_dir,
    )
    ok("Length distributions rendered")

    labels = [
        r.label for r in valid
        if r.label is not None
    ]
    if len(set(labels)) >= 2:
        save_or_show(
            eda,
            "plot_label_comparison",
            save,
            out_dir,
            metric="word_count",
        )
        ok("Label comparison rendered")
    else:
        info(
            "Label comparison skipped — at least two labels/classes "
            "are required."
        )

    if len(valid) >= 2:
        save_or_show(
            eda,
            "plot_pairwise_document_distances",
            save,
            out_dir,
        )
        ok("Pairwise document distances rendered")
    else:
        info(
            "Pairwise distance plot skipped — at least two documents "
            "are required."
        )

    save_or_show(
        eda,
        "plot_text_samples",
        save,
        out_dir,
        n=min(12, len(valid)),
        cols=3,
    )
    ok("Text sample grid rendered")


def test_report(
    eda: TextEDA,
    report_path: str,
) -> None:
    section("6 · HTML REPORT")

    path = eda.report(report_path)
    p = Path(path)

    if not p.exists():
        raise AssertionError(
            f"Report was not created: {path}"
        )

    s = eda.summary()
    inv = s["inventory"]

    ok(
        f"Report saved → {path} "
        f"({inv['total_documents']} documents, "
        f"{inv['valid_documents']} valid)"
    )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VisEDA TextEDA user-dataset test. "
            "No synthetic dataset is generated."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--file",
        type=str,
        default=None,
        help=(
            "Single text, CSV, TSV, JSON, JSONL, HTML, "
            "Markdown, or other supported text file."
        ),
    )

    source.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing text documents.",
    )

    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help=(
            "Text/content column for structured files such as "
            "CSV, JSON, or JSONL."
        ),
    )

    parser.add_argument(
        "--label-field",
        type=str,
        default=None,
        help="Label/class column for structured files.",
    )

    parser.add_argument(
        "--label-from-parent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use parent folder name as label for directory datasets. "
            "Default: enabled."
        ),
    )

    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search directories. Default: enabled.",
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional maximum number of documents to analyse.",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots instead of displaying them.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_text",
        help="Directory for generated plots.",
    )

    parser.add_argument(
        "--report",
        type=str,
        default="viseda_text_report.html",
        help="HTML report output path.",
    )

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  VisEDA — TextEDA User Dataset Test                           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"\n  Save plots       : {args.save_plots}")
    print(f"  File             : {args.file}")
    print(f"  Directory        : {args.dir}")
    print(f"  Text field       : {args.text_field}")
    print(f"  Label field      : {args.label_field}")
    print(f"  Label from parent: {args.label_from_parent}")
    print(f"  Recursive        : {args.recursive}")
    print(f"  Max documents    : {args.max_documents}")

    out_dir = Path(args.output_dir)

    passed = 0
    failed = 0
    t0 = time.perf_counter()

    def run(name: str, fn) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"\n  ✘  TEST FAILED: {name}")
            import traceback
            traceback.print_exc()

    holder: dict[str, TextEDA] = {}

    def load_step() -> None:
        holder["eda"] = load_user_dataset(
            file_path=args.file,
            dir_path=args.dir,
            text_field=args.text_field,
            label_field=args.label_field,
            label_from_parent=args.label_from_parent,
            recursive=args.recursive,
            max_documents=args.max_documents,
        )

    run("load_user_dataset", load_step)

    if "eda" in holder:
        eda = holder["eda"]

        run(
            "summary",
            lambda: test_summary(eda),
        )

        run(
            "per_record_fields",
            lambda: test_per_record_fields(eda),
        )

        run(
            "vocabulary_distances",
            lambda: test_vocabulary_and_distances(eda),
        )

        run(
            "plots",
            lambda: test_plots(
                eda,
                args.save_plots,
                out_dir,
            ),
        )

        run(
            "report",
            lambda: test_report(
                eda,
                args.report,
            ),
        )

    elapsed = time.perf_counter() - t0

    print("\n" + "═" * 66)
    print(
        f"  Results: {passed} passed | "
        f"{failed} failed | {elapsed:.2f}s total"
    )
    print(f"  Plots:   {out_dir.resolve()}")
    print(f"  Report:  {Path(args.report).resolve()}")
    print("═" * 66)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
