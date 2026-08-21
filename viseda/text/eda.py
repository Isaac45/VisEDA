"""
viseda.text.eda
===============
Comprehensive exploratory data analysis for text and NLP datasets.

The module provides one unified class, :class:`TextEDA`, for analysing a
single text document, a list of documents, structured text files, or a full
directory of text data.

Supported inputs
----------------
* Plain text: ``.txt``, ``.text``, ``.md``, ``.rst``, ``.log``
* Web text: ``.html``, ``.htm`` (tags, script, and style content removed)
* Delimited data: ``.csv``, ``.tsv``
* Structured data: ``.json``, ``.jsonl``, ``.ndjson``
* In-memory strings through :meth:`TextEDA.load_texts`

Core analyses
-------------
* Inventory, encoding, labels, formats, corrupt files
* Character, word, sentence, paragraph, and line statistics
* Vocabulary size, lexical diversity, hapax ratio, token-length statistics
* Stopword, punctuation, digit, URL, email, hashtag, mention, emoji, and
  non-ASCII statistics
* Readability estimates (Flesch Reading Ease and Flesch-Kincaid grade)
* Dominant writing-script distribution
* Exact duplicate detection and TF-IDF document-distance analysis
* Word and n-gram frequencies
* Dataset and single-document dashboards
* Styled, self-contained HTML reports

The implementation intentionally uses a lightweight regex tokenizer by
default. It does not silently depend on spaCy, NLTK, or transformer models.
Optional scikit-learn support is used for TF-IDF distances when available;
a NumPy fallback is provided.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import statistics
import warnings
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Lazy plotting imports
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib.pyplot as plt
    return plt


def _mpl():
    import matplotlib as mpl
    return mpl


# ---------------------------------------------------------------------------
# Constants and regexes
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(
    r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*",
    flags=re.UNICODE,
)
WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+|\n+")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r"(?<!\w)#[\w_]+", flags=re.UNICODE)
MENTION_RE = re.compile(r"(?<!\w)@[\w_]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]",
    flags=re.UNICODE,
)

DEFAULT_TEXT_FIELDS = (
    "text", "content", "document", "sentence", "review", "comment",
    "body", "description", "message", "abstract", "article", "question",
)
DEFAULT_LABEL_FIELDS = (
    "label", "class", "category", "target", "sentiment", "topic", "intent",
)

# Compact, built-in English stopword list. The user can replace it.
DEFAULT_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
})

SCRIPT_RANGES = {
    "Latin": ((0x0041, 0x024F), (0x1E00, 0x1EFF)),
    "Cyrillic": ((0x0400, 0x052F),),
    "Greek": ((0x0370, 0x03FF),),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
    "Hebrew": ((0x0590, 0x05FF),),
    "Devanagari": ((0x0900, 0x097F),),
    "Bengali": ((0x0980, 0x09FF),),
    "Thai": ((0x0E00, 0x0E7F),),
    "CJK": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
    "Hiragana": ((0x3040, 0x309F),),
    "Katakana": ((0x30A0, 0x30FF),),
    "Hangul": ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stat_dict(values: Union[Sequence[float], np.ndarray]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "p25": None, "median": None, "p75": None, "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _normalise_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _normalised_hash(text: str) -> str:
    normal = _normalise_whitespace(text).casefold()
    return hashlib.sha256(normal.encode("utf-8", errors="ignore")).hexdigest()


def _tokenise(text: str, lowercase: bool = True, min_length: int = 1) -> List[str]:
    toks = TOKEN_RE.findall(text)
    if lowercase:
        toks = [t.casefold() for t in toks]
    if min_length > 1:
        toks = [t for t in toks if len(t) >= min_length or t[0].isdigit()]
    return toks


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def _count_syllables(word: str) -> int:
    """A lightweight English syllable heuristic for readability estimates."""
    w = re.sub(r"[^a-z]", "", word.casefold())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ye")) and count > 1:
        count -= 1
    if w.endswith("es") and len(w) > 4 and count > 1:
        count -= 1
    return max(1, count)


def _readability(words: Sequence[str], sentence_count: int) -> Tuple[Optional[float], Optional[float]]:
    alpha = [w for w in words if WORD_RE.fullmatch(w)]
    if not alpha or sentence_count <= 0:
        return None, None
    syllables = sum(_count_syllables(w) for w in alpha)
    n_words = len(alpha)
    flesch = 206.835 - 1.015 * (n_words / sentence_count) - 84.6 * (syllables / n_words)
    grade = 0.39 * (n_words / sentence_count) + 11.8 * (syllables / n_words) - 15.59
    return float(flesch), float(grade)


def _script_distribution(text: str) -> Tuple[Dict[str, float], str]:
    counts: Counter[str] = Counter()
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        total += 1
        matched = False
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                matched = True
                break
        if not matched:
            counts["Other"] += 1
    if total == 0:
        return {}, "None"
    dist = {k: float(v / total) for k, v in counts.items()}
    dominant = max(counts, key=counts.get)
    return dist, dominant


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape(" ".join(self.parts))
        lines = [WHITESPACE_RE.sub(" ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Per-document record
# ---------------------------------------------------------------------------
class TextRecord:
    """Container for per-document statistics."""

    __slots__ = (
        # identity/status
        "path", "name", "label", "file_ext", "source_index", "file_size_kb",
        "encoding", "is_corrupt", "error", "normalised_hash",
        # counts
        "char_count", "byte_count", "word_count", "alpha_word_count",
        "numeric_token_count", "unique_word_count", "sentence_count",
        "paragraph_count", "line_count", "nonempty_line_count",
        # length/distribution
        "avg_word_length", "median_word_length", "word_length_std",
        "avg_sentence_length", "median_sentence_length", "sentence_length_std",
        "avg_paragraph_words", "max_sentence_words",
        # lexical
        "lexical_diversity", "hapax_count", "hapax_ratio", "stopword_count",
        "stopword_ratio", "top_tokens", "top_bigrams", "top_trigrams",
        # character/symbol
        "punctuation_count", "punctuation_rate", "digit_count", "digit_rate",
        "uppercase_count", "uppercase_rate", "whitespace_count", "whitespace_rate",
        "non_ascii_count", "non_ascii_rate", "emoji_count", "url_count",
        "email_count", "hashtag_count", "mention_count",
        # quality/style
        "repeated_line_fraction", "empty", "very_short", "very_long",
        "readability_flesch", "readability_grade", "script_distribution",
        "dominant_script", "preview",
    )

    def __init__(self) -> None:
        for field in self.__slots__:
            setattr(self, field, None)
        self.is_corrupt = False
        self.error = None
        self.empty = False
        self.very_short = False
        self.very_long = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class TextEDA:
    """
    Comprehensive EDA for text and NLP datasets.

    Parameters
    ----------
    verbose:
        Print progress information.
    max_documents:
        Analyse at most this many documents after loading/expansion.
    lowercase:
        Lowercase/casefold tokens for vocabulary and frequency analyses.
    min_token_length:
        Minimum token length used for vocabulary statistics.
    stopwords:
        Optional iterable replacing the built-in English stopword list.
    short_document_words:
        Documents below this word count are flagged as very short.
    long_document_words:
        Documents above this word count are flagged as very long.
    top_n:
        Number of top unigrams/bigrams/trigrams retained in each record.
    encoding:
        Preferred file encoding. Automatic fallbacks are tried after it.
    random_seed:
        Seed used for reproducible sampling in plots and distance fallback.
    """

    SUPPORTED_EXTS = {
        ".txt", ".text", ".md", ".rst", ".log", ".html", ".htm",
        ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
    }

    def __init__(
        self,
        verbose: bool = True,
        max_documents: Optional[int] = None,
        lowercase: bool = True,
        min_token_length: int = 1,
        stopwords: Optional[Iterable[str]] = None,
        short_document_words: int = 5,
        long_document_words: int = 1_000,
        top_n: int = 30,
        encoding: Optional[str] = None,
        random_seed: int = 0,
    ) -> None:
        self.verbose = verbose
        self.max_documents = max_documents
        self.lowercase = lowercase
        self.min_token_length = max(1, int(min_token_length))
        self.stopwords = frozenset(
            s.casefold() if lowercase else s for s in (stopwords or DEFAULT_STOPWORDS)
        )
        self.short_document_words = int(short_document_words)
        self.long_document_words = int(long_document_words)
        self.top_n = int(top_n)
        self.encoding = encoding
        self.random_seed = int(random_seed)

        self._records: List[TextRecord] = []
        self._texts: Dict[str, str] = {}
        self._loaded = False
        self._results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        labels: Optional[Mapping[str, str]] = None,
        label_from_parent: bool = False,
        recursive: bool = True,
        text_field: Optional[str] = None,
        label_field: Optional[str] = None,
    ) -> "TextEDA":
        """Load one file, a list of files, or a directory of text data."""
        paths = self._resolve_paths(source, recursive=recursive)
        label_map = {str(Path(k).resolve()): str(v) for k, v in (labels or {}).items()}
        self._records = []
        self._texts = {}
        self._results = {}

        self._log(f"Found {len(paths)} text file(s) — extracting documents …")
        document_counter = 0

        for file_idx, path in enumerate(paths):
            if self.max_documents is not None and document_counter >= self.max_documents:
                break
            if self.verbose:
                self._log(f"  [{file_idx + 1}/{len(paths)}] {path.name}")

            base_label = label_map.get(str(path.resolve()))
            if base_label is None and label_from_parent:
                base_label = path.parent.name

            try:
                docs, detected_encoding = self._read_documents(
                    path, text_field=text_field, label_field=label_field
                )
            except Exception as exc:
                rec = TextRecord()
                rec.path = str(path)
                rec.name = path.name
                rec.label = base_label
                rec.file_ext = path.suffix.lower()
                rec.file_size_kb = path.stat().st_size / 1024 if path.exists() else None
                rec.is_corrupt = True
                rec.error = str(exc)
                self._records.append(rec)
                self._log(f"    ✗ {path.name}: {exc}")
                continue

            for source_index, text, structured_label in docs:
                if self.max_documents is not None and document_counter >= self.max_documents:
                    break
                rec = TextRecord()
                rec.path = str(path)
                rec.name = path.name if source_index is None else f"{path.name}#{source_index}"
                rec.label = structured_label if structured_label is not None else base_label
                rec.file_ext = path.suffix.lower()
                rec.source_index = source_index
                rec.file_size_kb = path.stat().st_size / 1024 if path.exists() else None
                rec.encoding = detected_encoding
                key = self._record_key(rec, len(self._records))
                try:
                    self._fill_stats(rec, str(text))
                    self._texts[key] = str(text)
                except Exception as exc:
                    rec.is_corrupt = True
                    rec.error = str(exc)
                self._records.append(rec)
                document_counter += 1

        self._loaded = True
        bad = sum(r.is_corrupt for r in self._records)
        self._log(f"Done. {len(self._records)} document(s) loaded ({bad} corrupt).")
        return self

    def load_texts(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[str]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> "TextEDA":
        """Load in-memory text strings directly."""
        self._records = []
        self._texts = {}
        self._results = {}
        n = len(texts) if self.max_documents is None else min(len(texts), self.max_documents)
        self._log(f"Loading {n} in-memory text document(s) …")
        for i, text in enumerate(texts[:n]):
            rec = TextRecord()
            rec.path = names[i] if names and i < len(names) else f"<text_{i}>"
            rec.name = rec.path
            rec.file_ext = "text"
            rec.source_index = i
            rec.label = labels[i] if labels and i < len(labels) else None
            rec.encoding = "unicode"
            key = self._record_key(rec, i)
            try:
                self._fill_stats(rec, str(text))
                self._texts[key] = str(text)
            except Exception as exc:
                rec.is_corrupt = True
                rec.error = str(exc)
            self._records.append(rec)
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Public access and summaries
    # ------------------------------------------------------------------
    def get_record(self, index: int = 0) -> TextRecord:
        self._check_loaded()
        return self._records[index]

    def get_text(self, index: int = 0) -> str:
        self._check_loaded()
        rec = self._records[index]
        key = self._record_key(rec, index)
        if key not in self._texts:
            raise KeyError(f"Text content is unavailable for record {index}.")
        return self._texts[key]

    def vocabulary(
        self,
        min_frequency: int = 1,
        top_n: Optional[int] = None,
        exclude_stopwords: bool = False,
    ) -> Dict[str, int]:
        """Return the dataset vocabulary ordered by descending frequency."""
        self._check_loaded()
        counts = self._dataset_token_counts(exclude_stopwords=exclude_stopwords)
        items = [(t, c) for t, c in counts.items() if c >= min_frequency]
        items.sort(key=lambda x: (-x[1], x[0]))
        if top_n is not None:
            items = items[:top_n]
        return dict(items)

    def summary(self) -> Dict[str, Any]:
        """Return a nested summary dictionary for all loaded documents."""
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]
        if not valid:
            result = {
                "inventory": {
                    "total_documents": len(self._records),
                    "valid_documents": 0,
                    "corrupt_documents": len(corrupt),
                    "corrupt_paths": [r.path for r in corrupt],
                    "format_distribution": {},
                    "label_distribution": None,
                },
                "error": "No valid text documents found.",
            }
            self._results["summary"] = result
            return result

        def arr(attr: str) -> np.ndarray:
            vals = [getattr(r, attr) for r in valid if getattr(r, attr) is not None]
            return np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)

        label_values = [r.label for r in valid if r.label is not None]
        label_dist = dict(Counter(label_values)) if label_values else None
        format_dist = dict(Counter(r.file_ext for r in valid))
        encoding_dist = dict(Counter(r.encoding for r in valid if r.encoding))
        script_dist = dict(Counter(r.dominant_script for r in valid if r.dominant_script))

        token_counts = self._dataset_token_counts(exclude_stopwords=False)
        content_counts = self._dataset_token_counts(exclude_stopwords=True)
        total_tokens = int(sum(token_counts.values()))
        unique_tokens = len(token_counts)
        hapax = sum(1 for c in token_counts.values() if c == 1)
        top_words = token_counts.most_common(50)
        top_content_words = content_counts.most_common(50)
        bigram_counts = self._dataset_ngram_counts(2, exclude_stopwords=False)
        trigram_counts = self._dataset_ngram_counts(3, exclude_stopwords=False)

        hash_groups: Dict[str, List[str]] = defaultdict(list)
        for r in valid:
            hash_groups[r.normalised_hash].append(r.name or r.path)
        duplicate_groups = [names for names in hash_groups.values() if len(names) > 1]
        duplicate_docs = sum(len(g) for g in duplicate_groups)

        result = {
            "inventory": {
                "total_documents": len(self._records),
                "valid_documents": len(valid),
                "corrupt_documents": len(corrupt),
                "corrupt_paths": [r.path for r in corrupt],
                "format_distribution": format_dist,
                "encoding_distribution": encoding_dist,
                "label_distribution": label_dist,
                "dominant_script_distribution": script_dist,
            },
            "length": {
                "characters": _stat_dict(arr("char_count")),
                "bytes": _stat_dict(arr("byte_count")),
                "words": _stat_dict(arr("word_count")),
                "sentences": _stat_dict(arr("sentence_count")),
                "paragraphs": _stat_dict(arr("paragraph_count")),
                "lines": _stat_dict(arr("line_count")),
                "avg_word_length": _stat_dict(arr("avg_word_length")),
                "avg_sentence_length": _stat_dict(arr("avg_sentence_length")),
                "sentence_length_std": _stat_dict(arr("sentence_length_std")),
            },
            "lexical": {
                "dataset_total_tokens": total_tokens,
                "dataset_unique_tokens": unique_tokens,
                "dataset_type_token_ratio": _safe_ratio(unique_tokens, total_tokens),
                "dataset_hapax_count": hapax,
                "dataset_hapax_ratio": _safe_ratio(hapax, unique_tokens),
                "document_lexical_diversity": _stat_dict(arr("lexical_diversity")),
                "document_hapax_ratio": _stat_dict(arr("hapax_ratio")),
                "stopword_ratio": _stat_dict(arr("stopword_ratio")),
                "top_words": top_words,
                "top_content_words": top_content_words,
                "top_bigrams": [(" ".join(k), v) for k, v in bigram_counts.most_common(50)],
                "top_trigrams": [(" ".join(k), v) for k, v in trigram_counts.most_common(50)],
            },
            "symbols": {
                "punctuation_rate": _stat_dict(arr("punctuation_rate")),
                "digit_rate": _stat_dict(arr("digit_rate")),
                "uppercase_rate": _stat_dict(arr("uppercase_rate")),
                "whitespace_rate": _stat_dict(arr("whitespace_rate")),
                "non_ascii_rate": _stat_dict(arr("non_ascii_rate")),
                "emoji_count": _stat_dict(arr("emoji_count")),
                "url_count": _stat_dict(arr("url_count")),
                "email_count": _stat_dict(arr("email_count")),
                "hashtag_count": _stat_dict(arr("hashtag_count")),
                "mention_count": _stat_dict(arr("mention_count")),
            },
            "readability": {
                "flesch_reading_ease": _stat_dict(arr("readability_flesch")),
                "flesch_kincaid_grade": _stat_dict(arr("readability_grade")),
            },
            "quality": {
                "empty_documents": int(sum(bool(r.empty) for r in valid)),
                "very_short_documents": int(sum(bool(r.very_short) for r in valid)),
                "very_long_documents": int(sum(bool(r.very_long) for r in valid)),
                "repeated_line_fraction": _stat_dict(arr("repeated_line_fraction")),
                "exact_duplicate_groups": duplicate_groups,
                "documents_in_duplicate_groups": duplicate_docs,
                "exact_duplicate_fraction": _safe_ratio(duplicate_docs, len(valid)),
            },
            "labels": {
                "label_distribution": label_dist,
                "class_imbalance_ratio": self._imbalance_ratio(label_dist),
            },
        }
        self._results["summary"] = result
        return result

    def pairwise_document_distances(
        self,
        max_documents: int = 50,
        max_features: int = 5_000,
        exclude_stopwords: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """Return a TF-IDF cosine-distance matrix and document names."""
        self._check_loaded()
        pairs = [
            (i, r) for i, r in enumerate(self._records) if not r.is_corrupt
        ][:max_documents]
        if len(pairs) < 2:
            raise ValueError("Need at least two valid documents.")
        texts = [self.get_text(i) for i, _ in pairs]
        names = [r.label or r.name or f"document_{i}" for i, r in pairs]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_distances

            vectorizer = TfidfVectorizer(
                lowercase=self.lowercase,
                token_pattern=r"(?u)\b\w+\b",
                stop_words=list(self.stopwords) if exclude_stopwords else None,
                max_features=max_features,
                min_df=1,
            )
            X = vectorizer.fit_transform(texts)
            if X.shape[1] == 0:
                return np.zeros((len(texts), len(texts))), names
            dist = cosine_distances(X)
            return np.asarray(dist, dtype=float), names
        except Exception:
            return self._pairwise_fallback(texts, names, max_features, exclude_stopwords)

    def near_duplicate_pairs(
        self,
        threshold: float = 0.15,
        max_documents: int = 100,
    ) -> List[Tuple[str, str, float]]:
        """Return pairs whose TF-IDF cosine distance is at most *threshold*."""
        dist, names = self.pairwise_document_distances(max_documents=max_documents)
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if dist[i, j] <= threshold:
                    pairs.append((names[i], names[j], float(dist[i, j])))
        return pairs

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_dataset(
        self,
        figsize: Tuple[int, int] = (22, 28),
        save_path: Optional[str] = None,
        dpi: int = 160,
        top_n: int = 20,
    ) -> None:
        """Generate a comprehensive dataset-level TextEDA dashboard."""
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        if not valid:
            raise RuntimeError("No valid documents to plot.")

        plt = _plt()
        mpl = _mpl()
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle("TextEDA — Dataset Analysis", fontsize=20, fontweight="bold", y=0.995)
        gs = mpl.gridspec.GridSpec(7, 4, figure=fig, hspace=0.65, wspace=0.42)

        self._plot_dataset_card(fig.add_subplot(gs[0, :2]), valid)
        self._plot_label_distribution(fig.add_subplot(gs[0, 2:]), valid)

        self._plot_hist(fig.add_subplot(gs[1, 0]), [r.word_count for r in valid], "Word Count", "Words")
        self._plot_hist(fig.add_subplot(gs[1, 1]), [r.char_count for r in valid], "Character Count", "Characters")
        self._plot_hist(fig.add_subplot(gs[1, 2]), [r.sentence_count for r in valid], "Sentence Count", "Sentences")
        self._plot_hist(fig.add_subplot(gs[1, 3]), [r.lexical_diversity for r in valid], "Lexical Diversity", "Unique / words")

        self._plot_hist(fig.add_subplot(gs[2, 0]), [r.avg_sentence_length for r in valid], "Average Sentence Length", "Words / sentence")
        self._plot_hist(fig.add_subplot(gs[2, 1]), [r.avg_word_length for r in valid], "Average Word Length", "Characters / word")
        self._plot_hist(fig.add_subplot(gs[2, 2]), [r.stopword_ratio for r in valid], "Stopword Ratio", "Fraction")
        self._plot_hist(fig.add_subplot(gs[2, 3]), [r.readability_flesch for r in valid if r.readability_flesch is not None], "Flesch Reading Ease", "Score")

        self._plot_hist(fig.add_subplot(gs[3, 0]), [r.punctuation_rate for r in valid], "Punctuation Rate", "Fraction of characters")
        self._plot_hist(fig.add_subplot(gs[3, 1]), [r.digit_rate for r in valid], "Digit Rate", "Fraction of characters")
        self._plot_hist(fig.add_subplot(gs[3, 2]), [r.non_ascii_rate for r in valid], "Non-ASCII Rate", "Fraction of characters")
        self._plot_hist(fig.add_subplot(gs[3, 3]), [r.repeated_line_fraction for r in valid], "Repeated-Line Fraction", "Fraction")

        self._plot_frequency(fig.add_subplot(gs[4, :2]), self.vocabulary(top_n=top_n), "Top Words")
        bigrams = {" ".join(k): v for k, v in self._dataset_ngram_counts(2).most_common(top_n)}
        self._plot_frequency(fig.add_subplot(gs[4, 2:]), bigrams, "Top Bigrams")

        self._plot_script_distribution(fig.add_subplot(gs[5, 0]), valid)
        self._plot_format_distribution(fig.add_subplot(gs[5, 1]), valid)
        self._plot_scatter(fig.add_subplot(gs[5, 2]), valid, "word_count", "unique_word_count", "Words vs Unique Words")
        self._plot_scatter(fig.add_subplot(gs[5, 3]), valid, "avg_sentence_length", "readability_flesch", "Sentence Length vs Readability")

        self._plot_pairwise_panel(fig.add_subplot(gs[6, :2]), max_documents=30)
        self._plot_quality_flags(fig.add_subplot(gs[6, 2:]), valid)

        self._finalise(fig, save_path, dpi)

    def plot(
        self,
        document_index: int = 0,
        figsize: Tuple[int, int] = (18, 15),
        save_path: Optional[str] = None,
        dpi: int = 160,
        top_n: int = 20,
    ) -> None:
        """Generate a single-document deep-dive dashboard."""
        self._check_loaded()
        rec = self._records[document_index]
        if rec.is_corrupt:
            raise RuntimeError(f"Document is corrupt: {rec.error}")
        text = self.get_text(document_index)
        tokens = _tokenise(text, self.lowercase, self.min_token_length)
        sentences = _sentences(text)

        plt = _plt()
        mpl = _mpl()
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle(f"TextEDA — {rec.label or rec.name}", fontsize=18, fontweight="bold", y=0.99)
        gs = mpl.gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

        self._plot_record_card(fig.add_subplot(gs[0, 0]), rec)
        self._plot_text_preview(fig.add_subplot(gs[0, 1:]), text)

        counts = Counter(tokens)
        self._plot_frequency(fig.add_subplot(gs[1, 0]), dict(counts.most_common(top_n)), "Top Tokens")
        self._plot_hist(fig.add_subplot(gs[1, 1]), [len(t) for t in tokens], "Token Length Distribution", "Characters")
        self._plot_hist(fig.add_subplot(gs[1, 2]), [len(_tokenise(s, self.lowercase, self.min_token_length)) for s in sentences], "Sentence Length Distribution", "Words")

        bigrams = {" ".join(k): v for k, v in Counter(_ngrams(tokens, 2)).most_common(top_n)}
        self._plot_frequency(fig.add_subplot(gs[2, 0]), bigrams, "Top Bigrams")
        self._plot_character_categories(fig.add_subplot(gs[2, 1]), rec)
        self._plot_token_coverage(fig.add_subplot(gs[2, 2]), counts)

        self._plot_script_record(fig.add_subplot(gs[3, 0]), rec)
        self._plot_quality_record(fig.add_subplot(gs[3, 1]), rec)
        self._plot_sentence_sequence(fig.add_subplot(gs[3, 2]), sentences)

        self._finalise(fig, save_path, dpi)

    def plot_word_frequency(
        self,
        top_n: int = 30,
        exclude_stopwords: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        counts = self.vocabulary(top_n=top_n, exclude_stopwords=exclude_stopwords)
        plt = _plt()
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        self._plot_frequency(ax, counts, "Dataset Word Frequency")
        self._finalise(fig, save_path, dpi)

    def plot_ngram_frequency(
        self,
        n: int = 2,
        top_n: int = 30,
        exclude_stopwords: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        counts = self._dataset_ngram_counts(n, exclude_stopwords=exclude_stopwords)
        data = {" ".join(k): v for k, v in counts.most_common(top_n)}
        plt = _plt()
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        self._plot_frequency(ax, data, f"Top {n}-gram Frequency")
        self._finalise(fig, save_path, dpi)

    def plot_length_distribution(
        self,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        plt = _plt()
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor="white")
        fig.suptitle("TextEDA — Length Distributions", fontsize=16, fontweight="bold")
        self._plot_hist(axes[0, 0], [r.word_count for r in valid], "Word Count", "Words")
        self._plot_hist(axes[0, 1], [r.char_count for r in valid], "Character Count", "Characters")
        self._plot_hist(axes[1, 0], [r.sentence_count for r in valid], "Sentence Count", "Sentences")
        self._plot_hist(axes[1, 1], [r.avg_sentence_length for r in valid], "Average Sentence Length", "Words")
        self._finalise(fig, save_path, dpi)

    def plot_label_comparison(
        self,
        metric: str = "word_count",
        figsize: Tuple[int, int] = (12, 7),
        save_path: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt and r.label is not None]
        if not valid:
            raise ValueError("No labelled documents are available.")
        if not hasattr(valid[0], metric):
            raise ValueError(f"Unknown TextRecord metric: {metric}")
        groups: Dict[str, List[float]] = defaultdict(list)
        for rec in valid:
            value = getattr(rec, metric)
            if value is not None and np.isfinite(value):
                groups[str(rec.label)].append(float(value))
        labels = sorted(groups)
        data = [groups[label] for label in labels]
        plt = _plt()
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        # Matplotlib compatibility:
        # older releases use ``labels``; newer releases renamed it to
        # ``tick_labels``. Try the new name first, then fall back.
        try:
            ax.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:
            ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Label")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
        self._style_axis(ax)
        self._finalise(fig, save_path, dpi)

    def plot_pairwise_document_distances(
        self,
        max_documents: int = 50,
        figsize: Tuple[int, int] = (11, 9),
        save_path: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        dist, names = self.pairwise_document_distances(max_documents=max_documents)
        plt = _plt()
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        im = ax.imshow(dist, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine distance")
        if len(names) <= 35:
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Pairwise Document Distances")
        ax.set_xlabel("Document")
        ax.set_ylabel("Document")
        self._finalise(fig, save_path, dpi)

    def plot_text_samples(
        self,
        n: int = 12,
        cols: int = 3,
        chars: int = 450,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        self._check_loaded()
        valid_indices = [i for i, r in enumerate(self._records) if not r.is_corrupt]
        if not valid_indices:
            raise RuntimeError("No valid documents to plot.")
        n = min(n, len(valid_indices))
        rows = int(math.ceil(n / cols))
        if figsize is None:
            figsize = (6 * cols, 3.5 * rows)
        plt = _plt()
        fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="white", squeeze=False)
        fig.suptitle("TextEDA — Text Sample Grid", fontsize=17, fontweight="bold")
        for ax, idx in zip(axes.flat, valid_indices[:n]):
            rec = self._records[idx]
            preview = _normalise_whitespace(self.get_text(idx))[:chars]
            ax.text(0.02, 0.95, preview, va="top", ha="left", wrap=True, fontsize=9, transform=ax.transAxes)
            ax.set_title(rec.label or rec.name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            self._style_axis(ax)
        for ax in axes.flat[n:]:
            ax.axis("off")
        self._finalise(fig, save_path, dpi)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    def report(self, output_path: str = "viseda_text_report.html") -> str:
        """Write a styled self-contained HTML report and return its path."""
        self._check_loaded()
        summary = self.summary()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        inv = summary.get("inventory", {})
        label_dist = inv.get("label_distribution") or {}
        format_dist = inv.get("format_distribution") or {}
        script_dist = inv.get("dominant_script_distribution") or {}

        sections = [
            self._html_section("Length", summary.get("length", {})),
            self._html_section("Symbols and Markup", summary.get("symbols", {})),
            self._html_section("Readability", summary.get("readability", {})),
            self._html_section("Quality", summary.get("quality", {}), skip_complex=True),
        ]

        lexical = summary.get("lexical", {})
        top_words = lexical.get("top_words", [])[:20]
        top_bigrams = lexical.get("top_bigrams", [])[:20]

        html_text = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VisEDA — Text Report</title>
<style>
:root{{--bg:white;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;
--muted:#57606a;--accent:#58a6ff;--green:#3fb950;--red:#f78166;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:2rem}}
h1{{font-size:1.9rem;margin-bottom:.25rem}} h2{{font-size:1.05rem;color:var(--accent);margin:1.8rem 0 .6rem}}
h3{{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}}
.sub{{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:.9rem}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem}}
.stat{{display:flex;justify-content:space-between;gap:.6rem;font-size:.82rem;padding:.18rem 0;border-bottom:1px solid var(--border)}}
.stat:last-child{{border-bottom:none}} .val{{color:var(--accent);font-variant-numeric:tabular-nums;text-align:right}}
.badge{{display:inline-block;padding:.15rem .5rem;border-radius:12px;font-size:.72rem;font-weight:600;margin:.15rem}}
.blue{{background:rgba(88,166,255,.15);color:var(--accent)}} .green{{background:rgba(63,185,80,.15);color:var(--green)}}
.bar-row{{display:flex;align-items:center;gap:.4rem;margin:.2rem 0;font-size:.76rem}}
.bar-label{{width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}}
.bar{{flex:1;background:var(--border);border-radius:3px;height:9px}} .bar-fill{{height:100%;border-radius:3px;background:var(--accent)}}
.bar-count{{width:55px;text-align:right;color:var(--accent)}} footer{{margin-top:3rem;color:var(--muted);font-size:.72rem;border-top:1px solid var(--border);padding-top:1rem}}
</style></head><body>
<h1>📝 VisEDA — Text EDA Report</h1>
<p class="sub">Generated by <strong>VisEDA TextEDA</strong></p>
<p><span class="badge blue">{inv.get('total_documents', 0)} documents</span>
<span class="badge green">{inv.get('valid_documents', 0)} valid</span></p>
<h2>📦 Inventory</h2><div class="grid">
{self._html_card('Counts', {'Total documents': inv.get('total_documents'), 'Valid documents': inv.get('valid_documents'), 'Corrupt documents': inv.get('corrupt_documents')})}
{self._html_bar_card('Label Distribution', label_dist)}
{self._html_bar_card('Format Distribution', format_dist)}
{self._html_bar_card('Dominant Script', script_dist)}
</div>
{''.join(sections)}
<h2>🔤 Vocabulary</h2><div class="grid">
{self._html_card('Dataset Vocabulary', {'Total tokens': lexical.get('dataset_total_tokens'), 'Unique tokens': lexical.get('dataset_unique_tokens'), 'Type-token ratio': lexical.get('dataset_type_token_ratio'), 'Hapax count': lexical.get('dataset_hapax_count'), 'Hapax ratio': lexical.get('dataset_hapax_ratio')})}
{self._html_bar_card('Top Words', dict(top_words))}
{self._html_bar_card('Top Bigrams', dict(top_bigrams))}
</div>
<footer>Generated by VisEDA TextEDA.</footer>
</body></html>"""
        output.write_text(html_text, encoding="utf-8")
        self._log(f"Report saved → {output}")
        return str(output)

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------
    def _resolve_paths(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        recursive: bool,
    ) -> List[Path]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.glob("*")
                return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS)
            if path.is_file():
                if path.suffix.lower() not in self.SUPPORTED_EXTS:
                    raise ValueError(f"Unsupported text format: {path.suffix}")
                return [path]
            raise FileNotFoundError(path)
        paths = [Path(p) for p in source]
        unsupported = [p for p in paths if p.suffix.lower() not in self.SUPPORTED_EXTS]
        if unsupported:
            raise ValueError(f"Unsupported text format(s): {unsupported}")
        return sorted(paths)

    def _read_documents(
        self,
        path: Path,
        text_field: Optional[str],
        label_field: Optional[str],
    ) -> Tuple[List[Tuple[Optional[int], str, Optional[str]]], str]:
        ext = path.suffix.lower()
        raw, encoding = self._read_text(path)

        if ext in {".txt", ".text", ".md", ".rst", ".log"}:
            return [(None, raw, None)], encoding
        if ext in {".html", ".htm"}:
            parser = _HTMLTextExtractor()
            parser.feed(raw)
            return [(None, parser.text(), None)], encoding
        if ext in {".csv", ".tsv"}:
            delimiter = "\t" if ext == ".tsv" else ","
            return self._read_delimited(raw, delimiter, text_field, label_field), encoding
        if ext == ".json":
            obj = json.loads(raw)
            return self._extract_json_documents(obj, text_field, label_field), encoding
        if ext in {".jsonl", ".ndjson"}:
            rows = []
            for line_number, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                extracted = self._extract_json_documents(obj, text_field, label_field)
                for _, txt, lbl in extracted:
                    rows.append((line_number, txt, lbl))
            return rows, encoding
        raise ValueError(f"Unsupported text format: {ext}")

    def _read_text(self, path: Path) -> Tuple[str, str]:
        encodings = []
        if self.encoding:
            encodings.append(self.encoding)
        encodings.extend(["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"])
        seen = set()
        last_error: Optional[Exception] = None
        for enc in encodings:
            if enc in seen:
                continue
            seen.add(enc)
            try:
                return path.read_text(encoding=enc), enc
            except UnicodeError as exc:
                last_error = exc
        raise UnicodeError(f"Unable to decode {path}: {last_error}")

    def _read_delimited(
        self,
        raw: str,
        delimiter: str,
        text_field: Optional[str],
        label_field: Optional[str],
    ) -> List[Tuple[Optional[int], str, Optional[str]]]:
        reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
        fields = reader.fieldnames or []
        selected_text = self._select_field(fields, text_field, DEFAULT_TEXT_FIELDS, "text")
        selected_label = self._select_optional_field(fields, label_field, DEFAULT_LABEL_FIELDS)
        documents = []
        for i, row in enumerate(reader):
            value = row.get(selected_text)
            if value is None:
                continue
            label = row.get(selected_label) if selected_label else None
            documents.append((i, str(value), str(label) if label not in (None, "") else None))
        return documents

    def _extract_json_documents(
        self,
        obj: Any,
        text_field: Optional[str],
        label_field: Optional[str],
    ) -> List[Tuple[Optional[int], str, Optional[str]]]:
        if isinstance(obj, str):
            return [(None, obj, None)]
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            for key in ("data", "records", "items", "documents", "examples"):
                if isinstance(obj.get(key), list):
                    items = obj[key]
                    break
            else:
                # One object can be one document.
                items = [obj]
        else:
            return [(None, str(obj), None)]

        documents: List[Tuple[Optional[int], str, Optional[str]]] = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                documents.append((i, item, None))
                continue
            if not isinstance(item, dict):
                documents.append((i, str(item), None))
                continue
            fields = list(item.keys())
            selected_text = self._select_field(fields, text_field, DEFAULT_TEXT_FIELDS, "text")
            selected_label = self._select_optional_field(fields, label_field, DEFAULT_LABEL_FIELDS)
            value = item.get(selected_text)
            if value is None:
                continue
            label = item.get(selected_label) if selected_label else None
            documents.append((i, str(value), str(label) if label not in (None, "") else None))
        return documents

    @staticmethod
    def _select_field(fields: Sequence[str], requested: Optional[str], candidates: Sequence[str], kind: str) -> str:
        if requested:
            if requested not in fields:
                raise KeyError(f"Requested {kind}_field '{requested}' not found. Available fields: {list(fields)}")
            return requested
        lower_map = {f.casefold(): f for f in fields}
        for candidate in candidates:
            if candidate.casefold() in lower_map:
                return lower_map[candidate.casefold()]
        if len(fields) == 1:
            return fields[0]
        raise KeyError(
            f"Could not identify a {kind} field automatically. Available fields: {list(fields)}. "
            f"Pass {kind}_field explicitly."
        )

    @staticmethod
    def _select_optional_field(fields: Sequence[str], requested: Optional[str], candidates: Sequence[str]) -> Optional[str]:
        if requested:
            if requested not in fields:
                raise KeyError(f"Requested label_field '{requested}' not found. Available fields: {list(fields)}")
            return requested
        lower_map = {f.casefold(): f for f in fields}
        for candidate in candidates:
            if candidate.casefold() in lower_map:
                return lower_map[candidate.casefold()]
        return None

    # ------------------------------------------------------------------
    # Internal analysis
    # ------------------------------------------------------------------
    def _fill_stats(self, rec: TextRecord, text: str) -> None:
        chars = len(text)
        rec.char_count = chars
        rec.byte_count = len(text.encode("utf-8", errors="replace"))
        rec.normalised_hash = _normalised_hash(text)
        rec.preview = _normalise_whitespace(text)[:500]

        tokens = _tokenise(text, lowercase=self.lowercase, min_length=self.min_token_length)
        words = [t for t in tokens if WORD_RE.fullmatch(t)]
        numeric = [t for t in tokens if any(ch.isdigit() for ch in t)]
        sentences = _sentences(text)
        paragraphs = _paragraphs(text)
        lines = text.splitlines() if text else []
        nonempty_lines = [line.strip() for line in lines if line.strip()]

        rec.word_count = len(tokens)
        rec.alpha_word_count = len(words)
        rec.numeric_token_count = len(numeric)
        rec.unique_word_count = len(set(tokens))
        rec.sentence_count = len(sentences)
        rec.paragraph_count = len(paragraphs)
        rec.line_count = len(lines) if lines else (1 if text else 0)
        rec.nonempty_line_count = len(nonempty_lines)

        word_lengths = np.asarray([len(w) for w in tokens], dtype=float)
        sentence_lengths = np.asarray([
            len(_tokenise(s, lowercase=self.lowercase, min_length=self.min_token_length))
            for s in sentences
        ], dtype=float)
        paragraph_lengths = [
            len(_tokenise(p, lowercase=self.lowercase, min_length=self.min_token_length))
            for p in paragraphs
        ]

        rec.avg_word_length = float(word_lengths.mean()) if word_lengths.size else 0.0
        rec.median_word_length = float(np.median(word_lengths)) if word_lengths.size else 0.0
        rec.word_length_std = float(word_lengths.std()) if word_lengths.size else 0.0
        rec.avg_sentence_length = float(sentence_lengths.mean()) if sentence_lengths.size else 0.0
        rec.median_sentence_length = float(np.median(sentence_lengths)) if sentence_lengths.size else 0.0
        rec.sentence_length_std = float(sentence_lengths.std()) if sentence_lengths.size else 0.0
        rec.max_sentence_words = int(sentence_lengths.max()) if sentence_lengths.size else 0
        rec.avg_paragraph_words = float(np.mean(paragraph_lengths)) if paragraph_lengths else 0.0

        counts = Counter(tokens)
        rec.lexical_diversity = _safe_ratio(rec.unique_word_count, rec.word_count)
        rec.hapax_count = sum(1 for c in counts.values() if c == 1)
        rec.hapax_ratio = _safe_ratio(rec.hapax_count, rec.unique_word_count)
        rec.stopword_count = sum(c for t, c in counts.items() if t in self.stopwords)
        rec.stopword_ratio = _safe_ratio(rec.stopword_count, rec.word_count)
        rec.top_tokens = counts.most_common(self.top_n)
        rec.top_bigrams = [(" ".join(k), v) for k, v in Counter(_ngrams(tokens, 2)).most_common(self.top_n)]
        rec.top_trigrams = [(" ".join(k), v) for k, v in Counter(_ngrams(tokens, 3)).most_common(self.top_n)]

        rec.punctuation_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        rec.punctuation_rate = _safe_ratio(rec.punctuation_count, chars)
        rec.digit_count = sum(1 for ch in text if ch.isdigit())
        rec.digit_rate = _safe_ratio(rec.digit_count, chars)
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        rec.uppercase_count = sum(1 for ch in text if ch.isupper())
        rec.uppercase_rate = _safe_ratio(rec.uppercase_count, alpha_chars)
        rec.whitespace_count = sum(1 for ch in text if ch.isspace())
        rec.whitespace_rate = _safe_ratio(rec.whitespace_count, chars)
        rec.non_ascii_count = sum(1 for ch in text if ord(ch) > 127)
        rec.non_ascii_rate = _safe_ratio(rec.non_ascii_count, chars)
        rec.emoji_count = len(EMOJI_RE.findall(text))
        rec.url_count = len(URL_RE.findall(text))
        rec.email_count = len(EMAIL_RE.findall(text))
        rec.hashtag_count = len(HASHTAG_RE.findall(text))
        rec.mention_count = len(MENTION_RE.findall(text))

        if nonempty_lines:
            normal_lines = [_normalise_whitespace(line).casefold() for line in nonempty_lines]
            rec.repeated_line_fraction = 1.0 - len(set(normal_lines)) / len(normal_lines)
        else:
            rec.repeated_line_fraction = 0.0

        rec.empty = not bool(text.strip())
        rec.very_short = rec.word_count < self.short_document_words
        rec.very_long = rec.word_count > self.long_document_words
        rec.readability_flesch, rec.readability_grade = _readability(words, rec.sentence_count)
        rec.script_distribution, rec.dominant_script = _script_distribution(text)

    def _dataset_token_counts(self, exclude_stopwords: bool = False) -> Counter[str]:
        counts: Counter[str] = Counter()
        for i, rec in enumerate(self._records):
            if rec.is_corrupt:
                continue
            tokens = _tokenise(self.get_text(i), self.lowercase, self.min_token_length)
            if exclude_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords]
            counts.update(tokens)
        return counts

    def _dataset_ngram_counts(self, n: int, exclude_stopwords: bool = False) -> Counter[Tuple[str, ...]]:
        counts: Counter[Tuple[str, ...]] = Counter()
        for i, rec in enumerate(self._records):
            if rec.is_corrupt:
                continue
            tokens = _tokenise(self.get_text(i), self.lowercase, self.min_token_length)
            if exclude_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords]
            counts.update(_ngrams(tokens, n))
        return counts

    def _pairwise_fallback(
        self,
        texts: Sequence[str],
        names: List[str],
        max_features: int,
        exclude_stopwords: bool,
    ) -> Tuple[np.ndarray, List[str]]:
        doc_counts: List[Counter[str]] = []
        df: Counter[str] = Counter()
        corpus_counts: Counter[str] = Counter()
        for text in texts:
            tokens = _tokenise(text, self.lowercase, self.min_token_length)
            if exclude_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords]
            c = Counter(tokens)
            doc_counts.append(c)
            corpus_counts.update(c)
            df.update(c.keys())
        vocab = [t for t, _ in corpus_counts.most_common(max_features)]
        if not vocab:
            return np.zeros((len(texts), len(texts))), names
        index = {t: j for j, t in enumerate(vocab)}
        X = np.zeros((len(texts), len(vocab)), dtype=float)
        n_docs = len(texts)
        for i, counts in enumerate(doc_counts):
            total = sum(counts.values()) or 1
            for token, count in counts.items():
                j = index.get(token)
                if j is None:
                    continue
                tf = count / total
                idf = math.log((1 + n_docs) / (1 + df[token])) + 1.0
                X[i, j] = tf * idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        similarity = np.clip(Xn @ Xn.T, -1.0, 1.0)
        dist = 1.0 - similarity
        np.fill_diagonal(dist, 0.0)
        return dist, names

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _style_axis(ax) -> None:
        ax.set_facecolor("#f6f8fa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d7de")
        ax.grid(alpha=0.16, linewidth=0.6)

    def _plot_hist(self, ax, values: Sequence[float], title: str, xlabel: str = "Value") -> None:
        vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        self._style_axis(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Documents", fontsize=8)
        if vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        bins = min(30, max(5, int(np.sqrt(vals.size)) + 1))
        ax.hist(vals, bins=bins, alpha=0.9)
        mean = float(vals.mean())
        ax.axvline(mean, linestyle="--", linewidth=1)
        ax.legend([f"Mean = {mean:.3g}"], fontsize=7)

    def _plot_dataset_card(self, ax, valid: Sequence[TextRecord]) -> None:
        ax.axis("off")
        s = self.summary()
        inv = s["inventory"]
        lexical = s["lexical"]
        quality = s["quality"]
        lines = [
            f"Total documents:   {inv['total_documents']}",
            f"Valid documents:   {inv['valid_documents']}",
            f"Corrupt documents: {inv['corrupt_documents']}",
            f"Unique labels:      {len(inv['label_distribution'] or {})}",
            f"Mean words:         {s['length']['words']['mean']:.2f}",
            f"Vocabulary size:    {lexical['dataset_unique_tokens']:,}",
            f"Type-token ratio:   {lexical['dataset_type_token_ratio']:.4f}",
            f"Exact dup. frac:    {quality['exact_duplicate_fraction']:.4f}",
        ]
        ax.set_title("Dataset Overview", fontsize=10)
        ax.text(
            0.03, 0.92, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaeff5", edgecolor="#d0d7de"),
            transform=ax.transAxes,
        )

    def _plot_label_distribution(self, ax, valid: Sequence[TextRecord]) -> None:
        counts = Counter(r.label for r in valid if r.label is not None)
        self._style_axis(ax)
        ax.set_title("Label Distribution", fontsize=10)
        if not counts:
            ax.text(0.5, 0.5, "No labels", ha="center", va="center", transform=ax.transAxes)
            return
        labels, vals = zip(*counts.most_common(25)[::-1])
        ax.barh(labels, vals)
        ax.set_xlabel("Documents", fontsize=8)

    def _plot_frequency(self, ax, data: Mapping[str, int], title: str) -> None:
        self._style_axis(ax)
        ax.set_title(title, fontsize=10)
        if not data:
            ax.text(0.5, 0.5, "No tokens", ha="center", va="center", transform=ax.transAxes)
            return
        items = list(data.items())
        labels = [str(k) for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        ax.barh(labels, vals)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Frequency", fontsize=8)

    def _plot_script_distribution(self, ax, valid: Sequence[TextRecord]) -> None:
        counts = Counter(r.dominant_script for r in valid if r.dominant_script)
        self._style_axis(ax)
        ax.set_title("Dominant Script", fontsize=10)
        if not counts:
            ax.text(0.5, 0.5, "No alphabetic text", ha="center", va="center", transform=ax.transAxes)
            return
        labels, vals = zip(*counts.most_common())
        ax.bar(labels, vals)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("Documents", fontsize=8)

    def _plot_format_distribution(self, ax, valid: Sequence[TextRecord]) -> None:
        counts = Counter(r.file_ext for r in valid)
        self._style_axis(ax)
        ax.set_title("Format Distribution", fontsize=10)
        labels, vals = zip(*counts.items())
        ax.bar(labels, vals)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("Documents", fontsize=8)

    def _plot_scatter(self, ax, valid: Sequence[TextRecord], x_attr: str, y_attr: str, title: str) -> None:
        pairs = [
            (getattr(r, x_attr), getattr(r, y_attr))
            for r in valid
            if getattr(r, x_attr) is not None and getattr(r, y_attr) is not None
            and np.isfinite(getattr(r, x_attr)) and np.isfinite(getattr(r, y_attr))
        ]
        self._style_axis(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_attr.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel(y_attr.replace("_", " ").title(), fontsize=8)
        if not pairs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        x, y = zip(*pairs)
        ax.scatter(x, y, alpha=0.65, s=18)

    def _plot_pairwise_panel(self, ax, max_documents: int) -> None:
        self._style_axis(ax)
        ax.set_title("Pairwise Document Distances", fontsize=10)
        try:
            dist, names = self.pairwise_document_distances(max_documents=max_documents)
            im = ax.imshow(dist, aspect="auto")
            if len(names) <= 25:
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=6)
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True, transform=ax.transAxes)

    def _plot_quality_flags(self, ax, valid: Sequence[TextRecord]) -> None:
        labels = ["Empty", "Very short", "Very long", "Exact duplicate"]
        hashes = Counter(r.normalised_hash for r in valid)
        values = [
            sum(bool(r.empty) for r in valid),
            sum(bool(r.very_short) for r in valid),
            sum(bool(r.very_long) for r in valid),
            sum(hashes[r.normalised_hash] > 1 for r in valid),
        ]
        rates = [_safe_ratio(v, len(valid)) for v in values]
        self._style_axis(ax)
        ax.set_title("Quality Flags / Rates", fontsize=10)
        ax.bar(labels, rates)
        ax.set_ylim(0, max(1.0, max(rates, default=0) * 1.15))
        ax.set_ylabel("Fraction of documents", fontsize=8)
        ax.tick_params(axis="x", rotation=25, labelsize=8)

    def _plot_record_card(self, ax, rec: TextRecord) -> None:
        ax.axis("off")
        lines = [
            f"Path:              {rec.name}",
            f"Label:             {rec.label}",
            f"Words:             {rec.word_count:,}",
            f"Unique words:      {rec.unique_word_count:,}",
            f"Sentences:         {rec.sentence_count:,}",
            f"Paragraphs:        {rec.paragraph_count:,}",
            f"Lexical diversity: {rec.lexical_diversity:.4f}",
            f"Stopword ratio:    {rec.stopword_ratio:.4f}",
            f"Readability:       {rec.readability_flesch if rec.readability_flesch is not None else 'N/A'}",
            f"Dominant script:   {rec.dominant_script}",
        ]
        ax.set_title("Document Overview", fontsize=10)
        ax.text(
            0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaeff5", edgecolor="#d0d7de"),
            transform=ax.transAxes,
        )

    def _plot_text_preview(self, ax, text: str) -> None:
        self._style_axis(ax)
        ax.set_title("Text Preview", fontsize=10)
        preview = _normalise_whitespace(text)[:1_300]
        ax.text(0.02, 0.95, preview, va="top", ha="left", wrap=True, fontsize=9, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_character_categories(self, ax, rec: TextRecord) -> None:
        labels = ["Punctuation", "Digits", "Uppercase", "Whitespace", "Non-ASCII"]
        values = [rec.punctuation_rate, rec.digit_rate, rec.uppercase_rate, rec.whitespace_rate, rec.non_ascii_rate]
        self._style_axis(ax)
        ax.set_title("Character Category Rates", fontsize=10)
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("Rate", fontsize=8)

    def _plot_token_coverage(self, ax, counts: Counter[str]) -> None:
        self._style_axis(ax)
        ax.set_title("Cumulative Token Coverage", fontsize=10)
        freqs = np.asarray(sorted(counts.values(), reverse=True), dtype=float)
        if freqs.size == 0:
            ax.text(0.5, 0.5, "No tokens", ha="center", va="center", transform=ax.transAxes)
            return
        coverage = np.cumsum(freqs) / freqs.sum()
        ax.plot(np.arange(1, len(coverage) + 1), coverage)
        ax.axhline(0.8, linestyle="--", linewidth=1)
        ax.set_xlabel("Top-N vocabulary items", fontsize=8)
        ax.set_ylabel("Token coverage", fontsize=8)
        ax.set_ylim(0, 1.03)

    def _plot_script_record(self, ax, rec: TextRecord) -> None:
        self._style_axis(ax)
        ax.set_title("Writing Script Composition", fontsize=10)
        data = rec.script_distribution or {}
        if not data:
            ax.text(0.5, 0.5, "No alphabetic text", ha="center", va="center", transform=ax.transAxes)
            return
        labels, values = zip(*sorted(data.items(), key=lambda x: -x[1]))
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("Fraction of alphabetic characters", fontsize=8)

    def _plot_quality_record(self, ax, rec: TextRecord) -> None:
        labels = ["Short", "Long", "Repeated lines", "Stopwords", "Non-ASCII"]
        values = [float(rec.very_short), float(rec.very_long), rec.repeated_line_fraction, rec.stopword_ratio, rec.non_ascii_rate]
        self._style_axis(ax)
        ax.set_title("Document Quality / Composition", fontsize=10)
        ax.bar(labels, values)
        ax.set_ylim(0, max(1.0, max(values, default=0) * 1.15))
        ax.tick_params(axis="x", rotation=35, labelsize=7)

    def _plot_sentence_sequence(self, ax, sentences: Sequence[str]) -> None:
        lengths = [len(_tokenise(s, self.lowercase, self.min_token_length)) for s in sentences]
        self._style_axis(ax)
        ax.set_title("Sentence Length Sequence", fontsize=10)
        if not lengths:
            ax.text(0.5, 0.5, "No sentences", ha="center", va="center", transform=ax.transAxes)
            return
        ax.plot(range(1, len(lengths) + 1), lengths, marker="o", markersize=3)
        ax.set_xlabel("Sentence index", fontsize=8)
        ax.set_ylabel("Words", fontsize=8)

    def _finalise(self, fig, save_path: Optional[str], dpi: int) -> None:
        plt = _plt()
        try:
            fig.set_constrained_layout(False)
        except Exception:
            pass
        fig.subplots_adjust(left=0.055, right=0.975, top=0.965, bottom=0.055, hspace=0.68, wspace=0.42)
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0.18, facecolor="white")
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_html(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:,.4f}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return html.escape(str(value))

    def _html_card(self, title: str, data: Mapping[str, Any]) -> str:
        rows = "".join(
            f'<div class="stat"><span>{html.escape(str(k))}</span><span class="val">{self._fmt_html(v)}</span></div>'
            for k, v in data.items()
        )
        return f'<div class="card"><h3>{html.escape(title)}</h3>{rows}</div>'

    def _html_bar_card(self, title: str, data: Mapping[Any, Any]) -> str:
        if not data:
            return self._html_card(title, {"Status": "No data"})
        max_value = max(float(v) for v in data.values()) or 1.0
        rows = "".join(
            '<div class="bar-row">'
            f'<span class="bar-label">{html.escape(str(k))}</span>'
            f'<div class="bar"><div class="bar-fill" style="width:{100 * float(v) / max_value:.1f}%"></div></div>'
            f'<span class="bar-count">{self._fmt_html(v)}</span></div>'
            for k, v in data.items()
        )
        return f'<div class="card"><h3>{html.escape(title)}</h3>{rows}</div>'

    def _html_section(self, title: str, section: Mapping[str, Any], skip_complex: bool = False) -> str:
        cards = []
        for key, value in section.items():
            if isinstance(value, dict) and {"mean", "min", "max"}.intersection(value):
                cards.append(self._html_card(key.replace("_", " ").title(), value))
            elif not isinstance(value, (dict, list, tuple)):
                cards.append(self._html_card(key.replace("_", " ").title(), {"value": value}))
            elif not skip_complex and isinstance(value, dict):
                cards.append(self._html_card(key.replace("_", " ").title(), value))
        return f'<h2>{html.escape(title)}</h2><div class="grid">{"".join(cards)}</div>'

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @staticmethod
    def _imbalance_ratio(label_dist: Optional[Mapping[str, int]]) -> Optional[float]:
        if not label_dist:
            return None
        values = [v for v in label_dist.values() if v > 0]
        if not values:
            return None
        return float(max(values) / min(values))

    @staticmethod
    def _record_key(rec: TextRecord, fallback_index: int) -> str:
        return f"{rec.path}|{rec.source_index if rec.source_index is not None else fallback_index}"

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() or load_texts() before requesting analysis.")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[viseda] {message}")


__all__ = ["TextEDA", "TextRecord"]
