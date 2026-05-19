"""
viseda.core.base
----------------
Abstract base class shared by all EDA modules.
"""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseEDA(abc.ABC):
    """Abstract base for all VisEDA analysers."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._results: Dict[str, Any] = {}
        self._timing: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface every subclass must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self, source: Any) -> "BaseEDA":
        """Load data from *source* (path, directory, array, …)."""

    @abc.abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary dict."""

    @abc.abstractmethod
    def plot(self, **kwargs) -> None:
        """Produce the main visualisation panel."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[viseda] {msg}")

    def _time(self, key: str):
        """Context-manager-like timing helper (use as decorator or manually)."""
        return _Timer(key, self._timing)

    @property
    def results(self) -> Dict[str, Any]:
        return self._results

    def _store(self, key: str, value: Any) -> None:
        self._results[key] = value


class _Timer:
    def __init__(self, key: str, store: Dict[str, float]):
        self.key = key
        self.store = store

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.store[self.key] = time.perf_counter() - self._start