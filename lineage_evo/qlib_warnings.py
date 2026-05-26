"""Small warning filters for noisy Qlib/pandas execution paths."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def suppress_qlib_all_nan_slice_warning() -> Iterator[None]:
    """Hide pandas rolling warnings while keeping deterministic NaN checks."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        yield
