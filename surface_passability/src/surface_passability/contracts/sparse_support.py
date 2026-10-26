"""Typed results for sparse structural-support evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SparseRowSpanSupportResult:
    """Masks and component evidence produced by sparse-support evaluation."""

    additions: np.ndarray
    route_guard: np.ndarray
    rejected: np.ndarray
    after_structural: np.ndarray
    records: list[dict[str, object]]
