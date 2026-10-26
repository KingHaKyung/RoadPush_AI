"""Policy contract for sparse structural-support evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseRowSpanSupportPolicy:
    """Surface-independent rejection policy for weak row-span additions."""

    ring_radius_px: int = 3
    minimum_component_pixels: int = 20
    maximum_source_support_ratio: float = 0.08
    route_guard_radius_px: int = 4


DEFAULT_SPARSE_ROW_SPAN_SUPPORT_POLICY = SparseRowSpanSupportPolicy()
