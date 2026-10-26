"""Sparse structural-support geometry primitives."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.sparse_support import (
    SparseRowSpanSupportResult,
)
from surface_passability.policies.sparse_support import (
    DEFAULT_SPARSE_ROW_SPAN_SUPPORT_POLICY,
    SparseRowSpanSupportPolicy,
)


def route_guard_from_path(
    shape: tuple[int, int],
    route_path: list[tuple[int, int]],
    radius_px: int,
) -> np.ndarray:
    """Rasterize a route whose points are stored as ``(y, x)``."""

    guard = np.zeros(shape, np.uint8)
    if not route_path:
        return guard.astype(bool)
    points = np.asarray(
        [(int(round(x)), int(round(y))) for y, x in route_path], np.int32
    )
    if len(points) == 1:
        cv2.circle(guard, tuple(points[0]), radius_px, 1, -1)
    else:
        cv2.polylines(guard, [points], False, 1, radius_px * 2 + 1)
    return guard > 0


def evaluate_sparse_row_span_support(
    selected_sources: np.ndarray,
    structural_mask: np.ndarray,
    route_guard: np.ndarray,
    policy: SparseRowSpanSupportPolicy,
) -> SparseRowSpanSupportResult:
    """Reject synthesized components with weak source and route support."""

    if not (
        selected_sources.shape == structural_mask.shape == route_guard.shape
    ):
        raise ValueError("selected, structural, and route guard shapes must match")

    selected = selected_sources.astype(bool)
    structural = structural_mask.astype(bool)
    guard = route_guard.astype(bool)
    additions = structural & ~selected
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        additions.astype(np.uint8), 8
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (policy.ring_radius_px * 2 + 1,) * 2
    )
    rejected = np.zeros_like(structural)
    records = []
    for component_index in range(1, count):
        component = components == component_index
        pixels = int(component.sum())
        if pixels < policy.minimum_component_pixels:
            continue
        ring = (cv2.dilate(component.astype(np.uint8), kernel) > 0) & ~component
        source_support_pixels = int((ring & selected).sum())
        support_ratio = source_support_pixels / max(
            1, pixels + source_support_pixels
        )
        route_overlap_pixels = int((component & guard).sum())
        reject = bool(
            support_ratio < policy.maximum_source_support_ratio
            and route_overlap_pixels == 0
        )
        if reject:
            rejected |= component
        x, y, width, height, _ = [int(v) for v in stats[component_index]]
        records.append(
            {
                "component_index": component_index,
                "pixels": pixels,
                "bbox_xywh": [x, y, width, height],
                "source_support_pixels": source_support_pixels,
                "source_support_ratio": support_ratio,
                "route_guard_overlap_pixels": route_overlap_pixels,
                "rejected": reject,
            }
        )
    return SparseRowSpanSupportResult(
        additions=additions,
        route_guard=guard,
        rejected=rejected,
        after_structural=structural & ~rejected,
        records=records,
    )


def apply_sparse_row_span_support(
    selected_sources: np.ndarray,
    structural_mask: np.ndarray,
    route_path: list[tuple[int, int]],
    policy: SparseRowSpanSupportPolicy = (
        DEFAULT_SPARSE_ROW_SPAN_SUPPORT_POLICY
    ),
) -> SparseRowSpanSupportResult:
    """Evaluate sparse support using the owning surface's route path."""

    route_guard = route_guard_from_path(
        structural_mask.shape,
        route_path,
        policy.route_guard_radius_px,
    )
    return evaluate_sparse_row_span_support(
        selected_sources,
        structural_mask,
        route_guard,
        policy,
    )


def sparse_row_span_support_record(
    result: SparseRowSpanSupportResult,
    policy: SparseRowSpanSupportPolicy = (
        DEFAULT_SPARSE_ROW_SPAN_SUPPORT_POLICY
    ),
) -> dict[str, object]:
    """Serialize sparse-support evidence using the frozen response contract."""

    rejected = [record for record in result.records if record["rejected"]]
    return {
        "policy": "phase5_surface_neutral_sparse_row_span_support_v1",
        "ring_radius_px": policy.ring_radius_px,
        "minimum_component_pixels": policy.minimum_component_pixels,
        "maximum_source_support_ratio_exclusive": (
            policy.maximum_source_support_ratio
        ),
        "route_guard_radius_px": policy.route_guard_radius_px,
        "rejected_component_count": len(rejected),
        "removed_structural_pixels": int(result.rejected.sum()),
        "rejected_components": rejected,
    }
