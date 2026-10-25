from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SparseRowSpanSupportPolicy:
    """Surface-independent rejection policy for weak row-span additions."""

    ring_radius_px: int = 3
    minimum_component_pixels: int = 20
    maximum_source_support_ratio: float = 0.08
    route_guard_radius_px: int = 4


PHASE5_SPARSE_ROW_SPAN_SUPPORT_POLICY = SparseRowSpanSupportPolicy()


@dataclass
class SparseRowSpanSupportResult:
    additions: np.ndarray
    route_guard: np.ndarray
    rejected: np.ndarray
    after_structural: np.ndarray
    records: list[dict]


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
    """Reject structurally synthesized islands with little source support.

    The calculation is deliberately surface-neutral. Callers define what counts
    as selected source evidence and supply the route guard belonging to that
    surface. This keeps Road and Sidewalk ownership/policy outside the geometry
    rule while sharing the same implementation.
    """

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
    policy: SparseRowSpanSupportPolicy = PHASE5_SPARSE_ROW_SPAN_SUPPORT_POLICY,
) -> SparseRowSpanSupportResult:
    """Production adapter using the owning surface's explicit route path."""

    route_guard = route_guard_from_path(
        structural_mask.shape, route_path, policy.route_guard_radius_px
    )
    return evaluate_sparse_row_span_support(
        selected_sources, structural_mask, route_guard, policy
    )


def sparse_row_span_support_record(
    result: SparseRowSpanSupportResult,
    policy: SparseRowSpanSupportPolicy = PHASE5_SPARSE_ROW_SPAN_SUPPORT_POLICY,
) -> dict:
    rejected = [record for record in result.records if record["rejected"]]
    return {
        "policy": "phase5_surface_neutral_sparse_row_span_support_v1",
        "ring_radius_px": policy.ring_radius_px,
        "minimum_component_pixels": policy.minimum_component_pixels,
        "maximum_source_support_ratio_exclusive": policy.maximum_source_support_ratio,
        "route_guard_radius_px": policy.route_guard_radius_px,
        "rejected_component_count": len(rejected),
        "removed_structural_pixels": int(result.rejected.sum()),
        "rejected_components": rejected,
    }
