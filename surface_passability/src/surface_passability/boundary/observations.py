"""Route-relative row observation primitives."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import cv2

from surface_passability.contracts.boundaries import (
    BoundaryEndpointKind,
    BoundaryRowObservation,
    BoundarySupportEvidence,
    BoundaryWidthState,
    OuterBoundaryObservation,
)
from surface_passability.geometry.intervals import true_intervals
from surface_passability.obstacles.taxonomy import (
    HAZARD_CLASS_IDS,
    OCCUPANCY_CLASS_IDS,
    TRANSIENT_OCCLUSION_CLASS_IDS,
)
from surface_passability.policies.boundaries import (
    BOUNDARY_ENDPOINT_CONTEXT_RADIUS_IMAGE_FRACTION,
    BOUNDARY_ENDPOINT_FRAME_MARGIN_PX,
    BOUNDARY_ENDPOINT_MINIMUM_CLASS_FRACTION,
    BOUNDARY_ENDPOINT_MINIMUM_CONTEXT_RADIUS_PX,
    BOUNDARY_ENDPOINT_PHYSICAL_CLASS_IDS,
    BOUNDARY_ROWS_ANCHOR_TOLERANCE_WIDTH_FRACTION,
    BOUNDARY_ROWS_BAND_RADIUS_IMAGE_FRACTION,
    BOUNDARY_ROWS_LEVEL_COUNT,
    BOUNDARY_ROWS_MINIMUM_ANCHOR_TOLERANCE_PX,
    BOUNDARY_ROWS_MINIMUM_BAND_RADIUS_PX,
    BOUNDARY_ROWS_MINIMUM_BAND_SUPPORT_FRACTION,
    BOUNDARY_ROWS_MINIMUM_VERTICAL_SPAN_FRACTION,
    BOUNDARY_ROWS_ROUTE_EXTENSION_FRACTION,
    BOUNDARY_ROWS_ROUTELESS_BOTTOM_FRACTION,
    BOUNDARY_ROWS_ROUTELESS_TOP_FRACTION,
    BOUNDARY_SUPPORT_CURB_CLASS_IDS,
    BOUNDARY_SUPPORT_GRADIENT_NORMALIZATION,
    BOUNDARY_SUPPORT_GRADIENT_PERCENTILE,
    BOUNDARY_SUPPORT_HORIZONTAL_RADIUS_PX,
    BOUNDARY_SUPPORT_MAXIMUM_GRADIENT,
    BOUNDARY_SUPPORT_SOBEL_KERNEL_SIZE,
    BOUNDARY_SUPPORT_VERTICAL_RADIUS_PX,
    OUTER_BOUNDARY_ROWS_BAND_RADIUS_IMAGE_FRACTION,
    OUTER_BOUNDARY_ROWS_LEVEL_COUNT,
    OUTER_BOUNDARY_ROWS_MINIMUM_BAND_RADIUS_PX,
    OUTER_BOUNDARY_ROWS_MINIMUM_BAND_SUPPORT_FRACTION,
    OUTER_BOUNDARY_ROWS_MINIMUM_VERTICAL_SPAN_FRACTION,
    OUTER_BOUNDARY_ROWS_ROUTE_EXTENSION_FRACTION,
    OUTER_BOUNDARY_ROWS_ROUTELESS_BOTTOM_FRACTION,
    OUTER_BOUNDARY_ROWS_ROUTELESS_TOP_FRACTION,
)
from surface_passability.surfaces.routes import route_x_for_nearest_row


class StraightBoundaryObservationPolicy(Protocol):
    """Configuration required to observe clean straight-boundary rows."""

    clean_fill_ratio: float
    endpoint_contact_radius_px: int
    minimum_span_fraction: float
    minimum_span_px: int


def collect_straight_boundary_rows(
    structural_mask: np.ndarray,
    blocker_mask: np.ndarray,
    obstacle_silhouette_mask: np.ndarray,
    policy: StraightBoundaryObservationPolicy,
) -> list[dict[str, Any]]:
    """Collect row endpoints and local blocker/silhouette contact evidence."""

    height, width = structural_mask.shape
    radius = policy.endpoint_contact_radius_px
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    blocker_contact = (
        cv2.dilate(blocker_mask.astype(np.uint8), kernel) > 0
    )
    silhouette_contact = (
        cv2.dilate(obstacle_silhouette_mask.astype(np.uint8), kernel) > 0
    )
    minimum_span = max(
        policy.minimum_span_px,
        int(np.ceil(policy.minimum_span_fraction * width)),
    )
    records: list[dict[str, Any]] = []
    for row_y in range(height):
        columns = np.flatnonzero(structural_mask[row_y])
        if not columns.size:
            continue
        left, right = int(columns[0]), int(columns[-1])
        span = right - left + 1
        fill = float(columns.size / span)
        left_frame = left == 0
        right_frame = right == width - 1
        left_blocker = bool(blocker_contact[row_y, left])
        right_blocker = bool(blocker_contact[row_y, right])
        left_silhouette = bool(silhouette_contact[row_y, left])
        right_silhouette = bool(silhouette_contact[row_y, right])
        reasons: list[str] = []
        if span < minimum_span:
            reasons.append("short_span")
        if fill < policy.clean_fill_ratio:
            reasons.append("low_fill")
        if left_frame:
            reasons.append("left_frame_contact")
        if right_frame:
            reasons.append("right_frame_contact")
        if left_blocker:
            reasons.append("left_endpoint_blocker_contact")
        if right_blocker:
            reasons.append("right_endpoint_blocker_contact")
        if left_silhouette:
            reasons.append("left_endpoint_obstacle_silhouette_contact")
        if right_silhouette:
            reasons.append("right_endpoint_obstacle_silhouette_contact")
        records.append(
            {
                "y": row_y,
                "left": left,
                "right": right,
                "span": span,
                "fill_ratio": fill,
                "blocker_share": float(
                    np.count_nonzero(blocker_mask[row_y, left : right + 1])
                    / span
                ),
                "frame_contact": left_frame or right_frame,
                "left_frame_contact": left_frame,
                "right_frame_contact": right_frame,
                "left_blocker_contact": left_blocker,
                "right_blocker_contact": right_blocker,
                "left_obstacle_silhouette_contact": left_silhouette,
                "right_obstacle_silhouette_contact": right_silhouette,
                "obstacle_contact": (
                    left_blocker
                    or right_blocker
                    or left_silhouette
                    or right_silhouette
                ),
                "clean_bilateral": not reasons,
                "reasons": reasons,
            }
        )
    return records


def compare_raw_and_pruned_structure(
    raw_structural_mask: np.ndarray | None,
    pruned_structural_mask: np.ndarray,
) -> tuple[dict[str, object] | None, np.ndarray]:
    """Describe pixels removed by pruning without granting runtime authority."""

    if raw_structural_mask is None:
        return None, np.zeros_like(pruned_structural_mask)
    raw = np.asarray(raw_structural_mask, dtype=bool)
    if raw.shape != pruned_structural_mask.shape:
        raise ValueError("raw_structural_mask must match structural_mask")
    removed = raw & ~pruned_structural_mask
    rows, columns = np.nonzero(removed)
    return {
        "available": True,
        "raw_pixels": int(np.count_nonzero(raw)),
        "pruned_pixels": int(np.count_nonzero(pruned_structural_mask)),
        "removed_pixels": int(np.count_nonzero(removed)),
        "removed_bbox_xyxy": (
            [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()),
                int(rows.max()),
            ]
            if columns.size
            else None
        ),
        "authority": "diagnostic_only",
    }, removed


def select_route_interval(
    row: np.ndarray,
    anchor_x: float,
    tolerance: int,
) -> tuple[int, int] | None:
    """Select the containing or nearest inclusive interval for a route anchor."""

    choices = true_intervals(row)
    if not choices:
        return None
    containing = [
        interval
        for interval in choices
        if interval[0] - tolerance <= anchor_x <= interval[1] + tolerance
    ]
    pool = containing if containing else choices
    return min(
        pool,
        key=lambda interval: abs((interval[0] + interval[1]) / 2 - anchor_x),
    )


def measure_boundary_support(
    grayscale: np.ndarray,
    class_map: np.ndarray,
    row_y: int,
    column_x: int,
) -> BoundarySupportEvidence:
    """Measure local horizontal-edge strength and curb semantic context."""

    height, width = grayscale.shape
    y0 = max(0, row_y - BOUNDARY_SUPPORT_VERTICAL_RADIUS_PX)
    y1 = min(height, row_y + BOUNDARY_SUPPORT_VERTICAL_RADIUS_PX + 1)
    x0 = max(0, column_x - BOUNDARY_SUPPORT_HORIZONTAL_RADIUS_PX)
    x1 = min(width, column_x + BOUNDARY_SUPPORT_HORIZONTAL_RADIUS_PX + 1)
    sobel = cv2.Sobel(
        grayscale,
        cv2.CV_32F,
        1,
        0,
        ksize=BOUNDARY_SUPPORT_SOBEL_KERNEL_SIZE,
    )
    patch = np.abs(sobel[y0:y1, x0:x1])
    gradient = (
        float(np.percentile(patch, BOUNDARY_SUPPORT_GRADIENT_PERCENTILE)
              / BOUNDARY_SUPPORT_GRADIENT_NORMALIZATION)
        if patch.size
        else 0.0
    )
    context = class_map[y0:y1, x0:x1]
    curb_fraction = (
        float(np.isin(context, BOUNDARY_SUPPORT_CURB_CLASS_IDS).mean())
        if context.size
        else 0.0
    )
    return BoundarySupportEvidence(
        gradient=min(BOUNDARY_SUPPORT_MAXIMUM_GRADIENT, gradient),
        curb_fraction=curb_fraction,
    )


def classify_boundary_endpoint(
    class_map: np.ndarray,
    row_y: int,
    column_x: int,
    side: str,
) -> BoundaryEndpointKind:
    """Classify why one observed surface boundary terminates."""

    height, width = class_map.shape
    if (
        column_x <= BOUNDARY_ENDPOINT_FRAME_MARGIN_PX
        or column_x >= width - BOUNDARY_ENDPOINT_FRAME_MARGIN_PX - 1
    ):
        return "frame_truncated"
    radius_y = max(
        BOUNDARY_ENDPOINT_MINIMUM_CONTEXT_RADIUS_PX,
        round(height * BOUNDARY_ENDPOINT_CONTEXT_RADIUS_IMAGE_FRACTION),
    )
    radius_x = max(
        BOUNDARY_ENDPOINT_MINIMUM_CONTEXT_RADIUS_PX,
        round(width * BOUNDARY_ENDPOINT_CONTEXT_RADIUS_IMAGE_FRACTION),
    )
    if side == "left":
        x0, x1 = max(0, column_x - radius_x), min(width, column_x + 2)
    else:
        x0, x1 = max(0, column_x - 1), min(width, column_x + radius_x + 1)
    y0, y1 = max(0, row_y - radius_y), min(height, row_y + radius_y + 1)
    context = class_map[y0:y1, x0:x1]
    blocked_ids = (
        tuple(OCCUPANCY_CLASS_IDS)
        + tuple(TRANSIENT_OCCLUSION_CLASS_IDS)
        + tuple(HAZARD_CLASS_IDS)
    )
    if (
        context.size
        and np.isin(context, blocked_ids).mean()
        >= BOUNDARY_ENDPOINT_MINIMUM_CLASS_FRACTION
    ):
        return "obstacle_occlusion"
    if (
        context.size
        and np.isin(context, BOUNDARY_ENDPOINT_PHYSICAL_CLASS_IDS).mean()
        >= BOUNDARY_ENDPOINT_MINIMUM_CLASS_FRACTION
    ):
        return "physical_boundary"
    return "semantic_boundary"


def collect_boundary_rows(
    structural_mask: np.ndarray,
    route_path: list[tuple[int, int]],
    class_map: np.ndarray,
    grayscale: np.ndarray,
) -> tuple[BoundaryRowObservation, ...]:
    """Collect route-relative boundary observations at accepted vertical levels."""

    height, width = structural_mask.shape
    if route_path:
        bottom = int(max(y for y, _ in route_path))
        top = int(min(y for y, _ in route_path))
        component_rows = np.flatnonzero(structural_mask.any(axis=1))
        if len(component_rows):
            top = min(
                top,
                int(component_rows[0] + BOUNDARY_ROWS_ROUTE_EXTENSION_FRACTION * (bottom - component_rows[0])),
            )
    else:
        bottom = round(height * BOUNDARY_ROWS_ROUTELESS_BOTTOM_FRACTION)
        top = round(height * BOUNDARY_ROWS_ROUTELESS_TOP_FRACTION)
    if bottom - top < round(height * BOUNDARY_ROWS_MINIMUM_VERTICAL_SPAN_FRACTION):
        return ()
    levels = np.unique(np.linspace(bottom, top, BOUNDARY_ROWS_LEVEL_COUNT).round().astype(int))[::-1]
    band = max(BOUNDARY_ROWS_MINIMUM_BAND_RADIUS_PX, round(height * BOUNDARY_ROWS_BAND_RADIUS_IMAGE_FRACTION))
    records: list[BoundaryRowObservation] = []
    anchor = width / 2
    for y in levels:
        anchor = route_x_for_nearest_row(route_path, int(y), anchor)
        y0, y1 = max(0, y - band), min(height, y + band + 1)
        row = structural_mask[y0:y1].sum(axis=0) >= max(
            1,
            round((y1 - y0) * BOUNDARY_ROWS_MINIMUM_BAND_SUPPORT_FRACTION),
        )
        selected = select_route_interval(
            row,
            anchor,
            max(BOUNDARY_ROWS_MINIMUM_ANCHOR_TOLERANCE_PX, round(width * BOUNDARY_ROWS_ANCHOR_TOLERANCE_WIDTH_FRACTION)),
        )
        if selected is None:
            continue
        left, right = selected
        left_support = measure_boundary_support(grayscale, class_map, int(y), left)
        right_support = measure_boundary_support(grayscale, class_map, int(y), right)
        records.append(BoundaryRowObservation(
            y=int(y), left=left, right=right, width=right - left + 1,
            anchor_x=anchor,
            left_gradient=left_support.gradient,
            right_gradient=right_support.gradient,
            left_curb=left_support.curb_fraction,
            right_curb=right_support.curb_fraction,
            left_kind=classify_boundary_endpoint(class_map, int(y), left, "left"),
            right_kind=classify_boundary_endpoint(class_map, int(y), right, "right"),
        ))
    return tuple(records)


def sample_outer_boundary_rows(
    component_mask: np.ndarray,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    class_map: np.ndarray,
) -> tuple[OuterBoundaryObservation, ...]:
    """Sample the outer endpoints of a routed component at accepted row levels."""

    height, width = component_mask.shape
    occupied_rows = np.flatnonzero(component_mask.any(axis=1))
    if not len(occupied_rows):
        return ()
    if route_path:
        bottom = max(y for y, _ in route_path)
        top = min(y for y, _ in route_path)
        top = min(
            top,
            int(
                occupied_rows[0]
                + OUTER_BOUNDARY_ROWS_ROUTE_EXTENSION_FRACTION
                * (bottom - occupied_rows[0])
            ),
        )
    else:
        bottom = min(
            height - 1,
            round(height * OUTER_BOUNDARY_ROWS_ROUTELESS_BOTTOM_FRACTION),
        )
        top = max(
            int(occupied_rows[0]),
            round(height * OUTER_BOUNDARY_ROWS_ROUTELESS_TOP_FRACTION),
        )
    if bottom - top < round(
        height * OUTER_BOUNDARY_ROWS_MINIMUM_VERTICAL_SPAN_FRACTION
    ):
        return ()
    levels = np.unique(
        np.linspace(bottom, top, OUTER_BOUNDARY_ROWS_LEVEL_COUNT)
        .round()
        .astype(int)
    )[::-1]
    band = max(
        OUTER_BOUNDARY_ROWS_MINIMUM_BAND_RADIUS_PX,
        round(height * OUTER_BOUNDARY_ROWS_BAND_RADIUS_IMAGE_FRACTION),
    )
    records: list[OuterBoundaryObservation] = []
    for y in levels:
        y0, y1 = max(0, y - band), min(height, y + band + 1)
        row = component_mask[y0:y1].sum(axis=0) >= max(
            1,
            round(
                (y1 - y0)
                * OUTER_BOUNDARY_ROWS_MINIMUM_BAND_SUPPORT_FRACTION
            ),
        )
        columns = np.flatnonzero(row)
        if not len(columns):
            continue
        left, right = int(columns[0]), int(columns[-1])
        left_kind = classify_boundary_endpoint(class_map, int(y), left, "left")
        right_kind = classify_boundary_endpoint(
            class_map,
            int(y),
            right,
            "right",
        )
        width_state: BoundaryWidthState
        if (
            left_kind not in ("frame_truncated", "obstacle_occlusion")
            and right_kind not in ("frame_truncated", "obstacle_occlusion")
        ):
            width_state = "complete"
        elif (
            left_kind == "frame_truncated"
            or right_kind == "frame_truncated"
        ):
            width_state = "lower_bound"
        else:
            width_state = "occlusion_limited"
        records.append(OuterBoundaryObservation(
            y=int(y),
            left=left,
            right=right,
            width=right - left + 1,
            anchor_x=route_x_for_nearest_row(route_path, int(y), width / 2),
            left_kind=left_kind,
            right_kind=right_kind,
            width_state=width_state,
        ))
    return tuple(records)
