"""Class-aware ground-position obstacle clearance envelopes."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.obstacles import (
    ClearanceStrategy,
    ObstacleFootprint,
    ObstacleFootprintKind,
)
from surface_passability.policies.obstacles import (
    BODY_PATCH_HEIGHT_FRACTION,
    BODY_PATCH_IMAGE_HEIGHT_FRACTION,
    BODY_PATCH_MINIMUM_DEPTH_PX,
    BODY_WIDTH_PERCENTILE,
    BODY_WIDTH_RATIO,
    CYCLE_CLASS_IDS,
    LOWER_SUPPORT_BAND_HEIGHT_FRACTION,
    LOWER_SUPPORT_BAND_MINIMUM_DEPTH_PX,
    LOWER_SUPPORT_CLASS_IDS,
    LOWER_SUPPORT_PATCH_HEIGHT_FRACTION,
    LOWER_SUPPORT_PATCH_IMAGE_HEIGHT_FRACTION,
    LOWER_SUPPORT_PATCH_MINIMUM_DEPTH_PX,
    LOWER_SUPPORT_WIDTH_PERCENTILE,
    LOWER_SUPPORT_WIDTH_RATIO,
    OBSTACLE_FOOTPRINT_HORIZONTAL_IMAGE_FRACTION,
    OBSTACLE_FOOTPRINT_HORIZONTAL_MINIMUM_MARGIN_PX,
    OBSTACLE_FOOTPRINT_HORIZONTAL_WIDTH_FRACTION,
    OBSTACLE_FOOTPRINT_LEFT_PERCENTILE,
    OBSTACLE_FOOTPRINT_MINIMUM_IMAGE_FRACTION,
    OBSTACLE_FOOTPRINT_MINIMUM_PIXELS,
    OBSTACLE_FOOTPRINT_RIGHT_PERCENTILE,
    OBSTACLE_FOOTPRINT_VERTICAL_IMAGE_FRACTION,
    OBSTACLE_FOOTPRINT_VERTICAL_MINIMUM_MARGIN_PX,
    POLE_CLASS_ID,
    WIDE_BODY_CLASS_IDS,
)


def component_row_spans(
    component_mask: np.ndarray,
    top: int,
    bottom: int,
) -> list[tuple[int, int, int, int]]:
    """Collect inclusive horizontal component spans for a vertical range."""

    rows = []
    for y in range(top, bottom + 1):
        columns = np.flatnonzero(component_mask[y])
        if len(columns):
            rows.append((
                y,
                int(columns[0]),
                int(columns[-1]),
                int(columns[-1] - columns[0] + 1),
            ))
    return rows


def robust_component_span(
    rows: list[tuple[int, int, int, int]],
    percentile: float,
    ratio: float,
) -> tuple[int, int, float, int]:
    """Select repeated near-maximum spans and return robust horizontal bounds."""

    widths = np.asarray([row[3] for row in rows], np.float32)
    target = max(1.0, float(np.percentile(widths, percentile)))
    selected = [row for row in rows if row[3] >= max(1.0, target * ratio)]
    if not selected:
        selected = [max(rows, key=lambda row: row[3])]
    left = int(round(np.percentile(
        [row[1] for row in selected],
        OBSTACLE_FOOTPRINT_LEFT_PERCENTILE,
    )))
    right = int(round(np.percentile(
        [row[2] for row in selected],
        OBSTACLE_FOOTPRINT_RIGHT_PERCENTILE,
    )))
    return left, right, target, len(selected)


def estimate_component_footprint(
    component_mask: np.ndarray,
    class_id: int,
) -> ObstacleFootprint | None:
    """Estimate a class-aware ground-position clearance envelope."""

    image_height, image_width = component_mask.shape
    rows_y, columns_x = np.nonzero(component_mask)
    minimum_pixels = max(
        OBSTACLE_FOOTPRINT_MINIMUM_PIXELS,
        round(component_mask.size * OBSTACLE_FOOTPRINT_MINIMUM_IMAGE_FRACTION),
    )
    if len(columns_x) < minimum_pixels:
        return None

    x0, x1 = int(columns_x.min()), int(columns_x.max())
    y0, y1 = int(rows_y.min()), int(rows_y.max())
    object_height = y1 - y0 + 1
    object_width = x1 - x0 + 1

    strategy: ClearanceStrategy
    if class_id in LOWER_SUPPORT_CLASS_IDS:
        band_depth = max(
            LOWER_SUPPORT_BAND_MINIMUM_DEPTH_PX,
            round(object_height * LOWER_SUPPORT_BAND_HEIGHT_FRACTION),
        )
        band_top = max(y0, y1 - band_depth + 1)
        rows = component_row_spans(component_mask, band_top, y1)
        left, right, stable_width, selected_rows = robust_component_span(
            rows,
            LOWER_SUPPORT_WIDTH_PERCENTILE,
            LOWER_SUPPORT_WIDTH_RATIO,
        )
        patch_depth = max(
            LOWER_SUPPORT_PATCH_MINIMUM_DEPTH_PX,
            min(
                round(object_height * LOWER_SUPPORT_PATCH_HEIGHT_FRACTION),
                round(image_height * LOWER_SUPPORT_PATCH_IMAGE_HEIGHT_FRACTION),
            ),
        )
        strategy = "lower_support_or_pedestal"
    else:
        rows = component_row_spans(component_mask, y0, y1)
        left, right, stable_width, selected_rows = robust_component_span(
            rows,
            BODY_WIDTH_PERCENTILE,
            BODY_WIDTH_RATIO,
        )
        patch_depth = max(
            BODY_PATCH_MINIMUM_DEPTH_PX,
            min(
                round(object_height * BODY_PATCH_HEIGHT_FRACTION),
                round(image_height * BODY_PATCH_IMAGE_HEIGHT_FRACTION),
            ),
        )
        band_top = y0
        strategy = (
            "wide_body_projected_to_ground"
            if class_id in WIDE_BODY_CLASS_IDS
            else "generic_body_projected_to_ground"
        )

    margin_x = max(
        OBSTACLE_FOOTPRINT_HORIZONTAL_MINIMUM_MARGIN_PX,
        round(image_width * OBSTACLE_FOOTPRINT_HORIZONTAL_IMAGE_FRACTION),
        round(
            (right - left + 1)
            * OBSTACLE_FOOTPRINT_HORIZONTAL_WIDTH_FRACTION
        ),
    )
    margin_y = max(
        OBSTACLE_FOOTPRINT_VERTICAL_MINIMUM_MARGIN_PX,
        round(image_height * OBSTACLE_FOOTPRINT_VERTICAL_IMAGE_FRACTION),
    )
    left = max(0, left - margin_x)
    right = min(image_width - 1, right + margin_x)
    top = max(0, y1 - patch_depth + 1 - margin_y)
    bottom = min(image_height - 1, y1 + margin_y)

    footprint = np.zeros_like(component_mask, dtype=np.uint8)
    cv2.rectangle(
        footprint,
        (left, top),
        (right, bottom),
        1,
        cv2.FILLED,
    )
    kind: ObstacleFootprintKind = (
        "pole_or_bollard"
        if class_id == POLE_CLASS_ID
        else (
            "cycle_like"
            if class_id in CYCLE_CLASS_IDS
            else "generic_obstacle"
        )
    )
    footprint_mask = footprint.astype(bool)
    return ObstacleFootprint(
        mask=footprint_mask,
        class_id=int(class_id),
        source_pixels=int(component_mask.sum()),
        source_bbox=(x0, y0, object_width, object_height),
        lower_band_top=int(band_top),
        stable_width_px=float(stable_width),
        qualifying_rows=int(selected_rows),
        selected_rows=int(selected_rows),
        footprint_bbox=(left, top, right - left + 1, bottom - top + 1),
        footprint_pixels=int(footprint.sum()),
        kind=kind,
        clearance_strategy=strategy,
    )
