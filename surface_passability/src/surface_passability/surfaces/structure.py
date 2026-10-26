"""Structural surface completion independent of route and obstacle ownership."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.surfaces import SurfaceStructureResult
from surface_passability.geometry.intervals import true_intervals
from surface_passability.policies.surfaces import (
    SURFACE_STRUCTURE_CLOSING_IMAGE_FRACTION,
    SURFACE_STRUCTURE_ENCLOSED_HOLE_IMAGE_FRACTION,
    SURFACE_STRUCTURE_MINIMUM_CLOSING_KERNEL_PX,
    SURFACE_STRUCTURE_MINIMUM_ENCLOSED_HOLE_PIXELS,
    SURFACE_STRUCTURE_MINIMUM_OBSTACLE_BRIDGE_PX,
    SURFACE_STRUCTURE_OBSTACLE_BRIDGE_WIDTH_FRACTION,
    SURFACE_STRUCTURE_OBSTACLE_STABILIZATION_KERNEL_PX,
)


def build_surface_structure(surface_mask: np.ndarray) -> SurfaceStructureResult:
    """Close local gaps and fill bounded holes without mutating the source mask."""

    height, width = surface_mask.shape
    kernel_size = max(
        SURFACE_STRUCTURE_MINIMUM_CLOSING_KERNEL_PX,
        round(
            min(height, width)
            * SURFACE_STRUCTURE_CLOSING_IMAGE_FRACTION
        ),
    ) | 1
    closed = cv2.morphologyEx(
        surface_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    ).astype(bool)
    inverse = (~closed).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    structural = closed.copy()
    maximum_enclosed_hole_pixels = max(
        SURFACE_STRUCTURE_MINIMUM_ENCLOSED_HOLE_PIXELS,
        round(
            height
            * width
            * SURFACE_STRUCTURE_ENCLOSED_HOLE_IMAGE_FRACTION
        ),
    )
    for component_id in range(1, count):
        x, y, component_width, component_height, area = map(
            int,
            stats[component_id],
        )
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_border and area <= maximum_enclosed_hole_pixels:
            structural[labels == component_id] = True

    return SurfaceStructureResult(
        structural_mask=structural,
        addition_mask=structural & ~surface_mask,
    )


def build_obstacle_transparent_structure(
    surface_mask: np.ndarray,
    blocker_mask: np.ndarray,
) -> SurfaceStructureResult:
    """Restore short blocker spans bracketed by structural surface on both sides."""

    base = build_surface_structure(surface_mask)
    structural = base.structural_mask.copy()
    additions = base.addition_mask.copy()
    height, width = surface_mask.shape
    maximum_bridge_pixels = max(
        SURFACE_STRUCTURE_MINIMUM_OBSTACLE_BRIDGE_PX,
        round(
            width * SURFACE_STRUCTURE_OBSTACLE_BRIDGE_WIDTH_FRACTION
        ),
    )
    blocked_u8 = blocker_mask.astype(np.uint8)
    for y in range(height):
        for left, right in true_intervals(blocked_u8[y] > 0):
            if right - left + 1 > maximum_bridge_pixels:
                continue
            left_window = structural[
                y,
                max(0, left - maximum_bridge_pixels) : left,
            ]
            right_window = structural[
                y,
                right + 1 : min(width, right + 1 + maximum_bridge_pixels),
            ]
            if left_window.any() and right_window.any():
                structural[y, left : right + 1] = True
                additions[y, left : right + 1] = True

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            SURFACE_STRUCTURE_OBSTACLE_STABILIZATION_KERNEL_PX,
            SURFACE_STRUCTURE_OBSTACLE_STABILIZATION_KERNEL_PX,
        ),
    )
    structural = cv2.morphologyEx(
        structural.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    ).astype(bool)
    additions |= structural & ~surface_mask
    return SurfaceStructureResult(
        structural_mask=structural,
        addition_mask=additions,
    )

