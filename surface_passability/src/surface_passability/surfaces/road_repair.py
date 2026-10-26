"""Class-local semantic repair for one Road source class."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.road import RoadRepairComponentEvidence
from surface_passability.policies.road import (
    ROAD_REPAIR_CATCH_BASIN_CLASS_ID,
    ROAD_REPAIR_CATCH_BASIN_MAXIMUM_IMAGE_FRACTION,
    ROAD_REPAIR_CLASS_IDS,
    ROAD_REPAIR_MINIMUM_RADIUS_PX,
    ROAD_REPAIR_MINIMUM_RING_FRACTION,
    ROAD_REPAIR_MINIMUM_SUPPORT_PIXELS,
    ROAD_REPAIR_RADIUS_FRACTION,
)


def repair_road_class_locally(
    class_map: np.ndarray,
    source_mask: np.ndarray,
    source_class_id: int,
    forbidden_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[RoadRepairComponentEvidence]]:
    """Inherit locally supported repair-class components for one Road class."""

    repaired_mask = source_mask.copy()
    addition_mask = np.zeros_like(source_mask)
    components: list[RoadRepairComponentEvidence] = []
    height, width = class_map.shape
    radius = max(
        ROAD_REPAIR_MINIMUM_RADIUS_PX,
        round(min(height, width) * ROAD_REPAIR_RADIUS_FRACTION),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    for repair_class_id in ROAD_REPAIR_CLASS_IDS:
        binary = (class_map == repair_class_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for component_index, contour in enumerate(contours, 1):
            x, y, component_width, component_height = cv2.boundingRect(contour)
            pad = radius + 1
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(width, x + component_width + pad)
            y1 = min(height, y + component_height + pad)
            local_class_map = class_map[y0:y1, x0:x1]
            component = np.zeros(local_class_map.shape, np.uint8)
            shifted_contour = contour.copy()
            shifted_contour[:, 0, 0] -= x0
            shifted_contour[:, 0, 1] -= y0
            cv2.drawContours(
                component,
                [shifted_contour],
                -1,
                1,
                cv2.FILLED,
            )
            component &= (local_class_map == repair_class_id).astype(np.uint8)
            pixels = int(component.sum())
            if not pixels:
                continue
            ring = (
                (cv2.dilate(component, kernel) > 0)
                & (component == 0)
                & (local_class_map != repair_class_id)
            )
            ring_pixels = int(ring.sum())
            support_pixels = int(
                ((local_class_map == source_class_id) & ring).sum()
            )
            support_fraction = support_pixels / max(1, ring_pixels)
            component_fraction = pixels / class_map.size
            touches_frame = (
                x == 0
                or y == 0
                or x + component_width == width
                or y + component_height == height
            )
            geometry_ok = (
                repair_class_id != ROAD_REPAIR_CATCH_BASIN_CLASS_ID
                or (
                    component_fraction
                    <= ROAD_REPAIR_CATCH_BASIN_MAXIMUM_IMAGE_FRACTION
                    and not touches_frame
                )
            )
            local_forbidden = forbidden_mask[y0:y1, x0:x1]
            accepted_mask = component.astype(bool) & ~local_forbidden
            accepted = bool(
                accepted_mask.any()
                and support_pixels >= ROAD_REPAIR_MINIMUM_SUPPORT_PIXELS
                and support_fraction >= ROAD_REPAIR_MINIMUM_RING_FRACTION
                and geometry_ok
            )
            if accepted:
                repaired_mask[y0:y1, x0:x1][accepted_mask] = True
                addition_mask[y0:y1, x0:x1][accepted_mask] = True
            components.append(
                RoadRepairComponentEvidence(
                    source_class_id=source_class_id,
                    repair_class_id=repair_class_id,
                    component_index=component_index,
                    pixels=pixels,
                    accepted_pixels=(
                        int(accepted_mask.sum()) if accepted else 0
                    ),
                    ring_pixels=ring_pixels,
                    source_support_pixels=support_pixels,
                    source_support_fraction=float(support_fraction),
                    component_fraction=float(component_fraction),
                    touches_frame=touches_frame,
                    geometry_ok=geometry_ok,
                    forbidden_overlap_pixels=int(
                        (component.astype(bool) & local_forbidden).sum()
                    ),
                    accepted=accepted,
                )
            )
    repaired_mask &= ~forbidden_mask
    addition_mask &= ~forbidden_mask
    return repaired_mask, addition_mask, components
