from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.surface.context import SurfaceContext, SurfaceProfile


ROAD_ID = 13
SERVICE_LANE_ID = 14
PARKING_ID = 10
BIKE_LANE_ID = 7
CURB_ID = 2
CURB_CUT_ID = 9
ROAD_REPAIR_IDS = (8, 23, 24, 36, 41)

# Phase-3 probe constants. They are fixed before the 40-image review and must
# not be tuned during review.
ROAD_REPAIR_RADIUS_FRACTION = 1 / 420
ROAD_REPAIR_MIN_RING_FRACTION = 0.02
ROAD_REPAIR_MIN_SUPPORT_PIXELS = 3
ROAD_UNION_SEAM_RADIUS_PX = 2


ROAD_PROFILE = SurfaceProfile(
    name="road",
    source_class_ids=(ROAD_ID, SERVICE_LANE_ID, PARKING_ID),
    repair_class_ids=ROAD_REPAIR_IDS,
    forbidden_class_ids=(CURB_ID, CURB_CUT_ID),
    side_pruning_enabled=True,
)


@dataclass
class RoadContext:
    surface: SurfaceContext
    raw_road: np.ndarray
    raw_service_lane: np.ndarray
    raw_parking: np.ndarray
    repaired_road: np.ndarray
    repaired_service_lane: np.ndarray
    repaired_parking: np.ndarray
    accepted_repair: np.ndarray
    trusted_sidewalk_guard: np.ndarray
    road_forbidden: np.ndarray
    surface_exclusion: np.ndarray
    bike_lane: np.ndarray
    repair_records: list[dict]
    policy: dict


def _repair_class_local(
    labels: np.ndarray,
    source: np.ndarray,
    source_class_id: int,
    forbidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    repaired = source.copy()
    additions = np.zeros_like(source)
    records = []
    height, width = labels.shape
    radius = max(3, round(min(height, width) * ROAD_REPAIR_RADIUS_FRACTION))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    for repair_class_id in ROAD_REPAIR_IDS:
        binary = (labels == repair_class_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for component_index, contour in enumerate(contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            pad = radius + 1
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
            local_labels = labels[y0:y1, x0:x1]
            component = np.zeros(local_labels.shape, np.uint8)
            shifted = contour.copy()
            shifted[:, 0, 0] -= x0
            shifted[:, 0, 1] -= y0
            cv2.drawContours(component, [shifted], -1, 1, cv2.FILLED)
            component &= (local_labels == repair_class_id).astype(np.uint8)
            pixels = int(component.sum())
            if not pixels:
                continue
            ring = (
                (cv2.dilate(component, kernel) > 0)
                & (component == 0)
                & (local_labels != repair_class_id)
            )
            ring_pixels = int(ring.sum())
            support_pixels = int(((local_labels == source_class_id) & ring).sum())
            support_fraction = support_pixels / max(1, ring_pixels)
            component_fraction = pixels / labels.size
            touches_frame = x == 0 or y == 0 or x + w == width or y + h == height
            geometry_ok = repair_class_id != 36 or (
                component_fraction <= 0.004 and not touches_frame
            )
            local_forbidden = forbidden[y0:y1, x0:x1]
            accepted_mask = component.astype(bool) & ~local_forbidden
            accepted = bool(
                accepted_mask.any()
                and support_pixels >= ROAD_REPAIR_MIN_SUPPORT_PIXELS
                and support_fraction >= ROAD_REPAIR_MIN_RING_FRACTION
                and geometry_ok
            )
            if accepted:
                repaired[y0:y1, x0:x1][accepted_mask] = True
                additions[y0:y1, x0:x1][accepted_mask] = True
            records.append(
                {
                    "source_class_id": source_class_id,
                    "repair_class_id": repair_class_id,
                    "component_index": component_index,
                    "pixels": pixels,
                    "accepted_pixels": int(accepted_mask.sum()) if accepted else 0,
                    "ring_pixels": ring_pixels,
                    "source_support_pixels": support_pixels,
                    "source_support_fraction": float(support_fraction),
                    "component_fraction": float(component_fraction),
                    "touches_frame": touches_frame,
                    "geometry_ok": geometry_ok,
                    "forbidden_overlap_pixels": int(
                        (component.astype(bool) & local_forbidden).sum()
                    ),
                    "accepted": accepted,
                }
            )
    repaired &= ~forbidden
    additions &= ~forbidden
    return repaired, additions, records


def build_road_context(labels: np.ndarray, sidewalk: SurfaceContext) -> RoadContext:
    trusted_sidewalk_guard = sidewalk.repaired_mask & ~sidewalk.forbidden_mask
    road_forbidden = (
        trusted_sidewalk_guard | (labels == CURB_ID) | (labels == CURB_CUT_ID)
    )
    bike_lane = labels == BIKE_LANE_ID
    surface_exclusion = road_forbidden | bike_lane

    raw_road = (labels == ROAD_ID) & ~road_forbidden
    raw_service = (labels == SERVICE_LANE_ID) & ~road_forbidden
    raw_parking = (labels == PARKING_ID) & ~road_forbidden
    repaired_road, add_road, records_road = _repair_class_local(
        labels, raw_road, ROAD_ID, road_forbidden
    )
    repaired_service, add_service, records_service = _repair_class_local(
        labels, raw_service, SERVICE_LANE_ID, road_forbidden
    )
    repaired_parking, add_parking, records_parking = _repair_class_local(
        labels, raw_parking, PARKING_ID, road_forbidden
    )

    repaired_union = repaired_road | repaired_service | repaired_parking
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2
    )
    repaired_union = cv2.morphologyEx(
        repaired_union.astype(np.uint8), cv2.MORPH_CLOSE, seam_kernel
    ).astype(bool)
    repaired_union &= ~surface_exclusion
    accepted_repair = (add_road | add_service | add_parking) & ~surface_exclusion

    structure, structure_additions = legacy.base.structure_copy(repaired_union)
    structure &= ~surface_exclusion
    structure_additions &= ~surface_exclusion
    route, route_path = legacy.outer.route_path(repaired_union)
    observed = legacy.outer.selected_component(structure, route_path)
    if not observed.any():
        observed = structure.copy()
    source_component = legacy.outer.selected_component(repaired_union, route_path)
    if not source_component.any():
        source_component = repaired_union.copy()

    surface = SurfaceContext(
        profile=ROAD_PROFILE,
        source_mask=source_component,
        repaired_mask=repaired_union,
        structure_copy=structure,
        forbidden_mask=surface_exclusion,
        observed_component=observed,
        route=route,
        route_path=route_path,
        structure_copy_additions=structure_additions,
    )
    return RoadContext(
        surface=surface,
        raw_road=raw_road,
        raw_service_lane=raw_service,
        raw_parking=raw_parking,
        repaired_road=repaired_road,
        repaired_service_lane=repaired_service,
        repaired_parking=repaired_parking,
        accepted_repair=accepted_repair,
        trusted_sidewalk_guard=trusted_sidewalk_guard,
        road_forbidden=road_forbidden,
        surface_exclusion=surface_exclusion,
        bike_lane=bike_lane,
        repair_records=records_road + records_service + records_parking,
        policy={
            "source_class_ids": list(ROAD_PROFILE.source_class_ids),
            "repair_class_ids": list(ROAD_REPAIR_IDS),
            "repair_radius_fraction": ROAD_REPAIR_RADIUS_FRACTION,
            "minimum_ring_fraction": ROAD_REPAIR_MIN_RING_FRACTION,
            "minimum_support_pixels": ROAD_REPAIR_MIN_SUPPORT_PIXELS,
            "union_seam_radius_px": ROAD_UNION_SEAM_RADIUS_PX,
            "forbidden_class_ids": list(ROAD_PROFILE.forbidden_class_ids),
            "trusted_sidewalk_uses_all_repaired_components": True,
            "sidewalk_outer_row_span_used_as_guard": False,
            "bike_lane_included": False,
        },
    )
