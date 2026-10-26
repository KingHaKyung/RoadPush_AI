"""Geometry evidence deciding whether a semantic vehicle needs partitioning."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.obstacles import (
    VehicleComponentAnalysis,
    VehicleDepthCut,
    VehicleTriggerComponent,
    VehicleTriggerEvidence,
)
from surface_passability.obstacles.footprints import estimate_component_footprint
from surface_passability.obstacles.ownership import (
    measure_projected_ground_support,
    measure_source_contact_support,
)
from surface_passability.obstacles.taxonomy import CAR_CLASS_ID
from surface_passability.policies.obstacles import (
    VEHICLE_BOTTOM_TOLERANCE_HEIGHT_FRACTION,
    VEHICLE_BOTTOM_TOLERANCE_MINIMUM_PX,
    VEHICLE_DEPTH_CUT_EDGE_MINIMUM_PX,
    VEHICLE_DEPTH_CUT_START_WIDTH_FRACTION,
    VEHICLE_DEPTH_CUT_STOP_WIDTH_FRACTION,
    VEHICLE_DEPTH_DIFFERENCE_HEIGHT_FRACTION,
    VEHICLE_DEPTH_DIFFERENCE_MINIMUM_PX,
    VEHICLE_DEPTH_SCORE_MINIMUM_DENOMINATOR,
    VEHICLE_PROFILE_BASE_WINDOW_PX,
    VEHICLE_PROFILE_MINIMUM_SMOOTHING_WINDOW_PX,
    VEHICLE_PROFILE_WINDOW_WIDTH_FRACTION,
    VEHICLE_SPLIT_MINIMUM_CHILD_FRACTION,
    VEHICLE_TRIGGER_COMPONENT_MINIMUM_IMAGE_FRACTION,
    VEHICLE_TRIGGER_COMPONENT_MINIMUM_PIXELS,
    OWNERSHIP_MINIMUM_GROUND_SUPPORT_FRACTION,
    OWNERSHIP_MINIMUM_GROUND_SUPPORT_PIXELS,
    OWNERSHIP_MINIMUM_SOURCE_CONTACT_COLUMNS,
    OWNERSHIP_MINIMUM_SOURCE_CONTACT_FRACTION,
)


def smooth_bottom_profile(values: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth a vehicle bottom profile with the frozen odd window."""

    window = max(VEHICLE_PROFILE_MINIMUM_SMOOTHING_WINDOW_PX, int(window) | 1)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.asarray([
        np.median(padded[index:index + window])
        for index in range(len(values))
    ])


def measure_vehicle_bottom_profile(
    component_mask: np.ndarray,
    left_x: int,
    width: int,
) -> np.ndarray:
    """Measure and interpolate the lowest component pixel in each bbox column."""

    profile = np.full(width, np.nan, np.float32)
    for local_x in range(width):
        rows_y = np.flatnonzero(component_mask[:, left_x + local_x])
        if len(rows_y):
            profile[local_x] = float(rows_y.max())
    valid = np.isfinite(profile)
    if not valid.any():
        return np.zeros(width, np.float32)
    positions = np.arange(width)
    profile = np.interp(
        positions,
        positions[valid],
        profile[valid],
    ).astype(np.float32)
    return smooth_bottom_profile(
        profile,
        max(
            VEHICLE_PROFILE_BASE_WINDOW_PX,
            round(width * VEHICLE_PROFILE_WINDOW_WIDTH_FRACTION),
        ),
    )


def select_vehicle_depth_cut(profile: np.ndarray) -> VehicleDepthCut | None:
    """Select the strongest robust depth step in a vehicle bottom profile."""

    width = len(profile)
    start = max(
        VEHICLE_DEPTH_CUT_EDGE_MINIMUM_PX,
        round(width * VEHICLE_DEPTH_CUT_START_WIDTH_FRACTION),
    )
    stop = min(
        width - VEHICLE_DEPTH_CUT_EDGE_MINIMUM_PX,
        round(width * VEHICLE_DEPTH_CUT_STOP_WIDTH_FRACTION),
    )
    if stop <= start:
        return None
    best: VehicleDepthCut | None = None
    for cut in range(start, stop):
        left, right = profile[:cut], profile[cut:]
        left_median = float(np.median(left))
        right_median = float(np.median(right))
        left_mad = float(np.median(np.abs(left - left_median)))
        right_mad = float(np.median(np.abs(right - right_median)))
        difference = abs(left_median - right_median)
        score = difference / max(
            VEHICLE_DEPTH_SCORE_MINIMUM_DENOMINATOR,
            1.0 + left_mad + right_mad,
        )
        candidate = VehicleDepthCut(
            cut,
            left_median,
            right_median,
            difference,
            left_mad,
            right_mad,
            score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def analyze_vehicle_component(
    component_mask: np.ndarray,
    bbox: list[int],
    image_height: int,
) -> VehicleComponentAnalysis:
    """Keep or split one semantic vehicle using bottom-depth evidence only."""

    left_x, top_y, width, height = bbox
    profile = measure_vehicle_bottom_profile(component_mask, left_x, width)
    candidate = select_vehicle_depth_cut(profile)
    bottom_gap = max(0, image_height - (top_y + height))
    bottom_tolerance = max(
        VEHICLE_BOTTOM_TOLERANCE_MINIMUM_PX,
        round(image_height * VEHICLE_BOTTOM_TOLERANCE_HEIGHT_FRACTION),
    )
    touches_bottom = bottom_gap <= bottom_tolerance
    if candidate is None:
        return VehicleComponentAnalysis(
            (component_mask,),
            bbox,
            int(component_mask.sum()),
            touches_bottom,
            bottom_gap,
            bottom_tolerance,
            "keep_parent",
            "no_reliable_multi_depth_evidence",
            profile,
        )

    threshold = max(
        VEHICLE_DEPTH_DIFFERENCE_MINIMUM_PX,
        image_height * VEHICLE_DEPTH_DIFFERENCE_HEIGHT_FRACTION,
    )
    split_x = left_x + int(candidate.cut)
    left = component_mask.copy()
    left[:, split_x:] = False
    right = component_mask.copy()
    right[:, :split_x] = False
    parent_pixels = max(1, int(component_mask.sum()))
    child_fractions = (
        float(left.sum()) / parent_pixels,
        float(right.sum()) / parent_pixels,
    )
    strong_depth_step = candidate.depth_difference_px >= threshold
    meaningful_children = (
        min(child_fractions) >= VEHICLE_SPLIT_MINIMUM_CHILD_FRACTION
    )
    if touches_bottom and strong_depth_step and meaningful_children:
        children = (left, right)
        decision = "split_candidate"
        reason = "bottom_truncated_component_with_two_strong_depth_levels"
    else:
        children = (component_mask,)
        decision = "keep_parent"
        if not touches_bottom:
            reason = "not_bottom_truncated"
        elif not strong_depth_step:
            reason = "depth_step_below_threshold"
        else:
            reason = "child_too_small"
    return VehicleComponentAnalysis(
        children,
        bbox,
        int(component_mask.sum()),
        touches_bottom,
        bottom_gap,
        bottom_tolerance,
        decision,
        reason,
        profile,
        candidate,
        split_x,
        threshold,
        child_fractions,
    )


def analyze_vehicle_trigger_components(
    class_map: np.ndarray,
    walkway_mask: np.ndarray,
) -> VehicleTriggerEvidence:
    """Decide which walkway-owned semantic car components require YOLO."""

    binary = (class_map == CAR_CLASS_ID).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        8,
    )
    records: list[VehicleTriggerComponent] = []
    minimum_area = max(
        VEHICLE_TRIGGER_COMPONENT_MINIMUM_PIXELS,
        round(
            class_map.size
            * VEHICLE_TRIGGER_COMPONENT_MINIMUM_IMAGE_FRACTION
        ),
    )
    for component_id in range(1, count):
        left_x, top_y, width, height, area = map(int, stats[component_id])
        if area < minimum_area:
            continue
        component = component_labels == component_id
        footprint = estimate_component_footprint(component, CAR_CLASS_ID)
        if footprint is None:
            continue
        projected = measure_projected_ground_support(
            component,
            footprint.mask,
            walkway_mask,
        )
        minimum_support = max(
            OWNERSHIP_MINIMUM_GROUND_SUPPORT_PIXELS,
            round(
                footprint.mask.sum()
                * OWNERSHIP_MINIMUM_GROUND_SUPPORT_FRACTION
            ),
        )
        contact = measure_source_contact_support(component, walkway_mask)
        source_contact_ok = contact.bottom_truncated or (
            contact.supported_columns
            >= OWNERSHIP_MINIMUM_SOURCE_CONTACT_COLUMNS
            and contact.support_ratio
            >= OWNERSHIP_MINIMUM_SOURCE_CONTACT_FRACTION
        )
        walkway_owned = (
            projected.support_pixels >= minimum_support
            and source_contact_ok
        )
        geometry = analyze_vehicle_component(
            component,
            [left_x, top_y, width, height],
            class_map.shape[0],
        )
        anomaly = geometry.decision == "split_candidate"
        trigger = walkway_owned and anomaly
        decision = (
            "skip_off_walkway"
            if not walkway_owned
            else ("call_yolo" if trigger else "keep_mapillary")
        )
        records.append(VehicleTriggerComponent(
            component_id=component_id,
            source_bbox=(left_x, top_y, width, height),
            source_pixels=int(area),
            walkway_owned=bool(walkway_owned),
            ground_support_pixels=int(projected.support_pixels),
            ground_support_fraction=float(projected.support_fraction),
            minimum_ground_support_pixels=int(minimum_support),
            source_contact_supported_columns=int(contact.supported_columns),
            source_contact_total_columns=int(contact.total_columns),
            source_contact_support_fraction=float(contact.support_ratio),
            source_bottom_truncated=bool(contact.bottom_truncated),
            geometry_anomaly=bool(anomaly),
            trigger_yolo=bool(trigger),
            decision=decision,
            geometry=geometry,
            component_mask=component,
            candidate_mask=footprint.mask,
            effective_mask=projected.effective_mask,
        ))
    return VehicleTriggerEvidence(tuple(records))
