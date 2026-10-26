"""Pure coordinate conversion and blocker-to-detection attribution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from roadpush_ai.obstacle import ObstacleDetection


@dataclass(frozen=True, slots=True)
class AttributionPolicy:
    minimum_component_pixels: int = 30
    minimum_component_in_box: float = 0.30
    minimum_box_on_blocker: float = 0.10
    box_margin_pixels: int = 3
    policy_id: str = "selected_surface_blocker_overlap_v1"


@dataclass(frozen=True, slots=True)
class AttributionResult:
    status: str
    components: tuple[dict[str, Any], ...]
    matched_detection_ids: tuple[int, ...]
    analysis_boxes: dict[int, tuple[float, float, float, float]]


def native_box_to_analysis(
    box: tuple[float, float, float, float],
    *,
    original_size: tuple[int, int],
    analysis_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale an EXIF-normalized native box into Surface analysis space."""

    original_width, original_height = original_size
    analysis_width, analysis_height = analysis_size
    if min(original_width, original_height, analysis_width, analysis_height) < 1:
        raise ValueError("image sizes must be positive")
    scale_x = analysis_width / original_width
    scale_y = analysis_height / original_height
    x0, y0, x1, y1 = box
    scaled = (
        min(max(x0 * scale_x, 0.0), float(analysis_width)),
        min(max(y0 * scale_y, 0.0), float(analysis_height)),
        min(max(x1 * scale_x, 0.0), float(analysis_width)),
        min(max(y1 * scale_y, 0.0), float(analysis_height)),
    )
    return tuple(round(value, 1) for value in scaled)


def attribute_detections(
    blocker_mask: np.ndarray,
    detections: tuple[ObstacleDetection, ...],
    *,
    original_size: tuple[int, int],
    analysis_size: tuple[int, int],
    policy: AttributionPolicy = AttributionPolicy(),
) -> AttributionResult:
    """Attach boxes to connected components of the selected Surface blocker."""

    blocker = np.asarray(blocker_mask, dtype=bool)
    expected_shape = (analysis_size[1], analysis_size[0])
    if blocker.shape != expected_shape:
        raise ValueError(
            f"blocker shape {blocker.shape!r} does not match {expected_shape!r}"
        )
    analysis_boxes = {
        detection.detection_id: native_box_to_analysis(
            detection.box_native_xyxy,
            original_size=original_size,
            analysis_size=analysis_size,
        )
        for detection in detections
    }
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        blocker.astype(np.uint8),
        connectivity=8,
    )
    components: list[dict[str, Any]] = []
    used: set[int] = set()
    height, width = blocker.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < policy.minimum_component_pixels:
            continue
        component = labels == label
        covered = np.zeros_like(component)
        causes: list[dict[str, Any]] = []
        for detection in detections:
            box = analysis_boxes[detection.detection_id]
            x0 = max(0, math.floor(box[0]) - policy.box_margin_pixels)
            y0 = max(0, math.floor(box[1]) - policy.box_margin_pixels)
            x1 = min(width, math.ceil(box[2]) + policy.box_margin_pixels)
            y1 = min(height, math.ceil(box[3]) + policy.box_margin_pixels)
            box_area = max(0, x1 - x0) * max(0, y1 - y0)
            if box_area == 0:
                continue
            overlap = int(component[y0:y1, x0:x1].sum())
            component_in_box = overlap / area
            box_on_blocker = overlap / box_area
            if (
                component_in_box < policy.minimum_component_in_box
                and box_on_blocker < policy.minimum_box_on_blocker
            ):
                continue
            used.add(detection.detection_id)
            covered[y0:y1, x0:x1] |= component[y0:y1, x0:x1]
            causes.append({
                "detection_id": detection.detection_id,
                "class": detection.class_name,
                "name_ko": detection.name_ko,
                "type": detection.type,
                "action": detection.action,
                "confidence": detection.confidence,
                "overlap_px": overlap,
                "component_in_box": round(component_in_box, 3),
                "box_on_blocker": round(box_on_blocker, 3),
            })
        causes.sort(key=lambda cause: (-cause["overlap_px"], -cause["confidence"]))
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append({
            "component_id": len(components),
            "area_px": area,
            "bbox_analysis_xyxy": [
                left,
                top,
                left + component_width,
                top + component_height,
            ],
            "primary_cause": causes[0]["class"] if causes else "unclassified",
            "causes": causes,
            "unexplained_px": area - int(covered.sum()),
        })
    classified = sum(bool(component["causes"]) for component in components)
    if not components:
        status = "no_blocker"
    elif classified == len(components):
        status = "attributed"
    elif classified:
        status = "partially_attributed"
    else:
        status = "unclassified_only"
    return AttributionResult(
        status=status,
        components=tuple(components),
        matched_detection_ids=tuple(sorted(used)),
        analysis_boxes=analysis_boxes,
    )
