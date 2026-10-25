from __future__ import annotations

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.surface.context import SurfacePipelineContext
from roadpush_ai.contracts.test4_frozen import (
    CONTACT_RADIUS_IMAGE_FRACTION,
    MINIMUM_COMPONENT_PIXELS,
    MINIMUM_COMPONENT_ROWS,
    MINIMUM_CONTACT_PIXELS_EACH_SIDE,
    MINIMUM_FORWARD_PROJECTION_PX,
    MINIMUM_ROUTE_ALIGNMENT_COSINE,
)


def connected_components(mask: np.ndarray, min_pixels: int = 1) -> list[dict]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    rows = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        rows.append(
            {
                "index": int(index),
                "mask": labels == index,
                "area": area,
                "bbox": [int(value) for value in stats[index, :4]],
            }
        )
    return rows


def main_component(components: list[dict], pipeline: SurfacePipelineContext) -> dict:
    if not components:
        raise RuntimeError("No structure-copy components")
    start = pipeline.route.get("start")
    ranked = []
    for component in components:
        overlap = int((component["mask"] & pipeline.observed_component).sum())
        contains_start = bool(start is not None and component["mask"][start])
        ranked.append((contains_start, overlap, component["area"], component))
    return max(ranked, key=lambda value: value[:3])[3]


def forward_alignment(
    pipeline: SurfacePipelineContext,
    selected_contact: np.ndarray,
    satellite_contact: np.ndarray,
) -> dict:
    route = pipeline.route_path
    if len(route) < 2:
        return {"available": False, "accepted": False}
    start_y, start_x = route[0]
    end_y, end_x = route[-1]
    route_vector = np.asarray([end_x - start_x, end_y - start_y], np.float64)
    selected_yx = np.argwhere(selected_contact).mean(axis=0)
    satellite_yx = np.argwhere(satellite_contact).mean(axis=0)
    contact_vector = np.asarray(
        [satellite_yx[1] - selected_yx[1], satellite_yx[0] - selected_yx[0]],
        np.float64,
    )
    route_norm = float(np.linalg.norm(route_vector))
    contact_norm = float(np.linalg.norm(contact_vector))
    if route_norm < 1e-6 or contact_norm < 1e-6:
        return {
            "available": True,
            "accepted": False,
            "forward_projection_px": 0.0,
            "cosine": None,
        }
    unit_route = route_vector / route_norm
    forward = float(np.dot(contact_vector, unit_route))
    cosine = float(forward / contact_norm)
    return {
        "available": True,
        "accepted": bool(
            forward >= MINIMUM_FORWARD_PROJECTION_PX
            and cosine >= MINIMUM_ROUTE_ALIGNMENT_COSINE
        ),
        "forward_projection_px": forward,
        "cosine": cosine,
        "selected_contact_center_yx": [float(selected_yx[0]), float(selected_yx[1])],
        "satellite_contact_center_yx": [
            float(satellite_yx[0]),
            float(satellite_yx[1]),
        ],
    }


def _crosses_forbidden(
    first: np.ndarray,
    second: np.ndarray,
    forbidden: np.ndarray,
    thickness: int,
) -> bool:
    first_center = np.argwhere(first).mean(axis=0)
    second_center = np.argwhere(second).mean(axis=0)
    line = np.zeros_like(forbidden, np.uint8)
    cv2.line(
        line,
        (round(first_center[1]), round(first_center[0])),
        (round(second_center[1]), round(second_center[0])),
        1,
        max(1, thickness),
    )
    return bool((line.astype(bool) & forbidden).any())


def select_obstacle_connected_components(
    pipeline: SurfacePipelineContext,
    obstacle_sources: list[dict],
    *,
    authorization_forbidden_mask: np.ndarray | None = None,
) -> dict:
    """Preserve frozen Test4 selection without importing its probe runner."""

    candidates = pipeline.structure_copy & ~pipeline.curb_suspect
    components = connected_components(candidates)
    radius = max(3, round(min(candidates.shape) * CONTACT_RADIUS_IMAGE_FRACTION))
    if not components:
        empty = np.zeros_like(candidates)
        result = {
            "candidate": candidates,
            "main": empty.copy(),
            "selected_sources": empty.copy(),
            "accepted_satellites": empty.copy(),
            "rejected_sources": empty.copy(),
            "span": empty.copy(),
            "components": [],
            "main_component_index": None,
            "accepted_component_indices": [],
            "evidence": [],
            "obstacle_component_count": 0,
            "contact_radius_px": radius,
            "empty_candidate": True,
        }
        if authorization_forbidden_mask is not None:
            result["forbidden_crossing_rejections"] = 0
        return result
    main = main_component(components, pipeline)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    obstacle_dilated = [
        cv2.dilate(row["mask"].astype(np.uint8), kernel).astype(bool)
        for row in obstacle_sources
    ]
    selected = main["mask"].copy()
    accepted = {main["index"]}
    evidence = []
    crossing_rejections = 0
    changed = True
    while changed:
        changed = False
        for component in components:
            if component["index"] in accepted:
                continue
            ys = np.nonzero(component["mask"])[0]
            if component["area"] < MINIMUM_COMPONENT_PIXELS or len(
                np.unique(ys)
            ) < MINIMUM_COMPONENT_ROWS:
                continue
            matches = []
            for obstacle, dilated in zip(obstacle_sources, obstacle_dilated):
                selected_contact = dilated & selected
                satellite_contact = dilated & component["mask"]
                selected_pixels = int(selected_contact.sum())
                satellite_pixels = int(satellite_contact.sum())
                if (
                    selected_pixels < MINIMUM_CONTACT_PIXELS_EACH_SIDE
                    or satellite_pixels < MINIMUM_CONTACT_PIXELS_EACH_SIDE
                ):
                    continue
                alignment = forward_alignment(
                    pipeline, selected_contact, satellite_contact
                )
                if not alignment["accepted"]:
                    continue
                if authorization_forbidden_mask is not None and _crosses_forbidden(
                    selected_contact,
                    satellite_contact,
                    authorization_forbidden_mask,
                    radius,
                ):
                    crossing_rejections += 1
                    continue
                matches.append(
                    {
                        "obstacle_record_index": obstacle["index"],
                        "obstacle_class_name": obstacle["class_name"],
                        "obstacle_kind": obstacle["kind"],
                        "obstacle_strategy": obstacle["strategy"],
                        "obstacle_area": obstacle["area"],
                        "obstacle_bbox": obstacle["bbox"],
                        "selected_contact_pixels": selected_pixels,
                        "satellite_contact_pixels": satellite_pixels,
                        "forward_alignment": alignment,
                    }
                )
            if not matches:
                continue
            selected |= component["mask"]
            accepted.add(component["index"])
            evidence.append(
                {
                    "candidate_component_index": component["index"],
                    "candidate_area": component["area"],
                    "candidate_bbox": component["bbox"],
                    "authorization": "same_obstacle_touches_selected_and_satellite",
                    "obstacle_matches": matches,
                }
            )
            changed = True
    rejected = candidates & ~selected
    valid, left, right = legacy.a2.row_spans(selected)
    span = legacy.a2.span_mask(valid, left, right, candidates.shape)
    span &= ~pipeline.curb_suspect
    result = {
        "candidate": candidates,
        "main": main["mask"],
        "selected_sources": selected,
        "accepted_satellites": selected & ~main["mask"],
        "rejected_sources": rejected,
        "span": span,
        "components": components,
        "main_component_index": main["index"],
        "accepted_component_indices": sorted(accepted),
        "evidence": evidence,
        "obstacle_component_count": len(obstacle_sources),
        "contact_radius_px": radius,
        "empty_candidate": False,
    }
    if authorization_forbidden_mask is not None:
        result["forbidden_crossing_rejections"] = crossing_rejections
    return result
