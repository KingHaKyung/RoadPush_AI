from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.road.context import ROAD_UNION_SEAM_RADIUS_PX, RoadContext
from roadpush_ai.surface.context import SurfaceContext


CATCH_BASIN_ID = 36

# Reviewed linear-drain thresholds. The original Phase-3 Catch Basin <=0.4%
# rule remains unchanged in road_context.py; the enhanced V2 builder applies
# this policy as an additional, independently evidenced acceptance path.
MIN_COMPONENT_PIXELS = 20
MIN_PCA_ELONGATION = 6.0
MIN_TRANSVERSE_ANGLE_DEG = 60.0
STRONG_CORRIDOR_MIN_TRANSVERSE_ANGLE_DEG = 55.0
STRONG_CORRIDOR_MIN_PCA_ELONGATION = 10.0
STRONG_CORRIDOR_MIN_SIDE_SUPPORT_FRACTION = 0.75
STRONG_CORRIDOR_MIN_COMPONENT_FRACTION = 0.70
MAX_MINOR_SPAN_FRACTION = 0.18
MAX_COMPONENT_FRACTION = 0.08
SUPPORT_DEPTH_FRACTION = 0.035
MIN_SIDE_SUPPORT_PIXELS = 20
MIN_SIDE_SUPPORT_FRACTION = 0.18
MIN_CORRIDOR_APPROVED_PIXELS = 20
MIN_CORRIDOR_COMPONENT_FRACTION = 0.20
CORRIDOR_BAND_DEPTH_FRACTION = 0.09
CORRIDOR_MARGIN_PX = 2


@dataclass
class LinearDrainResult:
    road: RoadContext
    raw_catch_basin: np.ndarray
    estimated_corridor: np.ndarray
    approved_linear_drain: np.ndarray
    rejected_linear_drain: np.ndarray
    component_records: list[dict]
    policy: dict


def _segments(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    starts = [int(xs[0])]
    ends: list[int] = []
    for left, right in zip(xs[:-1], xs[1:]):
        if right > left + 1:
            ends.append(int(left))
            starts.append(int(right))
    ends.append(int(xs[-1]))
    return list(zip(starts, ends))


def _component_geometry(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = (
        np.cov(centered, rowvar=False)
        if len(points) >= 2
        else np.zeros((2, 2), dtype=np.float64)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major = eigenvectors[:, order[0]]
    if major[0] < 0:
        major = -major
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    major_projection = centered @ major
    minor_projection = centered @ minor
    major_span = float(np.ptp(major_projection) + 1.0)
    minor_span = float(np.ptp(minor_projection) + 1.0)
    elongation = float(
        math.sqrt(float(eigenvalues[0]) / max(float(eigenvalues[1]), 1e-9))
    )
    return {
        "center": center,
        "major": major,
        "minor": minor,
        "major_span_px": major_span,
        "minor_span_px": minor_span,
        "pca_elongation": elongation,
        "major_axis_angle_deg": float(math.degrees(math.atan2(major[1], major[0]))),
    }


def _route_vector(
    route_path: list[tuple[int, int]],
    center_y: float,
    component_height: int,
) -> tuple[np.ndarray, dict]:
    if len(route_path) < 2:
        return np.array([0.0, 1.0]), {
            "status": "vertical_fallback",
            "point_count": len(route_path),
        }
    points = np.asarray([(x, y) for y, x in route_path], np.float64)
    radius = max(component_height * 3, 25)
    local = points[np.abs(points[:, 1] - center_y) <= radius]
    used = local if len(local) >= 2 else points
    y_values = used[:, 1]
    x_values = used[:, 0]
    if float(np.ptp(y_values)) < 1.0:
        vector = used[-1] - used[0]
    else:
        slope, _ = np.polyfit(y_values, x_values, 1)
        vector = np.array([float(slope), 1.0])
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        vector = np.array([0.0, 1.0])
        norm = 1.0
    return vector / norm, {
        "status": "local_route_fit" if len(local) >= 2 else "full_route_fit",
        "point_count": int(len(used)),
        "local_radius_px": int(radius),
    }


def _shift_union(mask: np.ndarray, direction: np.ndarray, depth: int) -> np.ndarray:
    height, width = mask.shape
    out = np.zeros_like(mask)
    source = mask.astype(np.uint8)
    for distance in range(1, depth + 1):
        dx, dy = np.rint(direction * distance).astype(int)
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        out |= cv2.warpAffine(
            source,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    return out & ~mask


def _side_support(
    component: np.ndarray,
    road_family: np.ndarray,
    minor: np.ndarray,
    depth: int,
) -> tuple[list[dict], list[np.ndarray]]:
    records = []
    bands = []
    for label, sign in (("before", -1.0), ("after", 1.0)):
        band = _shift_union(component, minor * sign, depth)
        support = band & road_family
        band_pixels = int(band.sum())
        support_pixels = int(support.sum())
        records.append(
            {
                "side": label,
                "band_pixels": band_pixels,
                "road_support_pixels": support_pixels,
                "road_support_fraction": float(support_pixels / max(1, band_pixels)),
            }
        )
        bands.append(support)
    return records, bands


def _nearby_road_rows(
    road_family: np.ndarray,
    center_x: float,
    y0: int,
    y1: int,
    depth: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    height = road_family.shape[0]
    groups: list[list[tuple[int, int, int]]] = []
    for start, stop in (
        (max(0, y0 - depth), max(0, y0)),
        (min(height, y1), min(height, y1 + depth)),
    ):
        rows = []
        for y in range(start, stop):
            spans = _segments(road_family[y])
            if not spans:
                continue
            containing = [span for span in spans if span[0] <= center_x <= span[1]]
            span = (
                max(containing, key=lambda value: value[1] - value[0])
                if containing
                else min(spans, key=lambda value: abs((value[0] + value[1]) / 2 - center_x))
            )
            if span[1] - span[0] + 1 >= 5:
                rows.append((y, span[0], span[1]))
        groups.append(rows)
    return groups[0], groups[1]


def _fit_line(rows: list[tuple[int, int, int]], index: int) -> tuple[float, float]:
    ys = np.asarray([row[0] for row in rows], np.float64)
    xs = np.asarray([row[index] for row in rows], np.float64)
    if len(rows) == 1:
        return 0.0, float(xs[0])
    slope, intercept = np.polyfit(ys, xs, 1)
    return float(slope), float(intercept)


def _estimate_corridor(
    shape: tuple[int, int],
    road_family: np.ndarray,
    component: np.ndarray,
    center_x: float,
) -> tuple[np.ndarray, dict]:
    ys, _ = np.nonzero(component)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    depth = max(10, round(shape[0] * CORRIDOR_BAND_DEPTH_FRACTION))
    before, after = _nearby_road_rows(road_family, center_x, y0, y1, depth)
    evidence = before + after
    corridor = np.zeros(shape, dtype=bool)
    if not before or not after or len(evidence) < 4:
        return corridor, {
            "status": "insufficient_independent_road_rows",
            "before_row_count": len(before),
            "after_row_count": len(after),
            "band_depth_px": depth,
        }
    left_fit = _fit_line(evidence, 1)
    right_fit = _fit_line(evidence, 2)
    for y in range(y0, y1):
        left = int(math.floor(left_fit[0] * y + left_fit[1])) - CORRIDOR_MARGIN_PX
        right = int(math.ceil(right_fit[0] * y + right_fit[1])) + CORRIDOR_MARGIN_PX
        left, right = max(0, left), min(shape[1] - 1, right)
        if right >= left:
            corridor[y, left : right + 1] = True
    center_fit = (
        (left_fit[0] + right_fit[0]) / 2,
        (left_fit[1] + right_fit[1]) / 2,
    )
    route_vector = np.array([center_fit[0], 1.0], dtype=np.float64)
    route_vector /= max(float(np.linalg.norm(route_vector)), 1e-9)
    return corridor, {
        "status": "estimated",
        "before_row_count": len(before),
        "after_row_count": len(after),
        "band_depth_px": depth,
        "left_fit_x_per_y": list(left_fit),
        "right_fit_x_per_y": list(right_fit),
        "center_fit_x_per_y": list(center_fit),
        "route_vector_xy": route_vector.tolist(),
    }


def _acute_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = abs(float(np.dot(first, second))) / max(
        float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-9
    )
    return float(math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0)))))


def _touches_frame(mask: np.ndarray) -> dict:
    return {
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
        "top": bool(mask[0].any()),
        "bottom": bool(mask[-1].any()),
    }


def evaluate_linear_transverse_drains(
    labels: np.ndarray,
    road: RoadContext,
    *,
    restore_whole_accepted_component: bool = False,
    production_policy: bool = False,
) -> LinearDrainResult:
    raw_catch_basin = labels == CATCH_BASIN_ID
    road_family = road.raw_road | road.raw_service_lane | road.raw_parking
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        raw_catch_basin.astype(np.uint8), 8
    )
    approved_all = np.zeros_like(raw_catch_basin)
    corridor_all = np.zeros_like(raw_catch_basin)
    records: list[dict] = []
    support_depth = max(6, round(min(labels.shape) * SUPPORT_DEPTH_FRACTION))

    for component_index in range(1, count):
        component = components == component_index
        pixels = int(component.sum())
        if not pixels:
            continue
        x, y, width, height, _ = [int(value) for value in stats[component_index]]
        geometry = _component_geometry(component)
        support, _ = _side_support(
            component, road_family, geometry["minor"], support_depth
        )
        corridor, corridor_record = _estimate_corridor(
            labels.shape, road_family, component, float(geometry["center"][0])
        )
        corridor_all |= corridor
        path_route_vector, route_direction_record = _route_vector(
            road.surface.route_path,
            float(geometry["center"][1]),
            height,
        )
        corridor_route_vector = np.asarray(
            corridor_record.get("route_vector_xy", [0.0, 1.0]), np.float64
        )
        left_slope = corridor_record.get("left_fit_x_per_y", [None])[0]
        right_slope = corridor_record.get("right_fit_x_per_y", [None])[0]
        converging_edges = bool(
            left_slope is not None
            and right_slope is not None
            and float(left_slope) * float(right_slope) <= 0.0
        )
        route_vector = corridor_route_vector if converging_edges else path_route_vector
        route_direction_record.update(
            {
                "selected_estimator": (
                    "corridor_center_from_converging_edges"
                    if converging_edges
                    else "existing_road_route_path"
                ),
                "path_route_vector_xy": path_route_vector.tolist(),
                "corridor_route_vector_xy": corridor_route_vector.tolist(),
                "corridor_edges_converge": converging_edges,
            }
        )
        transverse_angle = _acute_angle_deg(geometry["major"], route_vector)
        unclipped_candidate = component & corridor
        forbidden_overlap = component & road.surface_exclusion
        corridor_clipped_candidate = unclipped_candidate & ~road.surface_exclusion
        corridor_clipped_pixels = int(corridor_clipped_candidate.sum())
        component_fraction = float(pixels / labels.size)
        corridor_fraction = float(corridor_clipped_pixels / max(1, pixels))
        frame_contact = _touches_frame(component)
        strong_corridor_crossing_exception = bool(
            transverse_angle >= STRONG_CORRIDOR_MIN_TRANSVERSE_ANGLE_DEG
            and geometry["pca_elongation"] >= STRONG_CORRIDOR_MIN_PCA_ELONGATION
            and support[0]["road_support_fraction"]
            >= STRONG_CORRIDOR_MIN_SIDE_SUPPORT_FRACTION
            and support[1]["road_support_fraction"]
            >= STRONG_CORRIDOR_MIN_SIDE_SUPPORT_FRACTION
            and corridor_fraction >= STRONG_CORRIDOR_MIN_COMPONENT_FRACTION
        )

        conditions = {
            "minimum_pixels": pixels >= MIN_COMPONENT_PIXELS,
            "long_and_thin": (
                geometry["pca_elongation"] >= MIN_PCA_ELONGATION
                and geometry["minor_span_px"] / min(labels.shape)
                <= MAX_MINOR_SPAN_FRACTION
                and component_fraction <= MAX_COMPONENT_FRACTION
            ),
            "transverse_to_route": (
                transverse_angle >= MIN_TRANSVERSE_ANGLE_DEG
                or strong_corridor_crossing_exception
            ),
            "independent_road_support_before": (
                support[0]["road_support_pixels"] >= MIN_SIDE_SUPPORT_PIXELS
                and support[0]["road_support_fraction"] >= MIN_SIDE_SUPPORT_FRACTION
            ),
            "independent_road_support_after": (
                support[1]["road_support_pixels"] >= MIN_SIDE_SUPPORT_PIXELS
                and support[1]["road_support_fraction"] >= MIN_SIDE_SUPPORT_FRACTION
            ),
            "corridor_estimated_from_both_sides": corridor_record["status"]
            == "estimated",
            "corridor_intersection": (
                corridor_clipped_pixels >= MIN_CORRIDOR_APPROVED_PIXELS
                and corridor_fraction >= MIN_CORRIDOR_COMPONENT_FRACTION
            ),
            "approved_region_avoids_hard_exclusion": not bool(
                (corridor_clipped_candidate & road.surface_exclusion).any()
            ),
        }
        accepted = all(conditions.values())
        approved = (
            component & ~road.surface_exclusion
            if accepted and restore_whole_accepted_component
            else corridor_clipped_candidate
        )
        approved_pixels = int(approved.sum())
        approved_fraction = float(approved_pixels / max(1, pixels))
        if accepted:
            approved_all |= approved
        failed = [name for name, passed in conditions.items() if not passed]
        records.append(
            {
                "component_index": component_index,
                "original_class_id": CATCH_BASIN_ID,
                "original_class_name": "Catch Basin",
                "decision_type": "linear_transverse_drain_probe",
                "pixels": pixels,
                "component_fraction": component_fraction,
                "bbox_xywh": [x, y, width, height],
                "bbox_aspect_ratio": float(max(width, height) / max(1, min(width, height))),
                "pca_elongation": geometry["pca_elongation"],
                "major_span_px": geometry["major_span_px"],
                "minor_span_px": geometry["minor_span_px"],
                "major_axis_angle_deg": geometry["major_axis_angle_deg"],
                "estimated_route_vector_xy": route_vector.tolist(),
                "route_direction": route_direction_record,
                "transverse_angle_deg": transverse_angle,
                "frame_contact": frame_contact,
                "support_depth_px": support_depth,
                "side_support": support,
                "corridor": corridor_record,
                "corridor_pixels_in_component": int(unclipped_candidate.sum()),
                "corridor_clipped_candidate_pixels": corridor_clipped_pixels,
                "forbidden_overlap_pixels": int(forbidden_overlap.sum()),
                "approved_hard_exclusion_overlap_pixels": int(
                    (approved & road.surface_exclusion).sum()
                ),
                "approved_pixels": approved_pixels if accepted else 0,
                "approved_fraction_of_component": approved_fraction if accepted else 0.0,
                "whole_object_restoration_enabled": restore_whole_accepted_component,
                "whole_object_restored_pixels": (
                    int((approved & ~corridor_clipped_candidate).sum()) if accepted else 0
                ),
                "already_in_production_repair_pixels": int(
                    (component & road.accepted_repair).sum()
                ),
                "conditions": conditions,
                "base_transverse_condition_passed": transverse_angle
                >= MIN_TRANSVERSE_ANGLE_DEG,
                "strong_corridor_crossing_exception_passed": strong_corridor_crossing_exception,
                "accepted": accepted,
                "decision_reasons": (
                    [
                        "strong_corridor_crossing_exception"
                        if strong_corridor_crossing_exception
                        and transverse_angle < MIN_TRANSVERSE_ANGLE_DEG
                        else "all_linear_transverse_drain_conditions_passed"
                    ]
                    + (["whole_accepted_component_restored"] if restore_whole_accepted_component else [])
                    if accepted
                    else [f"failed:{name}" for name in failed]
                ),
            }
        )

    approved_all &= ~road.surface_exclusion
    repaired_union = road.surface.repaired_mask | approved_all
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2
    )
    repaired_union = cv2.morphologyEx(
        repaired_union.astype(np.uint8), cv2.MORPH_CLOSE, seam_kernel
    ).astype(bool)
    repaired_union &= ~road.surface_exclusion
    structure, additions = legacy.base.structure_copy(repaired_union)
    structure &= ~road.surface_exclusion
    additions &= ~road.surface_exclusion
    route, route_path = legacy.outer.route_path(repaired_union)
    observed = legacy.outer.selected_component(structure, route_path)
    if not observed.any():
        observed = structure.copy()
    source_component = legacy.outer.selected_component(repaired_union, route_path)
    if not source_component.any():
        source_component = repaired_union.copy()
    surface = SurfaceContext(
        profile=road.surface.profile,
        source_mask=source_component,
        repaired_mask=repaired_union,
        structure_copy=structure,
        forbidden_mask=road.surface.forbidden_mask,
        observed_component=observed,
        route=route,
        route_path=route_path,
        structure_copy_additions=additions,
    )
    probe_road = replace(
        road,
        surface=surface,
        repaired_road=road.repaired_road | approved_all,
        accepted_repair=road.accepted_repair | approved_all,
        policy={
            **road.policy,
            "linear_transverse_drain_enabled": True,
            "linear_transverse_drain_probe_enabled": not production_policy,
            "linear_drain_whole_object_restoration": restore_whole_accepted_component,
            "production_catch_basin_rule_changed": production_policy,
        },
    )
    policy = {
        "probe_only": not production_policy,
        "production_policy_modified": production_policy,
        "catch_basin_class_id": CATCH_BASIN_ID,
        "minimum_component_pixels": MIN_COMPONENT_PIXELS,
        "minimum_pca_elongation": MIN_PCA_ELONGATION,
        "minimum_transverse_angle_deg": MIN_TRANSVERSE_ANGLE_DEG,
        "strong_corridor_exception": {
            "minimum_transverse_angle_deg": STRONG_CORRIDOR_MIN_TRANSVERSE_ANGLE_DEG,
            "minimum_pca_elongation": STRONG_CORRIDOR_MIN_PCA_ELONGATION,
            "minimum_side_support_fraction_each": STRONG_CORRIDOR_MIN_SIDE_SUPPORT_FRACTION,
            "minimum_corridor_component_fraction": STRONG_CORRIDOR_MIN_COMPONENT_FRACTION,
            "all_conditions_required": True,
        },
        "maximum_minor_span_fraction": MAX_MINOR_SPAN_FRACTION,
        "maximum_component_fraction": MAX_COMPONENT_FRACTION,
        "support_depth_fraction": SUPPORT_DEPTH_FRACTION,
        "minimum_side_support_pixels": MIN_SIDE_SUPPORT_PIXELS,
        "minimum_side_support_fraction": MIN_SIDE_SUPPORT_FRACTION,
        "minimum_corridor_approved_pixels": MIN_CORRIDOR_APPROVED_PIXELS,
        "minimum_corridor_component_fraction": MIN_CORRIDOR_COMPONENT_FRACTION,
        "corridor_band_depth_fraction": CORRIDOR_BAND_DEPTH_FRACTION,
        "corridor_margin_px": CORRIDOR_MARGIN_PX,
        "approval_clipped_to_estimated_corridor": not restore_whole_accepted_component,
        "whole_accepted_component_restoration": restore_whole_accepted_component,
        "whole_object_eligibility_uses_corridor_clipped_evidence": True,
        "hard_exclusion_applied_after_approval": True,
        "original_catch_basin_diagnostics_preserved": True,
    }
    return LinearDrainResult(
        road=probe_road,
        raw_catch_basin=raw_catch_basin,
        estimated_corridor=corridor_all,
        approved_linear_drain=approved_all,
        rejected_linear_drain=raw_catch_basin & ~approved_all,
        component_records=records,
        policy=policy,
    )


# Historical name retained for callers that recorded the probe result type.
LinearDrainProbeResult = LinearDrainResult
