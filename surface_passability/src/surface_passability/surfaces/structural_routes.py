"""Structural-route helpers with exact observed-seed semantics."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.surfaces import (
    RecomputedStructuralRoute,
    SurfacePipelineContext,
)
from surface_passability.policies.surfaces import (
    ROUTE_COMPONENT_DISTANCE_MASK_SIZE,
    ROUTE_HIGH_CONFIDENCE_MAXIMUM_ABSOLUTE_ANGLE_DEGREES,
    ROUTE_HIGH_CONFIDENCE_MINIMUM_VERTICAL_PROGRESS,
)
from surface_passability.surfaces.routes import (
    connected_component_from_seed,
    find_weighted_route_paths,
    measure_route_path_angle,
    select_route_goals,
    select_routed_surface_component,
    smooth_route_path,
    stabilize_route_surface,
)


def select_component_at_exact_seed(
    mask: np.ndarray,
    seed: tuple[int, int] | None,
) -> np.ndarray:
    """Return only the component containing an explicit foreground seed."""

    if seed is None or not mask[seed]:
        return np.zeros_like(mask)
    return connected_component_from_seed(mask, seed)


def measure_route_displacement(
    observed_path: list[tuple[int, int]],
    candidate_path: list[tuple[int, int]],
) -> dict[str, object]:
    """Measure horizontal displacement between two routes at shared rows."""

    if not observed_path or not candidate_path:
        return {
            "status": "unavailable",
            "rows": [],
            "mean_abs_dx_px": None,
            "max_abs_dx_px": None,
        }
    observed = np.asarray(observed_path, np.float32)
    candidate = np.asarray(candidate_path, np.float32)
    y0 = int(max(observed[:, 0].min(), candidate[:, 0].min()))
    y1 = int(min(observed[:, 0].max(), candidate[:, 0].max()))
    if y1 < y0:
        return {
            "status": "no_y_overlap",
            "rows": [],
            "mean_abs_dx_px": None,
            "max_abs_dx_px": None,
        }
    levels = np.unique(
        np.linspace(y1, y0, 18).round().astype(int)
    )[::-1]
    rows = []
    for y in levels:
        observed_x = float(
            observed[
                int(np.argmin(np.abs(observed[:, 0] - y))),
                1,
            ]
        )
        candidate_x = float(
            candidate[
                int(np.argmin(np.abs(candidate[:, 0] - y))),
                1,
            ]
        )
        rows.append(
            {
                "y": int(y),
                "observed_x": observed_x,
                "test_x": candidate_x,
                "signed_dx_px": candidate_x - observed_x,
                "abs_dx_px": abs(candidate_x - observed_x),
            }
        )
    values = [row["abs_dx_px"] for row in rows]
    return {
        "status": "available",
        "rows": rows,
        "mean_abs_dx_px": float(np.mean(values)),
        "median_abs_dx_px": float(np.median(values)),
        "max_abs_dx_px": float(np.max(values)),
    }


def group_consecutive_indices(indices: list[int]) -> list[list[int]]:
    """Group adjacent indices while preserving their input order and objects."""

    if not indices:
        return []
    groups = [[indices[0]]]
    for index in indices[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def paint_boundary_row_span(
    mask: np.ndarray,
    rows: list[dict[str, object]],
    indices: list[int],
    side: str,
) -> None:
    """Paint one side of selected boundary rows into the supplied mask."""

    if not indices:
        return
    height, width = mask.shape
    y_values = [rows[index]["y"] for index in indices]
    y0 = max(0, min(y_values) - 6)
    y1 = min(height - 1, max(y_values) + 6)
    for y in range(y0, y1 + 1):
        nearest = min(
            indices,
            key=lambda index: abs(rows[index]["y"] - y),
        )
        row = rows[nearest]
        if side == "left":
            x0, x1 = row["left"], row["main_left"]
        else:
            x0, x1 = row["main_right"] + 1, row["right"] + 1
        if x1 > x0:
            mask[y, max(0, x0) : min(width, x1)] = True


def recompute_structural_route(
    boundary_mask: np.ndarray,
    context: SurfacePipelineContext,
) -> RecomputedStructuralRoute:
    """Recompute a route while retaining observed start and direction identity."""

    observed_start = context.route.start
    component = select_component_at_exact_seed(boundary_mask, observed_start)
    if not component.any():
        component = select_routed_surface_component(
            boundary_mask,
            context.route_path,
        )
    if not component.any():
        component = boundary_mask.copy()

    clean = stabilize_route_surface(component)
    clean_component = select_component_at_exact_seed(clean, observed_start)
    if clean_component.any():
        component = clean_component
    padded = np.pad(component.astype(np.uint8), 1)
    distance_map = cv2.distanceTransform(
        padded,
        cv2.DIST_L2,
        ROUTE_COMPONENT_DISTANCE_MASK_SIZE,
    )[1:-1, 1:-1]
    start = (
        observed_start
        if observed_start is not None and component[observed_start]
        else None
    )
    if start is None:
        rows, columns = np.nonzero(component)
        if len(columns) and observed_start is not None:
            index = int(
                np.argmin(
                    (rows - observed_start[0]) ** 2
                    + (columns - observed_start[1]) ** 2
                )
            )
            start = (int(rows[index]), int(columns[index]))

    target_y, goals = (
        select_route_goals(component, distance_map)
        if start
        else (0, [])
    )
    paths, totals = (
        find_weighted_route_paths(
            component,
            distance_map,
            start,
            goals,
        )
        if start and goals
        else ([], [])
    )
    paths = [smooth_route_path(path, component) for path in paths]
    observed_angle = measure_route_path_angle(context.route_path)
    ranking = []
    for index, (path, total) in enumerate(zip(paths, totals)):
        angle = measure_route_path_angle(path)
        angle_delta = (
            abs(angle - observed_angle)
            if angle is not None and observed_angle is not None
            else float("inf")
        )
        ranking.append(
            (angle_delta, total / max(1, len(path)), index)
        )
    main_index = min(ranking)[2] if ranking else None
    selected_path = paths[main_index] if main_index is not None else []
    angle = measure_route_path_angle(selected_path)
    vertical_progress = (
        float(
            (selected_path[0][0] - selected_path[-1][0])
            / boundary_mask.shape[0]
        )
        if len(selected_path) >= 2
        else 0.0
    )
    return RecomputedStructuralRoute(
        start=start,
        component_mask=component,
        target_y=target_y,
        goals=goals,
        paths=paths,
        totals=totals,
        main_index=main_index,
        selected_path=selected_path,
        angle=angle,
        vertical_progress=vertical_progress,
        high_confidence=bool(
            main_index is not None
            and vertical_progress
            >= ROUTE_HIGH_CONFIDENCE_MINIMUM_VERTICAL_PROGRESS
            and angle is not None
            and abs(angle)
            <= ROUTE_HIGH_CONFIDENCE_MAXIMUM_ABSOLUTE_ANGLE_DEGREES
        ),
        observed_start_reused=bool(
            start == observed_start and start is not None
        ),
        observed_direction_angle_degrees=observed_angle,
        direction_constraint=(
            "minimum_absolute_angle_delta_then_normalized_cost"
        ),
    )
