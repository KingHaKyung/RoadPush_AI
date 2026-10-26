"""Route seed and connected-component selection for traversable surfaces."""

from __future__ import annotations

import heapq
import math

import cv2
import numpy as np

from surface_passability.contracts.surfaces import RouteAttempt
from surface_passability.geometry.intervals import true_intervals
from surface_passability.policies.surfaces import (
    ROUTE_COMPONENT_DISTANCE_MASK_SIZE,
    ROUTE_GOAL_MAXIMUM_COUNT,
    ROUTE_GOAL_MINIMUM_RUN_WIDTH_PX,
    ROUTE_GOAL_MINIMUM_TARGET_OFFSET_ROWS,
    ROUTE_GOAL_TARGET_OFFSET_FRACTION,
    ROUTE_HIGH_CONFIDENCE_MAXIMUM_ABSOLUTE_ANGLE_DEGREES,
    ROUTE_HIGH_CONFIDENCE_MINIMUM_VERTICAL_PROGRESS,
    ROUTE_PATH_CLEARANCE_OFFSET_PX,
    ROUTE_PATH_DOWNWARD_DIRECTION_PENALTY,
    ROUTE_PATH_EDGE_CLEARANCE_WEIGHT,
    ROUTE_PATH_LATERAL_DIRECTION_PENALTY,
    ROUTE_PATH_UPWARD_DIRECTION_PENALTY,
    ROUTE_SMOOTHING_MAXIMUM_WINDOW_POINTS,
    ROUTE_SMOOTHING_MINIMUM_PATH_POINTS,
    ROUTE_SMOOTHING_MINIMUM_WINDOW_POINTS,
    ROUTE_SMOOTHING_PATH_LENGTH_DIVISOR,
    ROUTE_QUALITY_ANGLE_NORMALIZATION_DEGREES,
    ROUTE_START_CENTER_DISTANCE_WEIGHT,
    ROUTE_START_FALLBACK_HEIGHT_FRACTION,
    ROUTE_START_MAXIMUM_HORIZONTAL_FRACTION,
    ROUTE_START_MINIMUM_HEIGHT_FRACTION,
    ROUTE_START_MINIMUM_HORIZONTAL_FRACTION,
    ROUTE_START_MINIMUM_SEARCH_ROWS,
    ROUTE_START_PRIMARY_HEIGHT_FRACTION,
    ROUTE_START_SEARCH_HEIGHT_FRACTION,
    ROUTE_SURFACE_STABILIZATION_KERNEL_PX,
)


def measure_route_path_angle(
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> float | None:
    """Measure endpoint direction in degrees relative to upward image travel."""

    if len(path) < 2:
        return None
    start, end = path[0], path[-1]
    return float(
        math.degrees(
            math.atan2(
                end[1] - start[1],
                max(1, start[0] - end[0]),
            )
        )
    )


def stabilize_route_surface(surface_mask: np.ndarray) -> np.ndarray:
    """Close only small jagged gaps before route search."""

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            ROUTE_SURFACE_STABILIZATION_KERNEL_PX,
            ROUTE_SURFACE_STABILIZATION_KERNEL_PX,
        ),
    )
    return (
        cv2.morphologyEx(
            surface_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        )
        > 0
    )


def select_route_start(
    surface_mask: np.ndarray,
    distance_map: np.ndarray,
    height_fraction: float,
) -> tuple[int, int] | None:
    """Select the first eligible near-viewer seed using the frozen score."""

    height, width = surface_mask.shape
    center_x = (width - 1) / 2
    preferred_y = min(height - 2, round(height * height_fraction))
    for offset in range(
        0,
        max(
            ROUTE_START_MINIMUM_SEARCH_ROWS,
            round(height * ROUTE_START_SEARCH_HEIGHT_FRACTION),
        ),
    ):
        rows = (
            [preferred_y - offset]
            if offset == 0
            else [preferred_y - offset, preferred_y + offset]
        )
        for y in rows:
            if (
                y < round(height * ROUTE_START_MINIMUM_HEIGHT_FRACTION)
                or y >= height - 1
            ):
                continue
            xs = np.flatnonzero(surface_mask[y])
            if not len(xs):
                continue
            central = xs[
                (
                    xs
                    >= round(
                        width * ROUTE_START_MINIMUM_HORIZONTAL_FRACTION
                    )
                )
                & (
                    xs
                    <= round(
                        width * ROUTE_START_MAXIMUM_HORIZONTAL_FRACTION
                    )
                )
            ]
            candidates = central if len(central) else xs
            score = distance_map[y, candidates] - (
                ROUTE_START_CENTER_DISTANCE_WEIGHT
                * np.abs(candidates - center_x)
            )
            x = int(candidates[int(np.argmax(score))])
            return y, x
    return None


def connected_component_from_seed(
    surface_mask: np.ndarray,
    seed: tuple[int, int],
) -> np.ndarray:
    """Return the eight-connected foreground component containing ``seed``."""

    count, labels = cv2.connectedComponents(surface_mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(surface_mask)
    label = int(labels[seed])
    return labels == label if label else np.zeros_like(surface_mask)


def select_routed_surface_component(
    structural_mask: np.ndarray,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> np.ndarray:
    """Select the structural component reached by the route or fallback seed."""

    seed = route_path[0] if route_path else None
    if seed is None or not structural_mask[seed]:
        padded = np.pad(structural_mask.astype(np.uint8), 1)
        distance_map = cv2.distanceTransform(
            padded,
            cv2.DIST_L2,
            ROUTE_COMPONENT_DISTANCE_MASK_SIZE,
        )[1:-1, 1:-1]
        seed = select_route_start(
            structural_mask,
            distance_map,
            ROUTE_START_PRIMARY_HEIGHT_FRACTION,
        )
        if seed is None:
            seed = select_route_start(
                structural_mask,
                distance_map,
                ROUTE_START_FALLBACK_HEIGHT_FRACTION,
            )
    return (
        connected_component_from_seed(structural_mask, seed)
        if seed
        else np.zeros_like(structural_mask)
    )


def select_route_goals(
    component_mask: np.ndarray,
    distance_map: np.ndarray,
) -> tuple[int, list[tuple[int, int]]]:
    """Select the safest upper exits from the seeded route component."""

    _, width = component_mask.shape
    occupied_rows = np.flatnonzero(component_mask.any(axis=1))
    if not len(occupied_rows):
        return 0, []
    top, bottom = int(occupied_rows[0]), int(occupied_rows[-1])
    target_y = min(
        bottom,
        top
        + max(
            ROUTE_GOAL_MINIMUM_TARGET_OFFSET_ROWS,
            round((bottom - top) * ROUTE_GOAL_TARGET_OFFSET_FRACTION),
        ),
    )
    while target_y <= bottom and not component_mask[target_y].any():
        target_y += 1

    candidates: list[tuple[int, int]] = []
    for left, right in true_intervals(component_mask[target_y]):
        if right - left + 1 < ROUTE_GOAL_MINIMUM_RUN_WIDTH_PX:
            continue
        xs = np.arange(left, right + 1)
        local = distance_map[target_y, xs]
        best_value = float(local.max())
        tied = xs[np.isclose(local, best_value)]
        x = int(tied[np.argmin(np.abs(tied - (width - 1) / 2))])
        candidates.append((target_y, x))
    candidates.sort(key=lambda point: float(distance_map[point]), reverse=True)
    return target_y, candidates[:ROUTE_GOAL_MAXIMUM_COUNT]


def find_weighted_route_paths(
    component_mask: np.ndarray,
    distance_map: np.ndarray,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Find frozen clearance- and direction-weighted paths to reachable goals."""

    height, width = component_mask.shape
    costs = np.full((height, width), np.inf, np.float64)
    previous = np.full((height, width, 2), -1, np.int32)
    costs[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    remaining = set(goals)
    moves = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    while queue and remaining:
        current, y, x = heapq.heappop(queue)
        if current != costs[y, x]:
            continue
        remaining.discard((y, x))
        for dy, dx in moves:
            next_y, next_x = y + dy, x + dx
            if (
                next_y < 0
                or next_y >= height
                or next_x < 0
                or next_x >= width
                or not component_mask[next_y, next_x]
            ):
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            edge_penalty = 1.0 + ROUTE_PATH_EDGE_CLEARANCE_WEIGHT / (
                float(distance_map[next_y, next_x])
                + ROUTE_PATH_CLEARANCE_OFFSET_PX
            )
            direction_penalty = (
                ROUTE_PATH_DOWNWARD_DIRECTION_PENALTY
                if dy > 0
                else (
                    ROUTE_PATH_LATERAL_DIRECTION_PENALTY
                    if dy == 0
                    else ROUTE_PATH_UPWARD_DIRECTION_PENALTY
                )
            )
            proposed = current + step * edge_penalty * direction_penalty
            if proposed < costs[next_y, next_x]:
                costs[next_y, next_x] = proposed
                previous[next_y, next_x] = (y, x)
                heapq.heappush(queue, (proposed, next_y, next_x))

    paths: list[list[tuple[int, int]]] = []
    totals: list[float] = []
    for goal in goals:
        if not np.isfinite(costs[goal]):
            continue
        path = [goal]
        cursor = goal
        while cursor != start:
            previous_y, previous_x = map(int, previous[cursor])
            if previous_y < 0:
                path = []
                break
            cursor = (previous_y, previous_x)
            path.append(cursor)
        if path:
            path.reverse()
            paths.append(path)
            totals.append(float(costs[goal]))
    return paths, totals


def smooth_route_path(
    path: list[tuple[int, int]],
    component_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Smooth a route while retaining endpoints and rejecting off-component points."""

    if len(path) < ROUTE_SMOOTHING_MINIMUM_PATH_POINTS:
        return path
    points = np.asarray(path, np.float32)
    window = max(
        ROUTE_SMOOTHING_MINIMUM_WINDOW_POINTS,
        min(
            ROUTE_SMOOTHING_MAXIMUM_WINDOW_POINTS,
            (len(path) // ROUTE_SMOOTHING_PATH_LENGTH_DIVISOR) | 1,
        ),
    )
    kernel = np.ones(window, np.float32) / window
    ys = np.convolve(points[:, 0], kernel, mode="same")
    xs = np.convolve(points[:, 1], kernel, mode="same")
    half = window // 2
    ys[:half], xs[:half] = points[:half, 0], points[:half, 1]
    ys[-half:], xs[-half:] = points[-half:, 0], points[-half:, 1]
    result: list[tuple[int, int]] = []
    for original, y, x in zip(path, ys, xs):
        candidate = (int(round(y)), int(round(x)))
        result.append(candidate if component_mask[candidate] else original)
    return result


def build_route_attempt(
    clean_surface_mask: np.ndarray,
    distance_map: np.ndarray,
    start_fraction: float,
) -> RouteAttempt:
    """Compose one route-search attempt from the approved route primitives."""

    start = select_route_start(clean_surface_mask, distance_map, start_fraction)
    component = (
        connected_component_from_seed(clean_surface_mask, start)
        if start
        else np.zeros_like(clean_surface_mask)
    )
    target_y, goals = (
        select_route_goals(component, distance_map)
        if start
        else (0, [])
    )
    paths, totals = (
        find_weighted_route_paths(component, distance_map, start, goals)
        if start and goals
        else ([], [])
    )
    paths = [smooth_route_path(path, component) for path in paths]
    main_index = (
        int(
            np.argmin(
                [
                    cost / max(1, len(path))
                    for cost, path in zip(totals, paths)
                ]
            )
        )
        if paths
        else None
    )
    angle = None
    vertical_progress = 0.0
    if main_index is not None and len(paths[main_index]) >= 2:
        first, last = paths[main_index][0], paths[main_index][-1]
        vertical_progress = float(
            (first[0] - last[0]) / clean_surface_mask.shape[0]
        )
        angle = math.degrees(
            math.atan2(
                last[1] - first[1],
                max(1, first[0] - last[0]),
            )
        )
    high_confidence = bool(
        main_index is not None
        and vertical_progress
        >= ROUTE_HIGH_CONFIDENCE_MINIMUM_VERTICAL_PROGRESS
        and angle is not None
        and abs(angle)
        <= ROUTE_HIGH_CONFIDENCE_MAXIMUM_ABSOLUTE_ANGLE_DEGREES
    )
    quality = vertical_progress - (
        abs(angle) / ROUTE_QUALITY_ANGLE_NORMALIZATION_DEGREES
        if angle is not None
        else 1.0
    )
    return RouteAttempt(
        requested_start_fraction=start_fraction,
        start=start,
        component_mask=component,
        target_y=target_y,
        goals=tuple(goals),
        paths=tuple(tuple(path) for path in paths),
        totals=tuple(totals),
        main_index=main_index,
        angle=angle,
        vertical_progress=vertical_progress,
        high_confidence=high_confidence,
        quality=quality,
    )


def solve_surface_route(repaired_surface_mask: np.ndarray) -> RouteAttempt:
    """Solve one surface route with the accepted missing-start fallback."""

    clean_surface_mask = stabilize_route_surface(repaired_surface_mask)
    padded = np.pad(clean_surface_mask.astype(np.uint8), 1)
    distance_map = cv2.distanceTransform(
        padded,
        cv2.DIST_L2,
        ROUTE_COMPONENT_DISTANCE_MASK_SIZE,
    )[1:-1, 1:-1]
    attempt = build_route_attempt(
        clean_surface_mask,
        distance_map,
        ROUTE_START_PRIMARY_HEIGHT_FRACTION,
    )
    if attempt.start is None:
        attempt = build_route_attempt(
            clean_surface_mask,
            distance_map,
            ROUTE_START_FALLBACK_HEIGHT_FRACTION,
        )
    return attempt


def route_x_for_nearest_row(
    path: list[tuple[int, int]],
    row_y: int,
    default_x: float,
) -> float:
    """Return the x-coordinate of the route point nearest to one row."""

    if not path:
        return default_x
    points = np.asarray(path, np.float32)
    index = int(np.argmin(np.abs(points[:, 0] - row_y)))
    return float(points[index, 1])


