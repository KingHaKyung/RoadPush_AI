from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.boundary.sidewalk import BoundaryMode, BoundaryResult
from roadpush_ai.surface.preprocessing import SharedPreprocessing


from roadpush_ai.legacy_policy import boundary_width as BRANCH
DIRECTION = legacy.outer.direction


@dataclass
class StructuralRouteResult:
    raw_boundary: np.ndarray
    pruned_boundary: np.ndarray
    route: dict
    route_path: list[tuple[int, int]]
    route_displacement: dict
    raw_candidate: np.ndarray
    vetoed_candidate: np.ndarray
    approved_removal: np.ndarray
    candidate_rows: list[dict]
    accepted_groups: list[dict]
    calls: dict[str, bool]


def _path_angle(path: list[tuple[int, int]]) -> float | None:
    if len(path) < 2:
        return None
    start, end = path[0], path[-1]
    return float(math.degrees(math.atan2(end[1] - start[1], max(1, start[0] - end[0]))))


def _exact_seed_component(mask: np.ndarray, seed: tuple[int, int] | None) -> np.ndarray:
    if seed is None or not mask[seed]:
        return np.zeros_like(mask)
    return DIRECTION.component_from_seed(mask, seed)


def recompute_structural_route(
    boundary: np.ndarray,
    shared: SharedPreprocessing,
) -> tuple[dict, list[tuple[int, int]]]:
    """Recompute a route while retaining the observed start and direction identity.

    The exact Test1 start seed first selects the connected Test2/Test3 component.
    Candidate exits are then ranked by angular agreement with the common observed
    path before normalized path cost. No blocker or free-space mask is consulted.
    """
    observed_start = shared.route.get("start")
    component = _exact_seed_component(boundary, observed_start)
    if not component.any():
        component = legacy.outer.selected_component(boundary, shared.route_path)
    if not component.any():
        component = boundary.copy()

    clean = DIRECTION.preprocess(component)
    clean_component = _exact_seed_component(clean, observed_start)
    if clean_component.any():
        component = clean_component
    padded = np.pad(component.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    start = observed_start if observed_start is not None and component[observed_start] else None
    if start is None:
        # This is a diagnostic fallback only; normal V3 boundaries retain the
        # observed start. It never switches to a disconnected component.
        ys, xs = np.nonzero(component)
        if len(xs) and observed_start is not None:
            index = int(np.argmin((ys - observed_start[0]) ** 2 + (xs - observed_start[1]) ** 2))
            start = (int(ys[index]), int(xs[index]))

    target_y, goals = DIRECTION.choose_goals(component, distance) if start else (0, [])
    paths, totals = DIRECTION.shortest_paths(component, distance, start, goals) if start and goals else ([], [])
    paths = [DIRECTION.smooth_path(path, component) for path in paths]
    observed_angle = _path_angle(shared.route_path)
    ranking = []
    for index, (path, total) in enumerate(zip(paths, totals)):
        angle = _path_angle(path)
        angle_delta = abs(angle - observed_angle) if angle is not None and observed_angle is not None else float("inf")
        ranking.append((angle_delta, total / max(1, len(path)), index))
    main_index = min(ranking)[2] if ranking else None
    selected = paths[main_index] if main_index is not None else []
    angle = _path_angle(selected)
    progress = float((selected[0][0] - selected[-1][0]) / boundary.shape[0]) if len(selected) >= 2 else 0.0
    route = {
        "requested_start_fraction": None,
        "start": start,
        "component": component,
        "target_y": target_y,
        "goals": goals,
        "paths": paths,
        "totals": totals,
        "main_index": main_index,
        "angle": angle,
        "vertical_progress": progress,
        "high_confidence": bool(main_index is not None and progress >= 0.20 and angle is not None and abs(angle) <= 60.0),
        "observed_start_reused": bool(start == observed_start and start is not None),
        "observed_direction_angle_deg": observed_angle,
        "direction_constraint": "minimum_absolute_angle_delta_then_normalized_cost",
    }
    return route, selected


def route_displacement(
    observed_path: list[tuple[int, int]],
    test_path: list[tuple[int, int]],
) -> dict:
    if not observed_path or not test_path:
        return {"status": "unavailable", "rows": [], "mean_abs_dx_px": None, "max_abs_dx_px": None}
    observed = np.asarray(observed_path, np.float32)
    test = np.asarray(test_path, np.float32)
    y0 = int(max(observed[:, 0].min(), test[:, 0].min()))
    y1 = int(min(observed[:, 0].max(), test[:, 0].max()))
    if y1 < y0:
        return {"status": "no_y_overlap", "rows": [], "mean_abs_dx_px": None, "max_abs_dx_px": None}
    levels = np.unique(np.linspace(y1, y0, 18).round().astype(int))[::-1]
    rows = []
    for y in levels:
        ox = float(observed[int(np.argmin(np.abs(observed[:, 0] - y))), 1])
        tx = float(test[int(np.argmin(np.abs(test[:, 0] - y))), 1])
        rows.append({"y": int(y), "observed_x": ox, "test_x": tx, "signed_dx_px": tx - ox, "abs_dx_px": abs(tx - ox)})
    values = [row["abs_dx_px"] for row in rows]
    return {
        "status": "available",
        "rows": rows,
        "mean_abs_dx_px": float(np.mean(values)),
        "median_abs_dx_px": float(np.median(values)),
        "max_abs_dx_px": float(np.max(values)),
    }


def _runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups = [[indices[0]]]
    for index in indices[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _paint_row_span(mask: np.ndarray, rows: list[dict], indices: list[int], side: str) -> None:
    if not indices:
        return
    h, w = mask.shape
    y_values = [rows[index]["y"] for index in indices]
    y0, y1 = max(0, min(y_values) - 6), min(h - 1, max(y_values) + 6)
    for y in range(y0, y1 + 1):
        nearest = min(indices, key=lambda index: abs(rows[index]["y"] - y))
        row = rows[nearest]
        if side == "left":
            x0, x1 = row["left"], row["main_left"]
        else:
            x0, x1 = row["main_right"] + 1, row["right"] + 1
        if x1 > x0:
            mask[y, max(0, x0):min(w, x1)] = True


def prune_side_branches(
    raw_boundary: np.ndarray,
    route: dict,
    path: list[tuple[int, int]],
    shared: SharedPreprocessing,
    mode: BoundaryMode,
) -> dict:
    if mode == BoundaryMode.TEST1_OBSERVED_ONLY:
        empty = np.zeros_like(raw_boundary)
        return {
            "rows": [], "raw_candidate": empty, "vetoed_candidate": empty,
            "approved_removal": empty, "after": raw_boundary.copy(),
            "accepted_groups": [], "policy": "disabled_for_test1",
            "route_high_confidence": bool(route.get("high_confidence")),
            "far_limit_y": round(raw_boundary.shape[0] * 0.70),
            "footprint_or_blocker_used": False,
        }

    gray = cv2.cvtColor(shared.image, cv2.COLOR_RGB2GRAY)
    rows = BRANCH.collect_rows(raw_boundary, path, shared.labels, gray)
    # The accepted integration contract forbids footprint/blocker evidence in
    # structural pruning. Endpoint semantic kinds are still recorded by the
    # historical row collector, but no blocker mask can suppress or approve a branch.
    BRANCH.branch_and_width_analysis(rows, shared.image.shape[1], np.zeros_like(raw_boundary))
    h = raw_boundary.shape[0]
    far_limit = round(h * 0.70)
    raw_candidate = np.zeros_like(raw_boundary)
    vetoed_candidate = np.zeros_like(raw_boundary)
    approved = np.zeros_like(raw_boundary)
    accepted_groups = []
    candidate_rows = []

    for side in ("left", "right"):
        flag, kind = f"{side}_branch_candidate", f"{side}_kind"
        flagged = [index for index, row in enumerate(rows) if row.get(flag)]
        for index in flagged:
            _paint_row_span(raw_candidate, rows, [index], side)
        eligible = []
        for index in flagged:
            row = rows[index]
            veto = None
            if row["y"] > far_limit:
                veto = "below_upper_70_percent"
            elif row[kind] == "frame_truncated":
                veto = "frame_truncated"
            if veto is None:
                eligible.append(index)
            else:
                _paint_row_span(vetoed_candidate, rows, [index], side)
            candidate_rows.append({
                "row_index": index, "y": row["y"], "side": side,
                "endpoint_kind": row[kind], "veto_reason": veto,
                "accepted": False,
            })

        for group in _runs(eligible):
            if len(group) < 2:
                for index in group:
                    next(record for record in candidate_rows if record["row_index"] == index and record["side"] == side)["veto_reason"] = "fewer_than_2_consecutive_rows"
                    _paint_row_span(vetoed_candidate, rows, [index], side)
                continue
            if not route.get("high_confidence"):
                for index in group:
                    next(record for record in candidate_rows if record["row_index"] == index and record["side"] == side)["veto_reason"] = "structural_route_not_high_confidence"
                _paint_row_span(vetoed_candidate, rows, group, side)
                continue
            _paint_row_span(approved, rows, group, side)
            for index in group:
                record = next(record for record in candidate_rows if record["row_index"] == index and record["side"] == side)
                record["accepted"] = True
            accepted_groups.append({"side": side, "sampled_row_indices": group, "sampled_y": [rows[index]["y"] for index in group]})

    raw_candidate &= raw_boundary
    vetoed_candidate &= raw_boundary
    approved &= raw_boundary
    return {
        "rows": rows,
        "raw_candidate": raw_candidate,
        "vetoed_candidate": vetoed_candidate,
        "approved_removal": approved,
        "after": raw_boundary & ~approved,
        "accepted_groups": accepted_groups,
        "candidate_rows": candidate_rows,
        "policy": "upper70_two_consecutive_frame_veto",
        "route_high_confidence": bool(route.get("high_confidence")),
        "far_limit_y": far_limit,
        "footprint_or_blocker_used": False,
    }


def build_structural_route_stage(
    boundary: BoundaryResult,
    shared: SharedPreprocessing,
) -> StructuralRouteResult:
    if boundary.mode == BoundaryMode.TEST1_OBSERVED_ONLY:
        route, path = shared.route, shared.route_path
    else:
        route, path = recompute_structural_route(boundary.mask, shared)
    pruning = prune_side_branches(boundary.mask, route, path, shared, boundary.mode)
    calls = dict(boundary.calls)
    calls.update({
        "test_specific_structural_route": boundary.mode != BoundaryMode.TEST1_OBSERVED_ONLY,
        "side_branch_pruning": boundary.mode != BoundaryMode.TEST1_OBSERVED_ONLY,
        "blocker_used_for_side_pruning": False,
    })
    return StructuralRouteResult(
        raw_boundary=boundary.mask,
        pruned_boundary=pruning["after"],
        route=route,
        route_path=path,
        route_displacement=route_displacement(shared.route_path, path),
        raw_candidate=pruning["raw_candidate"],
        vetoed_candidate=pruning["vetoed_candidate"],
        approved_removal=pruning["approved_removal"],
        candidate_rows=pruning.get("candidate_rows", []),
        accepted_groups=pruning["accepted_groups"],
        calls=calls,
    )

