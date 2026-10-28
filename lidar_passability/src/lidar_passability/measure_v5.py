"""Curb-to-vertical-surface review hypothesis; metric raw-point support throughout.

This explicitly conditions width on a structural-boundary hypothesis. A generic
wall-like surface is not asserted to be a permanent wall, nor to be ground truth.
"""

import sys, math
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree
from .boundary_recovery import supported_at, Policy as PriorPolicy
from .spatial_indexed import _surrounds
from .bottleneck import measure
from .p2_strip_adapter import merge, complement


@dataclass(frozen=True)
class Policy:
    wall_plane_residual_m: float = 0.05
    min_wall_height_m: float = 1.0
    min_wall_span_m: float = 1.0
    min_wall_points: int = 50
    wall_object_veto_fraction: float = 0.65
    max_wall_downward_extension_m: float = 0.5
    max_wall_lateral_ground_gap_m: float = 0.4
    ground_residual_m: float = 0.06
    ground_support_radius_m: float = 0.3
    min_ground_points: int = 3
    max_profile_gap_m: float = 0.5
    hypothesis_trials: int = 128


def fit_floor(record):
    xyz = np.array(
        [
            [*c["center_xy_m"], c["ground_z_m"]]
            for c in record["cells"]
            if c["surface"] == "sidewalk"
        ]
    )
    if len(xyz) < 12:
        raise ValueError("insufficient supported floor cells")
    X = np.c_[xyz[:, :2], np.ones(len(xyz))]
    rng = np.random.default_rng(20261026)
    best = None
    for _ in range(64):
        idx = rng.choice(len(xyz), 3, replace=False)
        if np.linalg.matrix_rank(X[idx]) < 3:
            continue
        a = np.linalg.solve(X[idx], xyz[idx, 2])
        if np.linalg.norm(a[:2]) > math.tan(math.radians(25)):
            continue
        residual = abs(X @ a - xyz[:, 2])
        mask = residual <= 0.05
        rank = (int(mask.sum()), -float(np.median(residual[mask])))
        if best is None or rank > best[0]:
            best = (rank, mask)
    if best is None or best[1].mean() < 0.65:
        raise ValueError("no coherent supported floor plane")
    mask = best[1]
    a = np.linalg.lstsq(X[mask], xyz[mask, 2], rcond=None)[0]
    return (
        a,
        dict(
            coefficients_z_from_s_t=a.tolist(),
            supported_cell_count=len(xyz),
            inlier_cells=int(mask.sum()),
            max_accepted_residual_m=0.05,
            inlier_fraction=float(mask.mean()),
            independent_floor_truth_verified=False,
        ),
    )


def wall_hypotheses(p, labels, scores, height, record, side, curb, policy):
    sign = 1 if side == "right" else -1
    x0, x1, t0, t1 = record["config"]["bounds_xy"]
    curb_t = curb["slope"] * p[:, 0] + curb["intercept"]
    keep = (
        np.isfinite(p).all(1)
        & (p[:, 0] >= x0)
        & (p[:, 0] < x1)
        & (p[:, 1] >= t0)
        & (p[:, 1] < t1)
    )
    keep &= (height > 0.18) & (height < 3.0) & ((p[:, 1] - curb_t) * sign > 0.5)
    ids = np.flatnonzero(keep)
    if len(ids) < policy.min_wall_points:
        return []
    _, index = np.unique(
        np.floor(p[ids] / [0.15, 0.05, 0.15]).astype(np.int64),
        axis=0,
        return_index=True,
    )
    sample_ids = ids[index]
    q = p[sample_ids]
    X = np.c_[q[:, 0], q[:, 2], np.ones(len(q))]
    rng = np.random.default_rng(20261027)
    candidates = []
    seen = []
    for _ in range(policy.hypothesis_trials):
        ix = rng.choice(len(q), 3, replace=False)
        if np.linalg.matrix_rank(X[ix]) < 3:
            continue
        a = np.linalg.solve(X[ix], q[ix, 1])
        if abs(a[0]) > 0.5 or abs(a[1]) > 0.2:
            continue
        residual = abs(X @ a - q[:, 1])
        inliers = residual <= policy.wall_plane_residual_m
        if inliers.sum() < 20:
            continue
        if any((np.max(abs(X[inliers] @ (a - old))) < 0.04 for old in seen)):
            continue
        a = np.linalg.lstsq(X[inliers], q[inliers, 1], rcond=None)[0]
        mask = (
            abs(np.c_[p[ids, 0], p[ids, 2], np.ones(len(ids))] @ a - p[ids, 1])
            <= policy.wall_plane_residual_m
        )
        raw_ids = ids[mask]
        if len(raw_ids) < policy.min_wall_points:
            continue
        h = height[raw_ids]
        s = p[raw_ids, 0]
        if (
            np.quantile(h, 0.95) - np.quantile(h, 0.05) < policy.min_wall_height_m
            or np.ptp(s) < policy.min_wall_span_m
        ):
            continue
        fraction = float(
            np.mean(
                np.isin(labels[raw_ids], [1, 2, 3, 8, 10, 12])
                & (scores[raw_ids] >= 0.7)
            )
        )
        rejection = (
            "recognized_object_veto"
            if fraction >= policy.wall_object_veto_fraction
            else None
        )
        candidates.append(
            dict(
                plane_t_from_s_z=a.tolist(),
                raw_support_ids=list(map(int, raw_ids)),
                support_count=len(raw_ids),
                observed_s_range_m=[float(s.min()), float(s.max())],
                height_quantiles_m=np.quantile(h, [0, 0.05, 0.5, 0.95, 1]).tolist(),
                object_fraction=fraction,
                rejection=rejection,
                physical_identity="persistent_vertical_surface_hypothesis_not_wall_identity",
                physical_boundary_confirmed=False,
            )
        )
        seen.append(a)
    candidates.sort(key=lambda a: (-a["support_count"], a["plane_t_from_s_z"]))
    return candidates[:12]


def wall_station(candidate, s, p, height, policy, floor):
    a, b, c = candidate["plane_t_from_s_z"]
    g, h, k = floor
    denom = 1 - b * h
    if abs(denom) < 0.5:
        return (None, "unstable_plane_intersection")
    t = ((a + b * g) * s + b * k + c) / denom
    ids = np.array(candidate["raw_support_ids"], dtype=int)
    near = ids[abs(p[ids, 0] - s) <= 0.3]
    if len(near) < 8:
        return (None, "sparse_vertical_support")
    if np.ptp(height[near]) < policy.min_wall_height_m:
        return (None, "insufficient_vertical_extent")
    down = float(max(0, height[near].min()))
    if down > policy.max_wall_downward_extension_m:
        return (None, "vertical_surface_has_no_supported_foot")
    return (
        dict(
            t_m=float(t),
            support_raw_ids=list(map(int, near)),
            downward_extension_m=down,
        ),
        None,
    )


def evaluate(record, prev, p, labels, scores, policy=Policy()):
    models = prev["models"]
    known = [side for side in ("left", "right") if models[side] is not None]
    if len(known) != 1:
        return dict(
            state="requires_exactly_one_supported_curb",
            conditional_width_m=None,
            conditional_ratio=None,
            production_eligible=False,
        )
    floor, floor_report = fit_floor(record)
    height = p[:, 2] - np.c_[p[:, :2], np.ones(len(p))] @ floor
    curb_side = known[0]
    missing = "right" if curb_side == "left" else "left"
    curb = models[curb_side]
    hypotheses = wall_hypotheses(
        p, labels, scores, height, record, missing, curb, policy
    )
    x0, x1, t0, t1 = record["config"]["bounds_xy"]
    res = record["config"]["resolution_m"]
    ground_ids = np.flatnonzero(
        np.isfinite(p).all(1) & (abs(height) <= policy.ground_residual_m)
    )
    tree = cKDTree(p[ground_ids, :2]) if len(ground_ids) else None
    elevated = np.flatnonzero(np.isfinite(p).all(1) & (height > 0.18) & (height <= 2.1))
    occupied = {}
    for r in elevated:
        key = (
            int(math.floor((p[r, 0] - x0) / res)),
            int(math.floor((p[r, 1] - t0) / res)),
        )
        occupied.setdefault(key, []).append(int(r))
    candidates = []
    for hypothesis in hypotheses:
        if hypothesis["rejection"]:
            continue
        boundary_ids = {
            r
            for r in hypothesis["raw_support_ids"]
            if not (labels[r] in (1, 2, 3, 8, 10, 12) and scores[r] >= 0.7)
        }
        sections = []
        failures = {}
        floor_support_records = []
        for i in range(record["grid"]["shape"][0]):
            s = x0 + (i + 0.5) * res
            if not supported_at(curb, s, PriorPolicy()):
                continue
            wall, reason = wall_station(hypothesis, s, p, height, policy, floor)
            if wall is None:
                failures[reason] = failures.get(reason, 0) + 1
                continue
            curb_t = curb["slope"] * s + curb["intercept"]
            wall_t = wall["t_m"]
            left, right = (curb_t, wall_t) if curb_side == "left" else (wall_t, curb_t)
            if not t0 <= left < right <= t1:
                continue
            if not 0.5 <= right - left <= 6.0:
                continue
            near_floor = ground_ids[
                (abs(p[ground_ids, 0] - s) <= 0.3)
                & (
                    abs(p[ground_ids, 1] - wall_t)
                    <= policy.max_wall_lateral_ground_gap_m
                )
            ]
            if len(near_floor) < policy.min_ground_points:
                failures["no_floor_near_vertical_foot"] = (
                    failures.get("no_floor_near_vertical_foot", 0) + 1
                )
                continue
            free = []
            blocked = []
            support = []
            unknown = []
            for j in range(record["grid"]["shape"][1]):
                a = max(left, t0 + j * res)
                b = min(right, t0 + (j + 1) * res)
                if b - a <= 1e-08:
                    continue
                q = np.array([s, (a + b) / 2])
                hit = [r for r in occupied.get((i, j), []) if r not in boundary_ids]
                if hit:
                    blocked.append([a, b])
                    support.append(
                        dict(index=[i, j], kind="occupied_raw_return", raw_ids=hit)
                    )
                    continue
                local = (
                    ground_ids[tree.query_ball_point(q, policy.ground_support_radius_m)]
                    if tree is not None
                    else np.array([], int)
                )
                if len(local) >= policy.min_ground_points and _surrounds(
                    p[local, :2], q
                ):
                    free.append([a, b])
                    support.append(
                        dict(
                            index=[i, j],
                            kind="locally_supported_ground_candidate",
                            raw_ids=list(map(int, local)),
                        )
                    )
                else:
                    unknown.append([a, b])
            sections.append(
                dict(
                    station_index=i,
                    s_m=s,
                    structural_span_m=[float(left), float(right)],
                    boundary_supported=True,
                    candidate_evidence_supported=bool(free or blocked),
                    free_intervals_m=merge(free),
                    unknown_intervals_m=merge(unknown),
                    occupied_candidate_intervals_m=merge(blocked),
                    vertical_boundary_evidence=wall,
                    cell_support=support,
                )
            )
        runs = []
        current = []
        excluded_stations = []
        for row in sections:
            if row["unknown_intervals_m"]:
                excluded_stations.append(
                    dict(
                        station_index=row["station_index"],
                        reason="unknown_prevents_exact_relative_width",
                        unknown_intervals_m=row["unknown_intervals_m"],
                    )
                )
                if current:
                    runs.append(current)
                    current = []
                continue
            if current and row["station_index"] != current[-1]["station_index"] + 1:
                runs.append(current)
                current = []
            current.append(row)
        if current:
            runs.append(current)
        measured = []
        for rows in runs:
            if len(rows) < 5:
                continue
            payload = dict(
                schema_version="p3_width_draft_v1",
                sample_id=record["metadata"]["frame_id"],
                scope="partial",
                upstream_contract_version="raw_supported_ground_vertical_boundary_hypothesis_v4",
                frame_kind="common_straight_strip_st",
                station_spacing_m=res,
                evidence_mode="surface_candidate_review",
                sections=rows,
            )
            result = measure(payload)
            if result["minimum_passable_ratio"] == 0 and any(
                (row["unknown_intervals_m"] for row in rows)
            ):
                for key in (
                    "minimum_passable_ratio",
                    "bottleneck_width_m",
                    "same_section_full_width_m",
                    "raw_minimum_ratio_on_selected_path",
                ):
                    result[key] = None
                result.update(
                    authority="none", selected_corridor_path=[], bottleneck=None
                )
                result["reasons"].append("candidate_disconnected_with_unknown_space")
            measured.append(
                dict(
                    station_range=[rows[0]["station_index"], rows[-1]["station_index"]],
                    result=result,
                )
            )
        candidates.append(
            dict(
                hypothesis=hypothesis,
                sections=sections,
                runs=measured,
                failures=failures,
                excluded_stations=excluded_stations,
                complete_stations=len(sections) - len(excluded_stations),
                boundary_surface_excluded_from_occupancy_ids=sorted(boundary_ids),
            )
        )
    chosen = candidates[0] if candidates else None
    numeric = (
        [r for r in chosen["runs"] if r["result"]["minimum_passable_ratio"] is not None]
        if chosen
        else []
    )
    return dict(
        schema_version="roadpush.curb_vertical_hypothesis/0.5-draft",
        frame_id=record["metadata"]["frame_id"],
        state=(
            "conditional_partial_numeric"
            if numeric
            else "conditional_measurement_unavailable"
        ),
        policy=asdict(policy),
        floor=floor_report,
        curb_side=curb_side,
        curb=curb,
        vertical_side=missing,
        vertical_hypotheses=hypotheses,
        evaluated_hypotheses=candidates,
        chosen_hypothesis_index=0 if chosen else None,
        confirmed_width_m=None,
        confirmed_ratio=None,
        conditional_runs=numeric,
        production_eligible=False,
        human_review_required=True,
        permanent_wall_identity_verified=False,
        ground_truth_width_verified=False,
        safety_verdict=None,
        raw_points=len(p),
        model_predictions_changed=False,
        reference_labels_used=False,
        warning="Conditional on vertical surface being a structural boundary. All raw IDs/evidence retained; unknown is not free.",
    )
