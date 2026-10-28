"""Evidence-first boundary recovery; never turn an observation frontier into width.

Own research prototype inspired by team observation/fit/scope separation and
LiDAR local elevation/planarity literature. Does not modify P2 or model outputs.
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from copy import deepcopy
import math, sys
import numpy as np
from .spatial_indexed import Config, _plane
from .bottleneck import measure
from .p2_strip_adapter import merge, complement


@dataclass(frozen=True)
class Policy:
    search_m: float = 0.4
    max_step_m: float = 0.35
    fit_residual_m: float = 0.15
    min_anchor_stations: int = 6
    min_anchor_span_m: float = 0.5
    min_inlier_fraction: float = 0.65
    max_interpolation_gap_m: float = 0.5
    min_run_stations: int = 5


def robust_fit(anchors, policy):
    """Deterministic consensus + least-squares refit; station-balanced votes."""
    if len(anchors) < policy.min_anchor_stations:
        return None
    groups = {}
    for a in anchors:
        groups.setdefault(a["station_index"], []).append(a)
    values = [
        min(v, key=lambda a: (a["uncertainty_m"], a["t_m"]))
        for _, v in sorted(groups.items())
    ]
    s = np.array([a["s_m"] for a in values])
    t = np.array([a["t_m"] for a in values])
    if len(s) < policy.min_anchor_stations or np.ptp(s) < policy.min_anchor_span_m:
        return None
    idx = np.unique(np.linspace(0, len(s) - 1, min(30, len(s)), dtype=int))
    best = None
    for pos, i in enumerate(idx[:-1]):
        for j in idx[pos + 1 :]:
            if s[i] == s[j]:
                continue
            slope = (t[j] - t[i]) / (s[j] - s[i])
            intercept = t[i] - slope * s[i]
            residual = abs(t - (slope * s + intercept))
            mask = residual <= policy.fit_residual_m
            rank = (int(mask.sum()), -float(np.median(residual[mask])), -abs(slope))
            if best is None or rank > best[0]:
                best = (rank, mask)
    if best is None or best[1].sum() < policy.min_anchor_stations:
        return None
    mask = best[1]
    for _ in range(2):
        slope, intercept = np.polyfit(s[mask], t[mask], 1)
        mask = abs(t - (slope * s + intercept)) <= policy.fit_residual_m
        if mask.sum() < policy.min_anchor_stations:
            return None
    if (
        mask.mean() < policy.min_inlier_fraction
        or np.ptp(s[mask]) < policy.min_anchor_span_m
    ):
        return None
    return dict(
        slope=float(slope),
        intercept=float(intercept),
        inlier_fraction=float(mask.mean()),
        observed_s_range_m=[float(s[mask].min()), float(s[mask].max())],
        anchors=[a for a, m in zip(values, mask) if m],
        rejected_anchors=[a for a, m in zip(values, mask) if not m],
        max_residual_m=float(abs(t[mask] - (slope * s[mask] + intercept)).max()),
        physical_boundary_confirmed=False,
        method="station_balanced_consensus_then_least_squares",
    )


def supported_at(model, s, policy):
    if model is None:
        return False
    values = sorted({a["s_m"] for a in model["anchors"]})
    if s < values[0] - 1e-09 or s > values[-1] + 1e-09:
        return False
    if any((abs(v - s) < 1e-09 for v in values)):
        return True
    left = max((v for v in values if v < s), default=None)
    right = min((v for v in values if v > s), default=None)
    return (
        left is not None
        and right is not None
        and (right - left <= policy.max_interpolation_gap_m + 1e-09)
    )


def recover(record, raw, labels, scores, policy=Policy()):
    """Join raw XYZ to existing strip transform; keep all original arrays intact."""
    if len(raw) > 150000 or len(labels) != len(raw) or len(scores) != len(raw):
        raise ValueError("raw row contract")
    transform = record["metadata"]["raw_to_processing_xy_transform"]
    p = np.asarray(raw, dtype=float).copy()
    p[:, :2] = (p[:, :2] - np.array(transform["origin_xy_m"])) @ np.array(
        transform["source_xy_to_strip_matrix"]
    ).T
    cfg = Config(**record["config"])
    res = cfg.resolution_m
    x0, x1, t0, t1 = cfg.bounds_xy
    cells = {tuple(c["index"]): c for c in record["cells"]}
    anchors = {"left": [], "right": []}
    frontier_count = 0
    rng = np.random.default_rng(20261026)
    finite = np.isfinite(p).all(1)
    inside = (
        finite & (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= t0) & (p[:, 1] < t1)
    )
    buckets = {}
    for r in np.flatnonzero(inside):
        i = int(math.floor((p[r, 0] - x0) / res))
        buckets.setdefault(i, []).append(int(r))
    station_count = record["grid"]["shape"][0]
    observations = []
    for i in range(station_count):
        row = [
            c for (ci, j), c in cells.items() if ci == i and c["surface"] == "sidewalk"
        ]
        if not row:
            continue
        s = x0 + (i + 0.5) * res
        for side, c in [
            ("left", min(row, key=lambda c: c["index"][1])),
            ("right", max(row, key=lambda c: c["index"][1])),
        ]:
            sign = -1 if side == "left" else 1
            edge = t0 + (c["index"][1] + (1 if sign == 1 else 0)) * res
            observed = dict(
                station_index=i,
                s_m=s,
                t_m=edge,
                side=side,
                kind="observation_frontier",
                support_point_ids=c["support_point_ids"],
            )
            observations.append(observed)
            frontier_count += 1
            if edge <= t0 + res / 2 or edge >= t1 - res / 2:
                continue
            rows = np.array(
                [r for k in range(i - 4, i + 5) for r in buckets.get(k, [])], dtype=int
            )
            if not len(rows):
                continue
            delta = (p[rows, 1] - edge) * sign
            rows = rows[
                (delta >= 0)
                & (delta <= policy.search_m)
                & (abs(p[rows, 0] - s) <= cfg.support_radius_m)
            ]
            if len(rows) < cfg.min_support:
                continue
            n = np.array(c["plane"]["normal"])
            d = c["plane"]["d"]
            floor = -(p[rows, :2] @ n[:2] + d) / n[2]
            height = p[rows, 2] - floor
            close = rows[
                (abs(height) <= policy.max_step_m)
                & ~np.isin(labels[rows], [1, 2, 3, 8, 10, 12])
            ]
            if len(close) < cfg.min_support:
                continue
            road = close[labels[close] == 40]
            candidates = [("raw_road_geometry_candidate", road)]
            candidates.append(("raw_height_step_candidate", close))
            for kind, rr in candidates:
                if len(rr) < cfg.min_support:
                    continue
                q = p[rr, :2].mean(0)
                fit = _plane(p[rr], q, cfg, rng)
                if fit is None:
                    continue
                nn, dd, inliers, rms = fit
                support = rr[inliers]
                predicted = -(p[support, :2] @ n[:2] + d) / n[2]
                dz = float(np.median(p[support, 2] - predicted))
                if kind == "raw_height_step_candidate" and (
                    not cfg.component_step_m < abs(dz) <= policy.max_step_m
                ):
                    continue
                near = float(np.quantile((p[support, 1] - edge) * sign, 0.1))
                uncertainty = max(res / 2, near / 2)
                a = dict(
                    station_index=i,
                    s_m=s,
                    t_m=float(edge + sign * near / 2),
                    side=side,
                    kind=kind,
                    uncertainty_m=uncertainty,
                    height_difference_m=dz,
                    outer_plane_rms_m=float(rms),
                    support_point_ids=sorted(
                        set(c["support_point_ids"]) | set(map(int, support))
                    ),
                    outer_support_point_ids=list(map(int, support)),
                    physical_boundary_confirmed=False,
                    semantic_confidence_kind="argmax_context_only_not_final_class_acceptance",
                    outer_score_mean=(
                        float(np.nanmean(scores[support]))
                        if np.isfinite(scores[support]).any()
                        else None
                    ),
                )
                anchors[side].append(a)
                break
    models = {side: robust_fit(values, policy) for side, values in anchors.items()}
    sections = []
    for i in range(station_count):
        s = x0 + (i + 0.5) * res
        row = dict(
            station_index=i,
            s_m=s,
            structural_span_m=None,
            boundary_supported=False,
            candidate_evidence_supported=False,
            free_intervals_m=None,
            unknown_intervals_m=[],
        )
        if all((supported_at(models[side], s, policy) for side in ("left", "right"))):
            limits = [
                models[side]["slope"] * s + models[side]["intercept"]
                for side in ("left", "right")
            ]
            if t0 <= limits[0] < limits[1] <= t1:
                candidate = []
                occupied = []
                source = []
                for j in range(record["grid"]["shape"][1]):
                    c = cells[i, j]
                    a = max(limits[0], t0 + j * res)
                    b = min(limits[1], t0 + (j + 1) * res)
                    if a >= b:
                        continue
                    source.append([i, j])
                    unresolved = c["height_unresolved_point_ids"] or set(
                        c["out_of_height_band_point_ids"]
                    ) - set(c["low_nonblocking_point_ids"])
                    if c["occupancy_candidate_point_ids"]:
                        occupied.append([a, b])
                    elif c["surface"] == "sidewalk" and (not unresolved):
                        candidate.append([a, b])
                row.update(
                    structural_span_m=list(map(float, limits)),
                    boundary_supported=True,
                    candidate_evidence_supported=bool(candidate or occupied),
                    free_intervals_m=merge(candidate),
                    unknown_intervals_m=complement(limits, candidate + occupied),
                    occupied_candidate_intervals_m=merge(occupied),
                    source_cell_indices=source,
                )
        sections.append(row)
    runs = []
    current = []
    for row in sections:
        if row["boundary_supported"] and row["candidate_evidence_supported"]:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    outputs = []
    for rows in runs:
        if len(rows) < policy.min_run_stations:
            continue
        payload = dict(
            schema_version="p3_width_draft_v1",
            sample_id=record["metadata"]["frame_id"],
            scope="partial",
            upstream_contract_version=record["schema_version"],
            frame_kind="common_straight_strip_st",
            station_spacing_m=res,
            evidence_mode="surface_candidate_review",
            sections=rows,
        )
        result = measure(payload)
        if result["minimum_passable_ratio"] == 0 and any(
            (r["unknown_intervals_m"] for r in rows)
        ):
            for key in (
                "minimum_passable_ratio",
                "bottleneck_width_m",
                "same_section_full_width_m",
                "raw_minimum_ratio_on_selected_path",
            ):
                result[key] = None
            result.update(authority="none", selected_corridor_path=[], bottleneck=None)
            result["reasons"].append("candidate_disconnected_with_unknown_space")
        outputs.append(
            dict(
                station_range=[rows[0]["station_index"], rows[-1]["station_index"]],
                result=result,
            )
        )
    return dict(
        schema_version="roadpush.boundary_recovery/0.3-draft",
        frame_id=record["metadata"]["frame_id"],
        policy=asdict(policy),
        models=models,
        anchors=anchors,
        observations=observations,
        sections=sections,
        runs=outputs,
        diagnostic=dict(
            frontier_observations=frontier_count,
            anchors={k: len(v) for k, v in anchors.items()},
            bilateral_models=all(models.values()),
            boundary_supported_stations=sum(
                (r["boundary_supported"] for r in sections)
            ),
            raw_points=len(raw),
            raw_ids_removed=0,
            old_p2_unchanged=True,
        ),
        frame_width_m=None,
        frame_relative_width=None,
        production_eligible=False,
        human_review_required=True,
        scope="local_partial_runs_only_no_frame_aggregate",
        reference_labels_used=False,
        caveat="Raw argmax context plus local geometry produces review candidates, not confirmed physical boundaries. Unknown remains unknown.",
    )
