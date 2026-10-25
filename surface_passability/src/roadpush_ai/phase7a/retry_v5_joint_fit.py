from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import roadpush_ai.phase7a.retry_v4 as v4
from roadpush_ai.phase7a.retry_v5_global_support import (
    _candidate_score,
    _raw_row_fit_passes,
    _v4_bands_and_runs,
)
from roadpush_ai.phase7a.retry_v5_runwise import (
    RunwiseDirectionPolicy,
    apply_runwise_direction_consistency,
    diagnose_runwise_direction_consistency,
)
from roadpush_ai.phase7a.retry_v5_direction import (
    _selected_bands,
)
from roadpush_ai.phase7a.retry_v5_support_window import (
    GeometricGapPolicy,
    fit_trapezoid_boundary_retry_v5_probe_g,
)


POLICY_ID = "phase7_trapezoid_boundary_retry_v5_probe_g_then_b_then_c_core"


@dataclass(frozen=True)
class ContinuousCorePolicy:
    """Continuous-core priority and conservative disconnected-support policy."""

    minimum_core_bands: int = 6
    eligible_run_classifications: tuple[str, ...] = ("consistent",)


def _support_for_centers(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: GeometricGapPolicy,
    selected_centers: set[int],
    *,
    allow_explained_gaps: bool = False,
):
    """Refit C-ranked observed support through unchanged V5 gates."""

    if not records:
        return None, [], []
    _, width = shape
    all_bands, runs = _v4_bands_and_runs(records, policy)
    selected = [
        band for band in all_bands if int(band["center_y"]) in selected_centers
    ]
    actual_centers = {int(band["center_y"]) for band in selected}
    if actual_centers != selected_centers or len(selected) < 3:
        return None, all_bands, runs
    has_gap = any(
        int(second["center_y"]) - int(first["center_y"]) != policy.band_rows
        for first, second in zip(selected, selected[1:])
    )
    if has_gap and not allow_explained_gaps:
        return None, all_bands, runs

    left, right = v4._fit_bands(selected, width, policy)
    records_by_y = {row["y"]: row for row in records}
    if (
        not v4._fit_passes(left, right, width, policy)
        or not _raw_row_fit_passes(selected, records_by_y, width, policy)
    ):
        return None, all_bands, runs

    passing_y = sorted({y for band in selected for y in band["passing_y"]})
    median_span = float(np.median([band["span"] for band in selected]))
    centers = [int(band["center_y"]) for band in selected]
    gap_count = sum(
        max(
            0,
            (int(second["center_y"]) - int(first["center_y"]))
            // policy.band_rows
            - 1,
        )
        for first, second in zip(selected, selected[1:])
    )
    return (
        {
            "bands": selected,
            "passing_y": passing_y,
            "median_span": median_span,
            "median_span_fraction": float(median_span / max(1, width)),
            "segment_span_gate_passed": (
                median_span / max(1, width)
                >= policy.minimum_segment_median_span_fraction
            ),
            "near_cap_centers": centers[-policy.minimum_clean_cap_bands :],
            "far_cap_center": centers[0],
            "internal_gap_band_count": int(gap_count),
            "extension_stop_reason": "probe_c_ranked_continuous_core_support",
        },
        all_bands,
        runs,
    )


def _refit_core(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    centers: list[int],
    g_policy: GeometricGapPolicy,
    *,
    allow_explained_gaps: bool = False,
):
    selected = set(int(center) for center in centers)

    def selector(records, shape, policy):
        return _support_for_centers(
            records,
            shape,
            policy,
            selected,
            allow_explained_gaps=allow_explained_gaps,
        )

    result, masks = v4.fit_trapezoid_boundary_retry_v4(
        *args,
        policy=g_policy,
        _support_selector=selector,
        **kwargs,
    )
    return result, masks


def _gap_is_explained(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    rows_by_y: dict[int, dict[str, Any]],
    policy: GeometricGapPolicy,
) -> bool:
    left, right = sorted(
        (first, second), key=lambda run: int(run[0]["center_y"])
    )
    start = max(int(y) for y in left[-1]["passing_y"]) + 1
    stop = min(int(y) for y in right[0]["passing_y"])
    gap_rows = [rows_by_y[y] for y in range(start, stop) if y in rows_by_y]
    if not gap_rows or len(gap_rows) != max(0, stop - start):
        return False
    allowed_fragments = (
        "blocker_contact",
        "obstacle_silhouette_contact",
        "frame_contact",
    )
    nonclean = [row for row in gap_rows if not row.get("clean_bilateral")]
    if not nonclean or not all(
        any(
            fragment in reason
            for reason in row.get("reasons", [])
            for fragment in allowed_fragments
        )
        for row in nonclean
    ):
        return False
    clean_run = 0
    for row in gap_rows:
        clean_run = clean_run + 1 if row.get("clean_bilateral") else 0
        if clean_run >= policy.minimum_passing_rows_per_band:
            return False
    return True


def _run_matches_core_line(
    core: list[dict[str, Any]],
    remote: list[dict[str, Any]],
    width: int,
    policy: GeometricGapPolicy,
) -> bool:
    left, right = v4._fit_bands(core, width, policy)
    if not left.get("valid") or not right.get("valid"):
        return False
    tolerance = policy.fit_inlier_tolerance_fraction * width
    for band in remote:
        y = float(band["center_y"])
        if (
            abs(float(band["left"]) - (left["slope"] * y + left["intercept"]))
            > tolerance
            or abs(
                float(band["right"])
                - (right["slope"] * y + right["intercept"])
            )
            > tolerance
        ):
            return False
    return True


def _gap_is_probe_g_demoted(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    selected_geometric_gap_centers: set[int],
    policy: GeometricGapPolicy,
) -> bool:
    left, right = sorted(
        (first, second), key=lambda run: int(run[0]["center_y"])
    )
    expected = set(
        range(
            int(left[-1]["center_y"]) + policy.band_rows,
            int(right[0]["center_y"]),
            policy.band_rows,
        )
    )
    return bool(expected) and expected.issubset(selected_geometric_gap_centers)


def _continuous_band_runs(
    bands: list[dict[str, Any]], band_rows: int
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    for band in bands:
        if (
            not runs
            or int(band["center_y"]) - int(runs[-1][-1]["center_y"])
            != band_rows
        ):
            runs.append([band])
        else:
            runs[-1].append(band)
    return runs


def _rank_all_candidates_by_continuous_core(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    policy: GeometricGapPolicy,
    safe_geometric_gap_centers: set[int],
) -> dict[str, Any] | None:
    """Re-rank every valid G/V5 window by its normalized continuous core.

    Probe G-demoted bands remain non-anchor gaps.  Each otherwise-valid global
    window is normalized to its best continuous run, and that run is scored by
    the existing V5 order: band count, unique passing rows, combined p90, then
    combined median residual.  No new tie-break criterion is introduced.
    """

    if not records:
        return None
    _, width = shape
    all_bands, _ = _v4_bands_and_runs(records, policy)
    records_by_y = {int(row["y"]): row for row in records}
    valid_window_count = 0
    normalized_by_centers: dict[tuple[int, ...], dict[str, Any]] = {}
    core_score_cache: dict[tuple[int, ...], tuple[float, ...] | None] = {}
    for start in range(len(all_bands)):
        for stop in range(
            start + policy.minimum_global_candidate_bands,
            len(all_bands) + 1,
        ):
            observed_window = all_bands[start:stop]
            anchor_window = [
                band
                for band in observed_window
                if int(band["center_y"]) not in safe_geometric_gap_centers
            ]
            if len(anchor_window) < policy.minimum_global_candidate_bands:
                continue
            left, right = v4._fit_bands(anchor_window, width, policy)
            if not v4._fit_passes(left, right, width, policy):
                continue
            valid_window_count += 1
            eligible_cores = []
            for run in _continuous_band_runs(anchor_window, policy.band_rows):
                if len(run) < policy.minimum_global_candidate_bands:
                    continue
                core_centers = tuple(
                    int(band["center_y"]) for band in run
                )
                if core_centers not in core_score_cache:
                    core_left, core_right = v4._fit_bands(
                        run, width, policy
                    )
                    core_score_cache[core_centers] = (
                        _candidate_score(run, core_left, core_right)
                        if v4._fit_passes(
                            core_left, core_right, width, policy
                        )
                        else None
                    )
                core_score = core_score_cache[core_centers]
                if core_score is None:
                    continue
                eligible_cores.append(
                    {
                        "bands": run,
                        "score": core_score,
                    }
                )
            if not eligible_cores:
                continue
            normalized = max(
                eligible_cores, key=lambda candidate: candidate["score"]
            )
            centers = tuple(
                int(band["center_y"]) for band in normalized["bands"]
            )
            item = normalized_by_centers.setdefault(
                centers,
                {
                    "band_centers": list(centers),
                    "score": [float(value) for value in normalized["score"]],
                    "source_window_count": 0,
                    "source_window_examples": [],
                },
            )
            item["source_window_count"] += 1
            if len(item["source_window_examples"]) < 3:
                item["source_window_examples"].append(
                    [int(band["center_y"]) for band in anchor_window]
                )

    normalized_candidates = sorted(
        normalized_by_centers.values(),
        key=lambda candidate: tuple(candidate["score"]),
        reverse=True,
    )
    raw_rejected = 0
    best = None
    band_by_center = {
        int(band["center_y"]): band for band in all_bands
    }
    for candidate in normalized_candidates:
        centers = tuple(candidate["band_centers"])
        core_bands = [
            band_by_center[center]
            for center in centers
            if center in band_by_center
        ]
        if len(core_bands) == len(centers) and _raw_row_fit_passes(
            core_bands, records_by_y, width, policy
        ):
            best = candidate
            break
        raw_rejected += 1
    if best is None:
        return {
            "status": "no_valid_normalized_candidate",
            "valid_global_window_count": valid_window_count,
            "normalized_candidates": normalized_candidates,
            "selected_core_band_centers": [],
            "selected_score": [],
            "ranked_normalized_candidates_raw_rejected": raw_rejected,
        }
    return {
        "status": "ranked",
        "valid_global_window_count": valid_window_count,
        "normalized_candidates": normalized_candidates,
        "selected_core_band_centers": list(best["band_centers"]),
        "selected_score": list(best["score"]),
        "ranked_normalized_candidates_raw_rejected": raw_rejected,
        "ranking_order": [
            "continuous_core_band_count",
            "unique_passing_row_count",
            "combined_p90_residual_ascending",
            "combined_median_residual_ascending",
        ],
    }


def _rank_continuous_core_support(
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    width: int,
    core_policy: ContinuousCorePolicy,
    g_policy: GeometricGapPolicy,
) -> dict[str, Any] | None:
    bands = _selected_bands(result)
    by_center = {int(band["center_y"]): band for band in bands}
    run_bands = [
        [by_center[center] for center in run["band_centers"] if center in by_center]
        for run in diagnostics["runs"]
    ]
    hard_failure = diagnostics["gate_would_reject"]
    candidates = []
    for run, observed in zip(diagnostics["runs"], run_bands):
        if hard_failure:
            eligible = (
                run["classification"] in core_policy.eligible_run_classifications
                and run["band_count"] >= core_policy.minimum_core_bands
            )
        else:
            eligible = (
                run["classification"] in {"consistent", "insufficient"}
                and run["band_count"] >= 3
            )
        if not eligible or len(observed) != run["band_count"]:
            continue
        left, right = v4._fit_bands(observed, width, g_policy)
        if not v4._fit_passes(left, right, width, g_policy):
            continue
        score = (
            1 if run["classification"] == "consistent" else 0,
            run["band_count"],
            sum(len(band["passing_y"]) for band in observed),
            run["band_centers"][-1] - run["band_centers"][0],
            -left["p90_residual_px"] - right["p90_residual_px"],
        )
        candidates.append((score, run, observed))
    if not candidates:
        return None

    _, core_run, core_bands = max(candidates, key=lambda item: item[0])
    rows_by_y = {int(row["y"]): row for row in result.get("row_evidence", [])}
    probe_g_gap_centers = set(
        result.get("probe_g_geometric_gap", {}).get(
            "selected_geometric_gap_band_centers", []
        )
    )
    attached_indices = []
    attached_gap_sources: dict[int, str] = {}
    selected_centers = list(core_run["band_centers"])
    for run, remote in zip(diagnostics["runs"], run_bands):
        if run["run_index"] == core_run["run_index"]:
            continue
        eligible_remote = (
            run["classification"] in {"consistent", "insufficient"}
            and run["band_count"] >= 3
            and len(remote) == run["band_count"]
        )
        gap_source = (
            "probe_g_safe_geometric_detour"
            if _gap_is_probe_g_demoted(
                core_bands, remote, probe_g_gap_centers, g_policy
            )
            else "observed_contact_gap"
            if _gap_is_explained(core_bands, remote, rows_by_y, g_policy)
            else None
        )
        line_contract_passed = bool(gap_source) and (
            gap_source == "probe_g_safe_geometric_detour"
            or _run_matches_core_line(core_bands, remote, width, g_policy)
        )
        if eligible_remote and line_contract_passed:
            selected_centers.extend(run["band_centers"])
            attached_indices.append(run["run_index"])
            attached_gap_sources[run["run_index"]] = gap_source

    selected_centers = sorted(set(selected_centers))
    original_centers = diagnostics.get("selected_band_centers", [])
    return {
        "core_run_index": core_run["run_index"],
        "core_band_centers": list(core_run["band_centers"]),
        "attached_disconnected_run_indices": attached_indices,
        "attached_disconnected_gap_sources": attached_gap_sources,
        "selected_band_centers": selected_centers,
        "excluded_band_centers": sorted(
            set(original_centers) - set(selected_centers)
        ),
        "selection_changed": selected_centers != original_centers,
        "explained_gap_support_attached": bool(attached_indices),
    }


def fit_trapezoid_boundary_retry_v5_probe_c(
    *args: Any,
    core_policy: ContinuousCorePolicy | None = None,
    direction_policy: RunwiseDirectionPolicy | None = None,
    geometric_gap_policy: GeometricGapPolicy | None = None,
    **kwargs: Any,
):
    """Apply G and B-aware continuous-core/disconnected-support ranking.

    A multi-run B failure may preserve one strong consistent core. For a
    passing multi-run candidate, continuous evidence ranks before disconnected
    evidence; a remote run is retained only when it has at least three bands,
    the intervening gap is explained by contact, and both boundaries agree
    with the core line. The final support is always refitted through unchanged
    V5 fit/raw-row gates and Probe B.
    """

    core_policy = core_policy or ContinuousCorePolicy()
    direction_policy = direction_policy or RunwiseDirectionPolicy()
    g_policy = geometric_gap_policy or GeometricGapPolicy()
    g_result, g_masks = fit_trapezoid_boundary_retry_v5_probe_g(
        *args, policy=g_policy, **kwargs
    )
    pre_c_g_centers = [
        int(band["center_y"]) for band in _selected_bands(g_result)
    ]
    global_ranking = None
    global_candidate_reselected = False
    if len(args) >= 4 and args[2] is not None and args[3] is not None:
        structural = np.asarray(args[2], dtype=bool)
        blocker = np.asarray(args[3], dtype=bool)
        silhouette = np.asarray(
            kwargs.get(
                "obstacle_silhouette_mask", np.zeros_like(structural)
            ),
            dtype=bool,
        )
        records = v4._records(structural, blocker, silhouette, g_policy)
        safe_gap_centers = set(
            g_result.get("probe_g_geometric_gap", {}).get(
                "safe_gap_band_centers", []
            )
        )
        global_ranking = _rank_all_candidates_by_continuous_core(
            records,
            structural.shape,
            g_policy,
            safe_gap_centers,
        )
        winning_centers = (
            global_ranking.get("selected_core_band_centers", [])
            if global_ranking
            else []
        )
        if winning_centers and not set(winning_centers).issubset(
            pre_c_g_centers
        ):
            reranked, reranked_masks = _refit_core(
                args, kwargs, winning_centers, g_policy
            )
            if reranked.get("envelope_fit", {}).get("status") == "accepted":
                reranked["probe_g_geometric_gap"] = deepcopy(
                    g_result["probe_g_geometric_gap"]
                )
                reranked["probe_g_contract"] = deepcopy(
                    g_result["probe_g_contract"]
                )
                g_result, g_masks = reranked, reranked_masks
                global_candidate_reselected = True
    original_b = diagnose_runwise_direction_consistency(g_result, direction_policy)
    base_diagnostics = {
        "status": "not_needed",
        "original_probe_b_classification": original_b["classification"],
        "original_probe_b_gate_would_reject": original_b["gate_would_reject"],
        "original_selected_band_centers": original_b.get(
            "selected_band_centers", []
        ),
        "candidate_runs": [],
        "pre_c_probe_g_selected_band_centers": pre_c_g_centers,
        "global_candidate_ranking": global_ranking,
        "global_candidate_reselected": global_candidate_reselected,
        "selected_original_run_index": None,
        "selected_core_band_centers": [],
        "discarded_band_centers": [],
        "attached_disconnected_run_indices": [],
        "attached_disconnected_gap_sources": {},
        "ranked_selected_band_centers": [],
        "thresholds": asdict(core_policy),
        "continuous_core_ranked_before_disconnected_support": True,
        "minimum_disconnected_support_bands": 3,
        "explicit_gap_provenance_required_for_attachment": True,
        "cross_gap_support_joined": False,
        "cross_gap_support_used_as_curvature": False,
        "extension_added": False,
        "curve_model_added": False,
    }

    if len(original_b["runs"]) < 2 or len(args) < 3 or args[2] is None:
        output, masks = apply_runwise_direction_consistency(
            g_result,
            g_masks,
            mode="candidate_gate",
            direction_policy=direction_policy,
        )
        output["probe_c_continuous_core"] = base_diagnostics
        if global_candidate_reselected:
            base_diagnostics["status"] = "global_candidate_reselected"
        output["probe_c_contract"] = {
            "mode": "candidate",
            "multi_run_stable_core_salvage": True,
            "continuous_core_disconnected_support_ranking": True,
            "all_valid_global_candidates_reranked": True,
            "existing_v5_tiebreak_reused": True,
            "minimum_core_bands": core_policy.minimum_core_bands,
            "eligible_run_classifications": list(
                core_policy.eligible_run_classifications
            ),
            "v5_fit_and_raw_row_gates_unchanged": True,
            "blocker_logic_changed": False,
        }
        return output, masks

    ranking = _rank_continuous_core_support(
        g_result,
        original_b,
        int(np.asarray(args[2]).shape[1]),
        core_policy,
        g_policy,
    )
    for run in original_b["runs"]:
        base_diagnostics["candidate_runs"].append(
            {
                "original_run_index": run["run_index"],
                "band_centers": run["band_centers"],
                "band_count": run["band_count"],
                "original_classification": run["classification"],
                "selected_as_core": bool(
                    ranking and run["run_index"] == ranking["core_run_index"]
                ),
                "attached_as_compatible_disconnected_support": bool(
                    ranking
                    and run["run_index"]
                    in ranking["attached_disconnected_run_indices"]
                ),
                "attached_gap_source": (
                    ranking["attached_disconnected_gap_sources"].get(
                        run["run_index"]
                    )
                    if ranking
                    else None
                ),
            }
        )

    if ranking is None:
        output, masks = apply_runwise_direction_consistency(
            g_result,
            g_masks,
            mode="candidate_gate",
            direction_policy=direction_policy,
        )
        base_diagnostics["status"] = "no_safe_continuous_core"
    else:
        refit, refit_masks = _refit_core(
            args,
            kwargs,
            ranking["selected_band_centers"],
            g_policy,
            allow_explained_gaps=ranking["explained_gap_support_attached"],
        )
        refit_b = diagnose_runwise_direction_consistency(refit, direction_policy)
        if (
            refit["envelope_fit"]["status"] != "accepted"
            or refit_b["gate_would_reject"]
        ):
            output, masks = apply_runwise_direction_consistency(
                g_result,
                g_masks,
                mode="candidate_gate",
                direction_policy=direction_policy,
            )
            base_diagnostics["status"] = "ranked_support_refit_failed"
        else:
            refit["policy_id"] = POLICY_ID
            refit["probe_g_geometric_gap"] = deepcopy(
                g_result["probe_g_geometric_gap"]
            )
            refit["probe_g_contract"] = deepcopy(g_result["probe_g_contract"])
            output, masks = apply_runwise_direction_consistency(
                refit,
                refit_masks,
                mode="candidate_gate",
                direction_policy=direction_policy,
            )
            if original_b["gate_would_reject"]:
                base_diagnostics["status"] = (
                    "salvaged_consistent_continuous_core"
                )
            elif ranking["selection_changed"]:
                base_diagnostics["status"] = (
                    "continuous_core_ranked_over_disconnected_support"
                )
            elif ranking["explained_gap_support_attached"]:
                base_diagnostics["status"] = (
                    "compatible_disconnected_support_retained"
                )
            else:
                base_diagnostics["status"] = "ranking_unchanged"
        base_diagnostics["selected_original_run_index"] = ranking[
            "core_run_index"
        ]
        base_diagnostics["selected_core_band_centers"] = ranking[
            "core_band_centers"
        ]
        base_diagnostics["ranked_selected_band_centers"] = ranking[
            "selected_band_centers"
        ]
        base_diagnostics["discarded_band_centers"] = ranking[
            "excluded_band_centers"
        ]
        base_diagnostics["attached_disconnected_run_indices"] = ranking[
            "attached_disconnected_run_indices"
        ]
        base_diagnostics["attached_disconnected_gap_sources"] = ranking[
            "attached_disconnected_gap_sources"
        ]
        base_diagnostics["cross_gap_support_joined"] = ranking[
            "explained_gap_support_attached"
        ]

    output["policy_id"] = POLICY_ID
    output["probe_c_continuous_core"] = base_diagnostics
    output["probe_c_contract"] = {
        "mode": "candidate",
        "multi_run_stable_core_salvage": True,
        "continuous_core_disconnected_support_ranking": True,
        "all_valid_global_candidates_reranked": True,
        "existing_v5_tiebreak_reused": True,
        "minimum_core_bands": core_policy.minimum_core_bands,
        "eligible_run_classifications": list(
            core_policy.eligible_run_classifications
        ),
        "v5_fit_and_raw_row_gates_unchanged": True,
        "remote_support_requires_three_bands": True,
        "remote_support_requires_explicit_gap_provenance": True,
        "probe_g_safe_geometric_detour_is_explained_gap": True,
        "observed_contact_support_requires_core_line_compatibility": True,
        "probe_g_gap_support_requires_final_v5_and_b_refit": True,
        "cross_gap_support_used_as_curvature": False,
        "blocker_logic_changed": False,
    }
    return output, masks


__all__ = [
    "POLICY_ID",
    "ContinuousCorePolicy",
    "fit_trapezoid_boundary_retry_v5_probe_c",
]
