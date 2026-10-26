"""Select the structural rows authorized for passability measurement."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from surface_passability.contracts.passability import (
    EligibleRowSelection,
    MeasurementOccupancyResult,
    PassabilityBoundaryHandoff,
    RowConfidence,
)
from surface_passability.policies.passability import EligibleRowPolicy


def consecutive_row_runs(rows: Sequence[int]) -> list[tuple[int, ...]]:
    """Group distinct row indexes into ascending consecutive runs."""

    result: list[tuple[int, ...]] = []
    current: list[int] = []
    for y in sorted(set(int(value) for value in rows)):
        if current and y != current[-1] + 1:
            result.append(tuple(current))
            current = []
        current.append(y)
    if current:
        result.append(tuple(current))
    return result


def row_range(rows: Sequence[int]) -> tuple[int, int] | None:
    """Return the inclusive outer range of an already ordered row sequence."""

    return (int(rows[0]), int(rows[-1])) if rows else None


def _upstream_abstention(
    handoff: PassabilityBoundaryHandoff,
    occupancy: MeasurementOccupancyResult,
    policy: EligibleRowPolicy,
) -> EligibleRowSelection:
    reasons = tuple(
        dict.fromkeys(
            [
                *(
                    handoff.typed_null_result.abstention_reasons
                    if handoff.typed_null_result
                    else []
                ),
                *occupancy.abstention_reasons,
            ]
        )
    ) or ("phase7b_measurement_not_authorized",)
    return EligibleRowSelection(
        schema_version="phase7_b_rows_v1",
        sample_id=handoff.sample_id,
        row_policy_id=policy.policy_id,
        measurement_scope=handoff.measurement_scope,
        authorized_row_range=handoff.authorized_row_range,
        measurement_confidence="insufficient",
        measurement_run_range=None,
        measurement_run_rows=(),
        candidate_run_range=None,
        candidate_run_rows=(),
        support_rows=(),
        eligible_rows=(),
        excluded_rows={},
        structural_spans={},
        row_support_sources={},
        minimum_structural_span_px=None,
        support_row_count=0,
        eligible_row_count=0,
        excluded_in_scope_row_count=0,
        candidate_run_row_count=0,
        measurement_run_coverage=None,
        representative_measurement_allowed=False,
        human_review_required=handoff.human_review_required,
        abstention_reasons=reasons,
        flags=("upstream_measurement_abstained",),
    )


def select_eligible_rows(
    handoff: PassabilityBoundaryHandoff,
    occupancy: MeasurementOccupancyResult,
    *,
    policy: EligibleRowPolicy = EligibleRowPolicy(),
) -> EligibleRowSelection:
    """Select the longest representative run of supported structural rows."""

    if (
        handoff.structural_mask is None
        or handoff.authorized_row_range is None
        or not occupancy.measurement_allowed
    ):
        return _upstream_abstention(handoff, occupancy, policy)

    structural = np.asarray(handoff.structural_mask, dtype=bool)
    if structural.ndim != 2:
        raise ValueError("Phase 7-A structural mask must be a 2D array")
    if (
        occupancy.occupancy_mask is None
        or occupancy.occupancy_mask.shape != structural.shape
    ):
        raise ValueError("measurement occupancy must match the structural mask")
    height, width = structural.shape
    start, end = handoff.authorized_row_range
    if start < 0 or end < start or end >= height:
        raise ValueError("authorized_row_range is outside the structural mask")

    minimum_span = policy.minimum_span(width)
    observed_rows = set(handoff.observed_rows)
    generated_rows = set(handoff.generated_rows)
    provenance_rows = observed_rows | generated_rows
    present_rows = tuple(int(y) for y in np.flatnonzero(structural.any(axis=1)))
    support_rows = tuple(y for y in present_rows if start <= y <= end)
    excluded: dict[str, list[int]] = {
        "out_of_scope": [y for y in present_rows if y < start or y > end],
        "undersized": [],
        "left_frame_clipped": [],
        "right_frame_clipped": [],
        "unresolved_boundary_support": [],
    }
    spans: dict[int, tuple[int, int]] = {}
    row_sources: dict[int, str] = {}
    eligible: list[int] = []
    for y in support_rows:
        xs = np.flatnonzero(structural[y])
        left, right = int(xs[0]), int(xs[-1])
        spans[y] = (left, right)
        if y in observed_rows and y in generated_rows:
            row_sources[y] = "observed_and_generated"
        elif y in observed_rows:
            row_sources[y] = "observed_phase7a_geometry"
        elif y in generated_rows:
            row_sources[y] = "approved_generated_phase7a_geometry"
        else:
            row_sources[y] = "unresolved"
        row_reasons: list[str] = []
        if right - left + 1 < minimum_span:
            excluded["undersized"].append(y)
            row_reasons.append("undersized")
        if left == 0:
            excluded["left_frame_clipped"].append(y)
            row_reasons.append("left_frame_clipped")
        if right == width - 1:
            excluded["right_frame_clipped"].append(y)
            row_reasons.append("right_frame_clipped")
        if y not in provenance_rows:
            excluded["unresolved_boundary_support"].append(y)
            row_reasons.append("unresolved_boundary_support")
        if not row_reasons:
            eligible.append(y)

    runs = consecutive_row_runs(eligible)
    candidate = max(runs, key=len, default=())
    coverage = len(candidate) / len(support_rows) if support_rows else None
    enough_rows = len(candidate) >= policy.minimum_consecutive_rows
    enough_coverage = bool(
        coverage is not None
        and coverage >= policy.minimum_measurement_run_coverage
    )
    if enough_rows and enough_coverage:
        confidence: RowConfidence = "reliable"
        measurement_run = candidate
        reasons: tuple[str, ...] = ()
    elif enough_rows:
        confidence = "low"
        measurement_run = ()
        reasons = ("measurement_run_coverage_below_minimum",)
    else:
        confidence = "insufficient"
        measurement_run = ()
        reasons = ("insufficient_consecutive_eligible_rows",)
        if support_rows and coverage is not None and not enough_coverage:
            reasons += ("measurement_run_coverage_below_minimum",)
        if not support_rows:
            reasons = ("no_structural_support_in_authorized_scope",)

    excluded_in_scope = set(support_rows) - set(eligible)
    flags: list[str] = []
    if excluded["out_of_scope"]:
        flags.append("structural_rows_outside_authorized_scope")
    if excluded_in_scope:
        flags.append("rows_excluded_from_measurement")
    if confidence != "reliable":
        flags.append(f"measurement_{confidence}_confidence")
    if handoff.measurement_scope == "partial":
        flags.append("partial_scope_only")
    if handoff.measurement_scope == "provisional":
        flags.append("provisional_review_only")
    serialized_excluded = {
        key: tuple(values) for key, values in excluded.items() if values
    }
    return EligibleRowSelection(
        schema_version="phase7_b_rows_v1",
        sample_id=handoff.sample_id,
        row_policy_id=policy.policy_id,
        measurement_scope=handoff.measurement_scope,
        authorized_row_range=handoff.authorized_row_range,
        measurement_confidence=confidence,
        measurement_run_range=row_range(measurement_run),
        measurement_run_rows=tuple(measurement_run),
        candidate_run_range=row_range(candidate),
        candidate_run_rows=tuple(candidate),
        support_rows=support_rows,
        eligible_rows=tuple(eligible),
        excluded_rows=serialized_excluded,
        structural_spans=spans,
        row_support_sources=row_sources,
        minimum_structural_span_px=minimum_span,
        support_row_count=len(support_rows),
        eligible_row_count=len(eligible),
        excluded_in_scope_row_count=len(excluded_in_scope),
        candidate_run_row_count=len(candidate),
        measurement_run_coverage=coverage,
        representative_measurement_allowed=confidence == "reliable",
        human_review_required=handoff.human_review_required,
        abstention_reasons=reasons,
        flags=tuple(flags),
    )
