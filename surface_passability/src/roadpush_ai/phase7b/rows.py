from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from roadpush_ai.phase7b.handoff import Phase7AAdapter, Phase7BHandoff
from roadpush_ai.phase7b.occupancy_builder import (
    ExistingOccupancyEvidenceIndex,
    Phase7BOccupancyResult,
    build_measurement_occupancy,
)


RowConfidence = Literal["reliable", "low", "insufficient"]


@dataclass(frozen=True)
class Phase7BRowPolicy:
    """Fixed Step-3 policy; this module does not track free-space corridors."""

    policy_id: str = "phase7b_rows_v1_longest_coverage50"
    minimum_structural_span_px: int = 12
    minimum_structural_span_fraction: float = 0.02
    minimum_consecutive_rows: int = 5
    minimum_measurement_run_coverage: float = 0.50

    def minimum_span(self, image_width: int) -> int:
        return max(
            self.minimum_structural_span_px,
            int(math.ceil(image_width * self.minimum_structural_span_fraction)),
        )


@dataclass(frozen=True)
class Phase7BRowSelection:
    schema_version: str
    sample_id: str
    row_policy_id: str
    measurement_scope: str
    authorized_row_range: tuple[int, int] | None
    measurement_confidence: RowConfidence
    measurement_run_range: tuple[int, int] | None
    measurement_run_rows: tuple[int, ...]
    candidate_run_range: tuple[int, int] | None
    candidate_run_rows: tuple[int, ...]
    support_rows: tuple[int, ...]
    eligible_rows: tuple[int, ...]
    excluded_rows: dict[str, tuple[int, ...]]
    structural_spans: dict[int, tuple[int, int]]
    row_support_sources: dict[int, str]
    minimum_structural_span_px: int | None
    support_row_count: int
    eligible_row_count: int
    excluded_in_scope_row_count: int
    candidate_run_row_count: int
    measurement_run_coverage: float | None
    representative_measurement_allowed: bool
    human_review_required: bool
    abstention_reasons: tuple[str, ...]
    flags: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "authorized_row_range",
            "measurement_run_range",
            "candidate_run_range",
        ):
            if value[key] is not None:
                value[key] = list(value[key])
        value["structural_spans"] = {
            str(y): list(span) for y, span in self.structural_spans.items()
        }
        value["row_support_sources"] = {
            str(y): source for y, source in self.row_support_sources.items()
        }
        return value


def _runs(rows: Sequence[int]) -> list[tuple[int, ...]]:
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


def _range(rows: Sequence[int]) -> tuple[int, int] | None:
    return (int(rows[0]), int(rows[-1])) if rows else None


def _upstream_abstention(
    handoff: Phase7BHandoff,
    occupancy: Phase7BOccupancyResult,
    policy: Phase7BRowPolicy,
) -> Phase7BRowSelection:
    reasons = tuple(
        dict.fromkeys(
            [
                *(handoff.typed_null_result.abstention_reasons if handoff.typed_null_result else []),
                *occupancy.abstention_reasons,
            ]
        )
    ) or ("phase7b_measurement_not_authorized",)
    return Phase7BRowSelection(
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


def determine_eligible_rows(
    handoff: Phase7BHandoff,
    occupancy: Phase7BOccupancyResult,
    *,
    policy: Phase7BRowPolicy = Phase7BRowPolicy(),
) -> Phase7BRowSelection:
    """Select one auditable consecutive run without measuring a corridor.

    Phase 7-A's approved mask supplies the bilateral boundary geometry. A row
    remains boundary-supported only when that geometry has observed/generated
    provenance, sufficient span, and neither endpoint is clipped by the frame.
    The longest consecutive run wins; equal-length runs prefer the smaller y.
    It is publishable only when it covers at least half of all in-scope
    structural support rows, preventing a short easy fragment from standing in
    for the larger authorized surface.
    """

    if (
        handoff.structural_mask is None
        or handoff.authorized_row_range is None
        or not occupancy.measurement_allowed
    ):
        return _upstream_abstention(handoff, occupancy, policy)

    structural = np.asarray(handoff.structural_mask, dtype=bool)
    if structural.ndim != 2:
        raise ValueError("Phase 7-A structural mask must be a 2D array")
    if occupancy.occupancy_mask is None or occupancy.occupancy_mask.shape != structural.shape:
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

    runs = _runs(eligible)
    # _runs returns y-sorted runs, so max preserves the earliest y on a tie.
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
    return Phase7BRowSelection(
        schema_version="phase7_b_rows_v1",
        sample_id=handoff.sample_id,
        row_policy_id=policy.policy_id,
        measurement_scope=handoff.measurement_scope,
        authorized_row_range=handoff.authorized_row_range,
        measurement_confidence=confidence,
        measurement_run_range=_range(measurement_run),
        measurement_run_rows=tuple(measurement_run),
        candidate_run_range=_range(candidate),
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


def materialize_focused_row_audit(output: Path, sample_ids: Sequence[str]) -> Path:
    """Write Step-3 numeric evidence only; no corridor tracking or rendering."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite row audit: {output}")
    output.mkdir(parents=True)
    adapter = Phase7AAdapter()
    evidence = ExistingOccupancyEvidenceIndex()
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        authoritative = adapter.authoritative_record(sample_id)
        handoff = adapter.reconstruct(sample_id)
        occupancy = build_measurement_occupancy(
            handoff,
            v6_final_disposition=authoritative["v6_final_disposition"],
            v6_subclassification=authoritative["v6_subclassification"],
            evidence_index=evidence,
        )
        row_result = determine_eligible_rows(handoff, occupancy)
        record = row_result.to_record()
        record["phase7a_status"] = handoff.phase7a_status
        record["occupancy_coverage_status"] = occupancy.coverage_status
        records.append(record)
    destination = output / "results.json"
    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    arguments = parser.parse_args()
    print(
        materialize_focused_row_audit(
            arguments.output, tuple(dict.fromkeys(arguments.sample_id))
        ).resolve()
    )
