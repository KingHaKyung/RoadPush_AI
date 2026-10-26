"""Compose the verified passability responsibilities into one typed result."""

from __future__ import annotations

import math
from typing import Any

from surface_passability.contracts.passability import (
    ConnectedCorridorTopology,
    EligibleRowSelection,
    MeasurementOccupancyResult,
    PassabilityBoundaryHandoff,
    PassabilityResult,
    PhysicalPassabilityResult,
    RelativePassabilityMetrics,
)
from surface_passability.passability.corridors import track_connected_corridors
from surface_passability.passability.eligible_rows import select_eligible_rows
from surface_passability.passability.metrics import calculate_relative_metrics
from surface_passability.passability.physical_width import calculate_physical_width


def canonical_passability_record(
    handoff: PassabilityBoundaryHandoff,
    occupancy: MeasurementOccupancyResult,
    rows: EligibleRowSelection,
    topology: ConnectedCorridorTopology,
    metrics: RelativePassabilityMetrics,
    physical: PhysicalPassabilityResult,
) -> dict[str, Any]:
    """Serialize the stable backend-compatible passability payload."""

    return {
        "schema_version": "phase7_b_result_v2",
        "sample_id": handoff.sample_id,
        "selected_surface": handoff.selected_surface,
        "phase7a_policy_id": handoff.boundary_policy_id,
        "phase7a_status": handoff.boundary_status,
        "result_authority": handoff.result_authority,
        "measurement_scope": physical.measurement_scope,
        "measurement_confidence": physical.measurement_confidence,
        "human_review_required": physical.human_review_required,
        "abstention_reasons": list(physical.abstention_reasons),
        "minimum_passable_ratio": physical.minimum_passable_ratio,
        "raw_minimum_passable_ratio": metrics.raw_minimum_passable_ratio,
        "maximum_obstructed_width_ratio": metrics.maximum_obstructed_width_ratio,
        "obstacle_area_ratio": metrics.obstacle_area_ratio,
        "bottleneck": metrics.bottleneck,
        "selected_corridor_path": list(metrics.selected_corridor_path),
        "maximum_obstruction": metrics.maximum_obstruction,
        "surface_width_m": physical.surface_width_m,
        "absolute_passable_width_m": physical.absolute_passable_width_m,
        "effective_blocked_width_m": physical.effective_blocked_width_m,
        "agent_width_m": physical.agent_width_m,
        "safety_factor": physical.safety_factor,
        "required_safe_width_m": physical.required_safe_width_m,
        "required_passable_ratio": physical.required_passable_ratio,
        "required_safe_ratio": physical.required_safe_ratio,
        "passability_state": physical.passability_state,
        "flags": list(physical.flags),
        "diagnostics": {
            "phase7a": {
                "measurement_authority": handoff.measurement_authority,
                "authorized_row_range": (
                    list(handoff.authorized_row_range)
                    if handoff.authorized_row_range
                    else None
                ),
                "structural_mask_sha256": handoff.structural_mask_sha256,
                "structural_area_px": handoff.structural_area_px,
                "observed_row_count": len(handoff.observed_rows),
                "generated_row_count": len(handoff.generated_rows),
                "provenance": handoff.provenance,
            },
            "occupancy": {
                "coverage_status": occupancy.coverage_status,
                "occupancy_mask_sha256": occupancy.occupancy_mask_sha256,
                "occupancy_pixels": occupancy.occupancy_pixels,
                "base_blocker_pixels": occupancy.base_blocker_pixels,
                "added_completion_pixels": occupancy.added_completion_pixels,
                "clipped_blocker_pixels": occupancy.clipped_blocker_pixels,
                "evidence_source": occupancy.evidence_source,
                "instance_records": list(occupancy.instance_records),
                "upstream_hashes_before": occupancy.upstream_hashes_before,
                "upstream_hashes_after": occupancy.upstream_hashes_after,
            },
            "rows": {
                "policy_id": rows.row_policy_id,
                "measurement_run_range": (
                    list(rows.measurement_run_range)
                    if rows.measurement_run_range
                    else None
                ),
                "support_row_count": rows.support_row_count,
                "eligible_row_count": rows.eligible_row_count,
                "excluded_in_scope_row_count": rows.excluded_in_scope_row_count,
                "candidate_run_row_count": rows.candidate_run_row_count,
                "measurement_run_coverage": rows.measurement_run_coverage,
                "minimum_structural_span_px": rows.minimum_structural_span_px,
                "excluded_rows": {
                    reason: list(ys)
                    for reason, ys in rows.excluded_rows.items()
                },
            },
            "topology": {
                "status": topology.topology_status,
                "node_count": len(topology.nodes),
                "edge_count": len(topology.edges),
                "full_run_terminal_count": len(topology.full_run_terminal_ids),
                "split_node_count": len(topology.split_node_ids),
                "merge_node_count": len(topology.merge_node_ids),
                "first_disconnected_row": topology.first_disconnected_row,
                "diagnostic_canonical_path": list(
                    topology.diagnostic_canonical_path
                ),
            },
        },
    }


def validate_passability_record(record: dict[str, Any]) -> None:
    """Validate the compatibility payload without recomputing measurements."""

    required = {
        "schema_version",
        "sample_id",
        "phase7a_policy_id",
        "phase7a_status",
        "result_authority",
        "measurement_scope",
        "measurement_confidence",
        "human_review_required",
        "abstention_reasons",
        "minimum_passable_ratio",
        "raw_minimum_passable_ratio",
        "maximum_obstructed_width_ratio",
        "obstacle_area_ratio",
        "bottleneck",
        "surface_width_m",
        "absolute_passable_width_m",
        "effective_blocked_width_m",
        "agent_width_m",
        "safety_factor",
        "required_safe_width_m",
        "required_passable_ratio",
        "required_safe_ratio",
        "passability_state",
        "flags",
        "diagnostics",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Phase 7-B record missing fields: {sorted(missing)}")
    if record["schema_version"] != "phase7_b_result_v2":
        raise ValueError("unexpected Phase 7-B schema version")
    if record["measurement_scope"] not in {
        "full",
        "partial",
        "provisional",
        "none",
    }:
        raise ValueError("invalid measurement_scope")
    if record["result_authority"] not in {
        "authoritative",
        "provisional_review",
        "none",
    }:
        raise ValueError("invalid result_authority")
    if record["passability_state"] not in {
        "undetermined",
        "relative_only",
        "blocked",
        "narrowed",
        "passable",
    }:
        raise ValueError("invalid passability_state")
    if "obstacle_width_m" in record or "maximum_obstructed_width_m" in record:
        raise ValueError("forbidden physical obstacle-width field")
    ratio = record["minimum_passable_ratio"]
    relative_fields = (
        "minimum_passable_ratio",
        "raw_minimum_passable_ratio",
        "maximum_obstructed_width_ratio",
        "obstacle_area_ratio",
    )
    if ratio is None:
        if any(record[field] is not None for field in relative_fields):
            raise ValueError("unmeasurable records must keep all relative metrics null")
        if record["bottleneck"] is not None:
            raise ValueError("unmeasurable records cannot expose a bottleneck")
        if record["passability_state"] != "undetermined":
            raise ValueError("unmeasurable records must be undetermined")
    else:
        for field in relative_fields:
            value = record[field]
            if (
                value is None
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"invalid relative metric: {field}")
        if record["measurement_confidence"] != "reliable":
            raise ValueError("representative metrics require reliable confidence")
        if record["bottleneck"] is None:
            raise ValueError("representative metrics require bottleneck evidence")
    width = record["surface_width_m"]
    absolute = record["absolute_passable_width_m"]
    blocked = record["effective_blocked_width_m"]
    if width is None or ratio is None:
        if absolute is not None or blocked is not None:
            raise ValueError("absolute-width null propagation failed")
    else:
        if not math.isclose(
            absolute, width * ratio, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("absolute_passable_width_m formula mismatch")
        if not math.isclose(
            blocked, width - absolute, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("effective_blocked_width_m formula mismatch")
    if record["measurement_scope"] == "partial" and not record[
        "human_review_required"
    ]:
        raise ValueError("partial measurement must require human review")
    if record["phase7a_status"] == "ambiguous_hold":
        if record["measurement_scope"] != "provisional":
            raise ValueError("ambiguous_hold must retain provisional scope")
        if record["result_authority"] != "provisional_review":
            raise ValueError("ambiguous_hold cannot become authoritative")
        if not record["human_review_required"]:
            raise ValueError("ambiguous_hold must require human review")
    elif record["measurement_scope"] == "provisional":
        raise ValueError("provisional scope is reserved for ambiguous_hold")
    if (
        record["result_authority"] == "provisional_review"
        and record["phase7a_status"] != "ambiguous_hold"
    ):
        raise ValueError("provisional result authority requires ambiguous_hold")
    if record["measurement_scope"] in {"full", "partial"} and record[
        "result_authority"
    ] != "authoritative":
        raise ValueError("publishable scopes require authoritative geometry")
    if (
        record["measurement_scope"] == "none"
        and record["result_authority"] != "none"
    ):
        raise ValueError("none scope must have no result authority")
    if record["measurement_scope"] == "none" and ratio is not None:
        raise ValueError("none scope cannot expose representative ratios")
    before = record["diagnostics"]["occupancy"]["upstream_hashes_before"]
    after = record["diagnostics"]["occupancy"]["upstream_hashes_after"]
    if before and before != after:
        raise ValueError("upstream hashes changed during Phase 7-B")


def compose_passability_result(
    handoff: PassabilityBoundaryHandoff,
    occupancy: MeasurementOccupancyResult,
    *,
    surface_width_m: float | None = None,
    agent_width_m: float | None = None,
    safety_factor: float | None = None,
) -> PassabilityResult:
    """Run row, corridor, relative, and physical responsibilities once."""

    rows = select_eligible_rows(handoff, occupancy)
    topology = track_connected_corridors(handoff, occupancy, rows)
    metrics = calculate_relative_metrics(handoff, occupancy, rows, topology)
    physical = calculate_physical_width(
        handoff,
        metrics,
        surface_width_m=surface_width_m,
        agent_width_m=agent_width_m,
        safety_factor=safety_factor,
    )
    record = canonical_passability_record(
        handoff, occupancy, rows, topology, metrics, physical
    )
    validate_passability_record(record)
    return PassabilityResult(
        handoff=handoff,
        occupancy=occupancy,
        rows=rows,
        topology=topology,
        metrics=metrics,
        physical=physical,
        record=record,
    )
