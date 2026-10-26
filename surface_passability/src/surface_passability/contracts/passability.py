"""Typed contracts between boundary analysis and passability measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


MeasurementScope = Literal["full", "partial", "provisional", "none"]
MeasurementAuthority = Literal[
    "allowed", "conditional_on_v6_reliability", "review_only", "denied"
]
ResultAuthority = Literal["authoritative", "provisional_review", "none"]
RowConfidence = Literal["reliable", "low", "insufficient"]
TopologyStatus = Literal["connected", "closed", "abstained"]
PassabilityState = Literal[
    "undetermined", "relative_only", "blocked", "narrowed", "passable"
]


@dataclass(frozen=True, slots=True)
class BoundaryStatusRoute:
    measurement_scope: MeasurementScope
    measurement_authority: MeasurementAuthority
    result_authority: ResultAuthority
    human_review_required: bool
    abstention_reason: str | None

    @property
    def calculation_allowed(self) -> bool:
        return self.measurement_authority != "denied"


@dataclass(frozen=True, slots=True)
class PassabilityNullResult:
    schema_version: str
    sample_id: str
    boundary_policy_id: str
    boundary_status: str
    result_authority: Literal["none"]
    selected_surface: str | None
    measurement_scope: Literal["none"]
    measurement_confidence: Literal["insufficient"]
    human_review_required: bool
    abstention_reasons: list[str]
    minimum_passable_ratio: None
    raw_minimum_passable_ratio: None
    maximum_obstructed_width_ratio: None
    obstacle_area_ratio: None
    bottleneck: None
    surface_width_m: None
    absolute_passable_width_m: None
    effective_blocked_width_m: None
    agent_width_m: None
    safety_factor: None
    required_safe_width_m: None
    required_passable_ratio: None
    required_safe_ratio: None
    passability_state: Literal["undetermined"]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase7a_policy_id"] = value.pop("boundary_policy_id")
        value["phase7a_status"] = value.pop("boundary_status")
        ordered = {
            "schema_version": value.pop("schema_version"),
            "sample_id": value.pop("sample_id"),
            "phase7a_policy_id": value.pop("phase7a_policy_id"),
            "phase7a_status": value.pop("phase7a_status"),
        }
        ordered.update(value)
        return ordered


@dataclass(frozen=True, slots=True)
class PassabilityBoundaryHandoff:
    sample_id: str
    selected_surface: str | None
    boundary_policy_id: str
    boundary_status: str
    measurement_scope: MeasurementScope
    measurement_authority: MeasurementAuthority
    result_authority: ResultAuthority
    human_review_required: bool
    authorized_row_range: tuple[int, int] | None
    observed_rows: tuple[int, ...]
    generated_rows: tuple[int, ...]
    structural_mask_sha256: str
    structural_area_px: int
    provenance: dict[str, Any]
    structural_mask: np.ndarray | None
    typed_null_result: PassabilityNullResult | None


@dataclass(frozen=True, slots=True)
class MeasurementOccupancyResult:
    """Verified in-memory obstacle occupancy used only for width measurement."""

    sample_id: str
    coverage_status: str
    abstention_reasons: tuple[str, ...]
    occupancy_mask: np.ndarray | None
    occupancy_mask_sha256: str | None
    occupancy_pixels: int | None
    base_blocker_pixels: int | None
    added_completion_pixels: int | None
    clipped_blocker_pixels: int | None
    instance_records: tuple[dict[str, Any], ...]
    evidence_source: str
    upstream_hashes_before: dict[str, str]
    upstream_hashes_after: dict[str, str]

    @property
    def measurement_allowed(self) -> bool:
        return self.occupancy_mask is not None and not self.abstention_reasons


@dataclass(frozen=True, slots=True)
class EligibleRowSelection:
    """Auditable structural rows authorized for width measurement."""

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


@dataclass(frozen=True, slots=True)
class CorridorNode:
    node_id: str
    y: int
    interval: tuple[int, int]
    width_px: int
    predecessor_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConnectedCorridorTopology:
    """Adjacent-row overlap DAG for the authorized measurement run."""

    schema_version: str
    sample_id: str
    measurement_scope: str
    measurement_run_range: tuple[int, int] | None
    measurement_run_row_count: int
    topology_status: TopologyStatus
    row_intervals: dict[int, tuple[tuple[int, int], ...]]
    nodes: tuple[CorridorNode, ...]
    edges: tuple[tuple[str, str], ...]
    full_run_terminal_ids: tuple[str, ...]
    diagnostic_canonical_path: tuple[str, ...]
    split_node_ids: tuple[str, ...]
    merge_node_ids: tuple[str, ...]
    first_disconnected_row: int | None
    abstention_reasons: tuple[str, ...]
    flags: tuple[str, ...]

    @property
    def full_run_corridor_exists(self) -> bool:
        return self.topology_status == "connected"

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        if value["measurement_run_range"] is not None:
            value["measurement_run_range"] = list(value["measurement_run_range"])
        value["row_intervals"] = {
            str(y): [list(interval) for interval in intervals]
            for y, intervals in self.row_intervals.items()
        }
        return value


@dataclass(frozen=True, slots=True)
class RelativePassabilityMetrics:
    """Scale-free passability measurements from the selected corridor path."""

    schema_version: str
    sample_id: str
    measurement_scope: str
    measurement_confidence: str
    representative_relative_metrics_available: bool
    minimum_passable_ratio: float | None
    raw_minimum_passable_ratio: float | None
    maximum_obstructed_width_ratio: float | None
    obstacle_area_ratio: float | None
    selected_corridor_path: tuple[dict[str, Any], ...]
    bottleneck: dict[str, Any] | None
    maximum_obstruction: dict[str, Any] | None
    abstention_reasons: tuple[str, ...]
    flags: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhysicalPassabilityResult:
    """Nullable metric-width projection and final passability state."""

    schema_version: str
    sample_id: str
    measurement_scope: str
    human_review_required: bool
    measurement_confidence: str
    minimum_passable_ratio: float | None
    surface_width_m: float | None
    absolute_passable_width_m: float | None
    effective_blocked_width_m: float | None
    agent_width_m: float | None
    safety_factor: float | None
    required_safe_width_m: float | None
    required_passable_ratio: float | None
    required_safe_ratio: float | None
    passability_state: PassabilityState
    abstention_reasons: tuple[str, ...]
    flags: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PassabilityResult:
    """Complete typed result and compatibility record for width analysis."""

    handoff: PassabilityBoundaryHandoff
    occupancy: MeasurementOccupancyResult
    rows: EligibleRowSelection
    topology: ConnectedCorridorTopology
    metrics: RelativePassabilityMetrics
    physical: PhysicalPassabilityResult
    record: dict[str, Any]
