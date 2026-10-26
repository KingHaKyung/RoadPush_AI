"""Typed boundary-analysis contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class SidewalkBoundaryMode(str, Enum):
    """Supported strategies for constructing the sidewalk boundary mask."""

    OBSERVED_COMPONENT = "observed_component"
    OBSTACLE_TRANSPARENT_COMPONENT = "obstacle_transparent_component"
    ALL_CANDIDATE_ROW_SPAN = "all_candidate_row_span"


@dataclass(slots=True)
class SidewalkBoundaryResult:
    """Constructed sidewalk boundary and compatibility diagnostics."""

    mask: np.ndarray
    additions: np.ndarray
    mode: SidewalkBoundaryMode
    calls: dict[str, bool]


BoundaryEndpointKind = Literal[
    "frame_truncated",
    "obstacle_occlusion",
    "physical_boundary",
    "semantic_boundary",
]
BoundarySide = Literal["left", "right"]
BoundaryWidthState = Literal[
    "complete",
    "lower_bound",
    "occlusion_limited",
    "unavailable",
]


@dataclass(frozen=True, slots=True)
class BoundarySupportEvidence:
    """Local image-gradient and curb-context support at one boundary point."""

    gradient: float
    curb_fraction: float

    def to_legacy_record(self) -> dict[str, float]:
        return {"gradient": self.gradient, "curb": self.curb_fraction}


@dataclass(frozen=True, slots=True)
class BoundaryRowObservation:
    y: int
    left: int
    right: int
    width: int
    anchor_x: float
    left_gradient: float
    right_gradient: float
    left_curb: float
    right_curb: float
    left_kind: BoundaryEndpointKind
    right_kind: BoundaryEndpointKind

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "y": self.y, "left": self.left, "right": self.right,
            "width": self.width, "anchor_x": self.anchor_x,
            "left_gradient": self.left_gradient,
            "right_gradient": self.right_gradient,
            "left_curb": self.left_curb, "right_curb": self.right_curb,
            "left_kind": self.left_kind, "right_kind": self.right_kind,
        }


@dataclass(frozen=True, slots=True)
class OuterBoundaryObservation:
    """Outer component endpoints and width authority observed at one row."""

    y: int
    left: int
    right: int
    width: int
    anchor_x: float
    left_kind: BoundaryEndpointKind
    right_kind: BoundaryEndpointKind
    width_state: BoundaryWidthState

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "y": self.y,
            "left": self.left,
            "right": self.right,
            "width": self.width,
            "anchor_x": self.anchor_x,
            "left_kind": self.left_kind,
            "right_kind": self.right_kind,
            "width_state": self.width_state,
        }


@dataclass(frozen=True, slots=True)
class SideBoundaryFit:
    """Robust linear continuation fitted for one observed corridor side."""

    coefficients: tuple[float, float] | None
    predicted_columns: tuple[float, ...]
    eligible_observation_count: int

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "coeff": None if self.coefficients is None else list(self.coefficients),
            "predicted": list(self.predicted_columns),
            "eligible": self.eligible_observation_count,
        }


@dataclass(frozen=True, slots=True)
class MainCorridorRow:
    """One observed row annotated with its branch-aware main-corridor width."""

    observation: BoundaryRowObservation
    left_branch_candidate: bool
    right_branch_candidate: bool
    main_left: int
    main_right: int
    width_state: BoundaryWidthState
    main_width: int

    def to_legacy_record(self) -> dict[str, object]:
        record = self.observation.to_legacy_record()
        record.update({
            "left_branch_candidate": self.left_branch_candidate,
            "right_branch_candidate": self.right_branch_candidate,
            "main_left": self.main_left,
            "main_right": self.main_right,
            "width_state": self.width_state,
            "main_width": self.main_width,
        })
        return record


@dataclass(frozen=True, slots=True)
class MainCorridorAnalysis:
    """Branch-aware main-corridor rows and the evidence used to derive them."""

    rows: tuple[MainCorridorRow, ...]
    left_fit: SideBoundaryFit
    right_fit: SideBoundaryFit
    branch_threshold_px: float

    def to_legacy_summary(self) -> dict[str, object]:
        return {
            "left_model": self.left_fit.to_legacy_record(),
            "right_model": self.right_fit.to_legacy_record(),
            "branch_threshold_px": self.branch_threshold_px,
        }


@dataclass(slots=True)
class SideBranchCandidate:
    """One sampled side-branch candidate and its final pruning decision."""

    row_index: int
    y: int
    side: BoundarySide
    endpoint_kind: BoundaryEndpointKind
    veto_reason: str | None
    accepted: bool = False

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "y": self.y,
            "side": self.side,
            "endpoint_kind": self.endpoint_kind,
            "veto_reason": self.veto_reason,
            "accepted": self.accepted,
        }


@dataclass(slots=True)
class AcceptedSideBranchGroup:
    """One consecutive group approved for removal from the boundary."""

    side: BoundarySide
    sampled_row_indices: list[int]
    sampled_y: list[int]

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "side": self.side,
            "sampled_row_indices": self.sampled_row_indices,
            "sampled_y": self.sampled_y,
        }


@dataclass(slots=True)
class SideBranchPruningResult:
    """Masks and typed evidence produced by structural side-branch pruning."""

    corridor_analysis: MainCorridorAnalysis | None
    raw_candidate_mask: np.ndarray
    vetoed_candidate_mask: np.ndarray
    approved_removal_mask: np.ndarray
    pruned_boundary_mask: np.ndarray
    accepted_groups: list[AcceptedSideBranchGroup]
    candidate_rows: list[SideBranchCandidate] | None
    policy: str
    route_high_confidence: bool
    far_limit_y: int
    footprint_or_blocker_used: bool

    def to_legacy_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "rows": (
                []
                if self.corridor_analysis is None
                else [
                    row.to_legacy_record()
                    for row in self.corridor_analysis.rows
                ]
            ),
            "raw_candidate": self.raw_candidate_mask,
            "vetoed_candidate": self.vetoed_candidate_mask,
            "approved_removal": self.approved_removal_mask,
            "after": self.pruned_boundary_mask,
            "accepted_groups": [
                group.to_legacy_record()
                for group in self.accepted_groups
            ],
        }
        if self.candidate_rows is not None:
            record["candidate_rows"] = [
                candidate.to_legacy_record()
                for candidate in self.candidate_rows
            ]
        record.update(
            {
                "policy": self.policy,
                "route_high_confidence": self.route_high_confidence,
                "far_limit_y": self.far_limit_y,
                "footprint_or_blocker_used": (
                    self.footprint_or_blocker_used
                ),
            }
        )
        return record
