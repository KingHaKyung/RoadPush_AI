"""Typed per-row evidence for evaluated surface corridors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from surface_passability.contracts.boundaries import OuterBoundaryObservation


BoundaryReliability = Literal["reliable", "uncertain"]
BranchReliability = Literal[
    "uncertain",
    "no_alternative_branch_detected",
]


@dataclass(frozen=True, slots=True)
class SurfaceRowEvidence:
    """Free-space and reliability evidence for one outer-boundary row."""

    observation: OuterBoundaryObservation
    structural_span: tuple[int, int]
    free_intervals: tuple[tuple[int, int], ...]
    selected_route_interval: tuple[int, int] | None
    transient_occlusion: bool
    boundary_reliability: BoundaryReliability
    branch_reliability: BranchReliability

    def to_legacy_record(self) -> dict[str, object]:
        record = self.observation.to_legacy_record()
        record.update(
            {
                "structural_sidewalk_span": list(self.structural_span),
                "free_segments": [list(interval) for interval in self.free_intervals],
                "selected_route_segment": (
                    list(self.selected_route_interval)
                    if self.selected_route_interval
                    else None
                ),
                "transient_occlusion": self.transient_occlusion,
                "boundary_reliability": self.boundary_reliability,
                "branch_reliability": self.branch_reliability,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class SurfaceRowEvaluation:
    """Evaluated component, free mask, rows, and aggregate reliability counts."""

    component_mask: np.ndarray
    free_mask: np.ndarray
    rows: tuple[SurfaceRowEvidence, ...]
    transient_occluded_row_count: int
    boundary_uncertain_row_count: int
    branch_uncertain_row_count: int

    def to_legacy_result(self) -> dict[str, object]:
        return {
            "component": self.component_mask,
            "free": self.free_mask,
            "rows": [row.to_legacy_record() for row in self.rows],
            "transient_occluded_rows": self.transient_occluded_row_count,
            "boundary_uncertain_rows": self.boundary_uncertain_row_count,
            "branch_uncertain_rows": self.branch_uncertain_row_count,
        }


@dataclass(frozen=True, slots=True)
class RoadRowEvidence:
    """Free and occupied Road intervals observed within one structural row."""

    observation: OuterBoundaryObservation
    structural_span: tuple[int, int]
    free_intervals: tuple[tuple[int, int], ...]
    occupied_intervals: tuple[tuple[int, int], ...]
    selected_route_interval: tuple[int, int] | None
    boundary_reliability: BoundaryReliability
    branch_reliability: BranchReliability

    def to_legacy_record(self) -> dict[str, object]:
        """Return the exact frozen Road row mapping and key order."""

        record = self.observation.to_legacy_record()
        record.update(
            {
                "structural_road_span": list(self.structural_span),
                "free_segments": [list(interval) for interval in self.free_intervals],
                "occupied_segments": [
                    list(interval) for interval in self.occupied_intervals
                ],
                "selected_route_segment": (
                    list(self.selected_route_interval)
                    if self.selected_route_interval
                    else None
                ),
                "boundary_reliability": self.boundary_reliability,
                "branch_reliability": self.branch_reliability,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class RoadRowEvaluation:
    """Selected Road component, blockers, free mask, and row evidence."""

    component_mask: np.ndarray
    blocker_mask: np.ndarray
    free_mask: np.ndarray
    rows: tuple[RoadRowEvidence, ...]
    boundary_uncertain_row_count: int
    branch_uncertain_row_count: int

    def to_legacy_result(self) -> dict[str, object]:
        """Return the exact frozen Road evaluation mapping and aliases."""

        return {
            "component": self.component_mask,
            "blocker": self.blocker_mask,
            "occupied": self.blocker_mask,
            "free": self.free_mask,
            "rows": [row.to_legacy_record() for row in self.rows],
            "boundary_uncertain_rows": self.boundary_uncertain_row_count,
            "branch_uncertain_rows": self.branch_uncertain_row_count,
        }
