"""Typed aggregate contracts for Road surface preparation."""

from __future__ import annotations

from dataclasses import dataclass

from surface_passability.contracts.obstacles import ObstacleOwnershipResult
from surface_passability.contracts.boundaries import SidewalkBoundaryResult
from surface_passability.contracts.drainage import LinearDrainAnalysisResult
from surface_passability.contracts.row_evidence import RoadRowEvaluation
from surface_passability.contracts.road import (
    RoadContext,
    RoadDrainRestorationResult,
    SurfaceOwnershipResult,
)
from surface_passability.contracts.sparse_support import (
    SparseRowSpanSupportResult,
)
from surface_passability.contracts.surfaces import (
    StructuralRouteStageResult,
    SurfaceComponentSelection,
    SurfaceObstacleSelectionResult,
    SurfacePipelineContext,
)


@dataclass(frozen=True, slots=True)
class RoadSurfacePreparation:
    """Final Road context and the one-way evidence used to construct it."""

    restoration: RoadDrainRestorationResult
    base_road: RoadContext
    linear_drain: LinearDrainAnalysisResult
    ownership: SurfaceOwnershipResult

    @property
    def road(self) -> RoadContext:
        """Return the final Road context after approved-drain restoration."""

        return self.restoration.road

    def road_policy_record(self) -> dict[str, object]:
        """Return the exact final frozen Road policy record."""

        return self.restoration.road_policy_record()


@dataclass(frozen=True, slots=True)
class RoadBoundaryPreparation:
    """Road-local pipeline view and boundaries before structural routing."""

    context: SurfacePipelineContext
    obstacle_selection: SurfaceObstacleSelectionResult
    current_boundary: SidewalkBoundaryResult
    proposed_boundary: SidewalkBoundaryResult


@dataclass(frozen=True, slots=True)
class RoadRowProcessing:
    """Initial, sparse-pruned, and final Road row evidence."""

    initial_evaluation: RoadRowEvaluation
    sparse_support: SparseRowSpanSupportResult
    final_evaluation: RoadRowEvaluation


@dataclass(frozen=True, slots=True)
class RoadCandidateComputation:
    """Live typed outputs of the complete Road calculation slice."""

    ownership: ObstacleOwnershipResult
    selection: SurfaceComponentSelection
    context: SurfacePipelineContext
    stage: StructuralRouteStageResult
    sparse_support: SparseRowSpanSupportResult
    final_evaluation: RoadRowEvaluation
