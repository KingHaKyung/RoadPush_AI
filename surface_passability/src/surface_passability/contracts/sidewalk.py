"""Typed contracts for Sidewalk candidate construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surface_passability.contracts.boundaries import SidewalkBoundaryResult
from surface_passability.contracts.obstacles import ObstacleOwnershipResult
from surface_passability.contracts.row_evidence import SurfaceRowEvaluation
from surface_passability.contracts.sparse_support import SparseRowSpanSupportResult
from surface_passability.contracts.surfaces import (
    StructuralRouteStageResult,
    SurfaceComponentSelection,
)


@dataclass(frozen=True, slots=True)
class SidewalkStructuralPreparation:
    """Current and obstacle-connected Sidewalk boundary/route stages."""

    current_boundary: SidewalkBoundaryResult
    current_stage: StructuralRouteStageResult
    proposed_boundary: SidewalkBoundaryResult
    proposed_stage: StructuralRouteStageResult


@dataclass(frozen=True, slots=True)
class SidewalkRowProcessing:
    """Initial, sparse-pruned, and final Sidewalk row evidence."""

    initial_evaluation: SurfaceRowEvaluation
    sparse_support: SparseRowSpanSupportResult
    evaluation_component_mask: np.ndarray
    final_evaluation: SurfaceRowEvaluation


@dataclass(frozen=True, slots=True)
class SidewalkCandidateComputation:
    """Live typed outputs of the complete Sidewalk calculation slice."""

    ownership: ObstacleOwnershipResult
    selection: SurfaceComponentSelection
    current_stage: StructuralRouteStageResult
    proposed_stage: StructuralRouteStageResult
    sparse_support: SparseRowSpanSupportResult
    final_evaluation: SurfaceRowEvaluation
