"""Typed aggregates for complete surface-candidate analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surface_passability.contracts.boundary import BoundaryAnalysisResult
from surface_passability.contracts.image import PreparedImage
from surface_passability.contracts.passability import PassabilityResult
from surface_passability.contracts.response import AnalysisResponse
from surface_passability.contracts.road import RoadContext
from surface_passability.contracts.road_analysis import RoadCandidateComputation
from surface_passability.contracts.sidewalk import SidewalkCandidateComputation
from surface_passability.contracts.surfaces import (
    SurfaceObstacleAnalysis,
    SurfacePipelineContext,
    SurfaceSelectionResult,
)


@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    """One complete in-memory surface candidate without artifact I/O."""

    name: str
    source_mask: np.ndarray
    repaired_mask: np.ndarray
    raw_structural_mask: np.ndarray
    structural_mask: np.ndarray
    blocker_mask: np.ndarray
    free_mask: np.ndarray
    obstacle_silhouette_mask: np.ndarray
    passability_instances: tuple[dict[str, object], ...]
    calculation: SidewalkCandidateComputation | RoadCandidateComputation


@dataclass(frozen=True, slots=True)
class SurfaceCandidates:
    """Both production candidates and their shared in-memory evidence."""

    sidewalk_context: SurfacePipelineContext
    road_context: RoadContext
    obstacles: SurfaceObstacleAnalysis
    sidewalk: SurfaceCandidate
    road: SurfaceCandidate


@dataclass(frozen=True, slots=True)
class SurfaceSelection:
    """Phase-6 decision paired with the selected candidate, when any."""

    decision: SurfaceSelectionResult
    candidate: SurfaceCandidate | None


@dataclass(frozen=True, slots=True)
class SelectedSurfaceAnalysis:
    """Selected surface, final boundary, and complete passability result."""

    selection: SurfaceSelection
    boundary: BoundaryAnalysisResult
    passability: PassabilityResult


@dataclass(frozen=True, slots=True)
class SurfaceIntegrationEvidence:
    """In-memory evidence exposed only to the top-level RoadPush pipeline."""

    prepared_image: PreparedImage
    selected_surface: str | None
    blocker_mask: np.ndarray | None
    structural_mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class SurfaceAnalysisBundle:
    """Public response paired with non-serializable integration evidence."""

    response: AnalysisResponse
    evidence: SurfaceIntegrationEvidence
