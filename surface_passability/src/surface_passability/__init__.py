"""Public API for the RoadPush surface-passability model."""

from __future__ import annotations

from surface_passability.contracts.request import AnalysisRequest, InputProfile
from surface_passability.contracts.response import AnalysisResponse
from surface_passability.dependencies import PipelineDependencies
from surface_passability.analyzer import SurfacePassabilityAnalyzer


def analyze_photo(
    request: AnalysisRequest,
    *,
    dependencies: PipelineDependencies | None = None,
) -> AnalysisResponse:
    """Load the production orchestrator only when an analysis is requested."""

    from surface_passability.pipeline import analyze_photo as run_analysis

    return run_analysis(request, dependencies=dependencies)

__all__ = [
    "AnalysisRequest",
    "AnalysisResponse",
    "InputProfile",
    "PipelineDependencies",
    "SurfacePassabilityAnalyzer",
    "analyze_photo",
]

