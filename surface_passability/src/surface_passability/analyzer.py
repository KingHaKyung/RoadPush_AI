"""Long-lived production analyzer for service workers."""

from __future__ import annotations

from surface_passability.config.settings import RuntimeSettings
from surface_passability.contracts.request import AnalysisRequest
from surface_passability.contracts.response import AnalysisResponse
from surface_passability.dependencies import (
    PipelineDependencies,
    get_default_dependencies,
)


class SurfacePassabilityAnalyzer:
    """Own and reuse verified models, caches, and inference locks per worker."""

    def __init__(
        self,
        *,
        dependencies: PipelineDependencies | None = None,
        requested_device: str | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        if dependencies is not None and (
            requested_device is not None or settings is not None
        ):
            raise ValueError(
                "explicit dependencies cannot be combined with runtime options"
            )
        self.dependencies = dependencies or get_default_dependencies(
            requested_device=requested_device,
            settings=settings,
        )

    def analyze_photo(self, request: AnalysisRequest) -> AnalysisResponse:
        from surface_passability.pipeline import analyze_photo

        return analyze_photo(request, dependencies=self.dependencies)
