"""Public response contract for the integrated RoadPush pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from surface_passability import AnalysisResponse


ROADPUSH_RESPONSE_SCHEMA_VERSION = "roadpush_analysis_response_v1"


@dataclass(frozen=True, slots=True)
class SurfaceObstacleResponse:
    status: str
    obstacle_count: int
    types: tuple[str, ...]
    obstacles: tuple[dict[str, Any], ...]
    components: tuple[dict[str, Any], ...]
    auto_routable: bool
    policy_id: str
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "status": self.status,
            "obstacle_count": self.obstacle_count,
            "types": list(self.types),
            "obstacles": list(self.obstacles),
            "components": list(self.components),
            "auto_routable": self.auto_routable,
            "policy_id": self.policy_id,
        }
        if self.error is not None:
            value["error"] = dict(self.error)
        return value


@dataclass(frozen=True, slots=True)
class RoadPushAnalysisResponse:
    request_id: str
    sample_id: str
    status: str
    surface: AnalysisResponse
    surface_obstacles: SurfaceObstacleResponse
    diagnostics: dict[str, Any]
    schema_version: str = ROADPUSH_RESPONSE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "sample_id": self.sample_id,
            "status": self.status,
            "surface": self.surface.to_dict(),
            "surface_obstacles": self.surface_obstacles.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }
