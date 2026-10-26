"""Long-lived sequential orchestrator for both RoadPush models."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Protocol

from surface_passability import AnalysisRequest, SurfacePassabilityAnalyzer
from surface_passability.config.settings import RuntimeSettings
from surface_passability.contracts.analysis import SurfaceAnalysisBundle

from roadpush_ai.attribution import AttributionPolicy, attribute_detections
from roadpush_ai.contracts import (
    RoadPushAnalysisResponse,
    SurfaceObstacleResponse,
)
from roadpush_ai.obstacle import ObstacleDetectionResult, ObstacleDetector, ObstacleSettings


class SurfaceAnalyzer(Protocol):
    def analyze_photo_with_evidence(
        self,
        request: AnalysisRequest,
    ) -> SurfaceAnalysisBundle: ...


class Detector(Protocol):
    def detect(self, native_rgb: object) -> ObstacleDetectionResult: ...


class RoadPushAnalyzer:
    """Run Surface then Obstacle sequentially, reusing both model owners."""

    def __init__(
        self,
        *,
        requested_device: str | None = None,
        runtime_root: Path | None = None,
        surface_analyzer: SurfaceAnalyzer | None = None,
        obstacle_detector: Detector | None = None,
        attribution_policy: AttributionPolicy = AttributionPolicy(),
    ) -> None:
        if (surface_analyzer is None) != (obstacle_detector is None):
            raise ValueError(
                "surface_analyzer and obstacle_detector must be injected together"
            )
        self.attribution_policy = attribution_policy
        if surface_analyzer is not None and obstacle_detector is not None:
            self.surface_analyzer = surface_analyzer
            self.obstacle_detector = obstacle_detector
            return
        root = (runtime_root or _runtime_root()).resolve()
        model_root = Path(
            os.environ.get("ROADPUSH_MODEL_ROOT", str(root / "models"))
        ).expanduser().resolve()
        surface_model_root = Path(
            os.environ.get(
                "SURFACE_PASSABILITY_MODEL_ROOT",
                str(model_root / "surface_passability"),
            )
        ).expanduser().resolve()
        surface_settings = RuntimeSettings(
            runtime_root=root,
            data_root=_configured_root("SURFACE_PASSABILITY_DATA_ROOT", root / "data"),
            model_root=surface_model_root,
            cache_root=_configured_root(
                "SURFACE_PASSABILITY_CACHE_ROOT",
                root / ".cache" / "surface_passability",
            ),
            artifact_root=_configured_root(
                "SURFACE_PASSABILITY_ARTIFACT_ROOT",
                root / "artifacts" / "surface_passability",
            ),
            report_root=_configured_root(
                "SURFACE_PASSABILITY_REPORT_ROOT",
                root / "reports" / "surface_passability",
            ),
        )
        self.surface_analyzer = SurfacePassabilityAnalyzer(
            requested_device=requested_device,
            settings=surface_settings,
        )
        obstacle_settings = ObstacleSettings.load(
            runtime_root=root,
            model_root=model_root,
        )
        self.obstacle_detector = ObstacleDetector(
            settings=obstacle_settings,
            requested_device=requested_device,
        )

    def analyze_photo(self, request: AnalysisRequest) -> RoadPushAnalysisResponse:
        started = perf_counter()
        surface_bundle = self.surface_analyzer.analyze_photo_with_evidence(request)
        after_surface = perf_counter()
        evidence = surface_bundle.evidence
        try:
            obstacle_result = self.obstacle_detector.detect(
                evidence.prepared_image.native_rgb
            )
        except Exception as exc:
            finished = perf_counter()
            return RoadPushAnalysisResponse(
                request_id=request.request_id,
                sample_id=request.sample_id,
                status="partial",
                surface=surface_bundle.response,
                surface_obstacles=SurfaceObstacleResponse(
                    status="failed",
                    obstacle_count=0,
                    types=(),
                    obstacles=(),
                    components=(),
                    auto_routable=False,
                    policy_id=self.attribution_policy.policy_id,
                    error={"type": type(exc).__name__, "message": str(exc)},
                ),
                diagnostics={
                    "all_detections": [],
                    "stage_timings_s": {
                        "surface": after_surface - started,
                        "total": finished - started,
                    },
                },
            )
        after_obstacle = perf_counter()
        all_detections = tuple(
            detection.to_dict()
            for detection in obstacle_result.detections
        )
        if evidence.selected_surface is None:
            finished = perf_counter()
            return self._compose(
                request=request,
                bundle=surface_bundle,
                obstacle_result=obstacle_result,
                status="no_selected_surface",
                obstacles=(),
                components=(),
                all_detections=all_detections,
                analysis_boxes={},
                started=started,
                after_surface=after_surface,
                after_obstacle=after_obstacle,
                finished=finished,
            )
        if evidence.phase7b_occupancy_mask is None:
            finished = perf_counter()
            return self._compose(
                request=request,
                bundle=surface_bundle,
                obstacle_result=obstacle_result,
                status="no_measurement_occupancy",
                obstacles=(),
                components=(),
                all_detections=all_detections,
                analysis_boxes={},
                started=started,
                after_surface=after_surface,
                after_obstacle=after_obstacle,
                finished=finished,
            )
        try:
            attribution = attribute_detections(
                evidence.phase7b_occupancy_mask,
                obstacle_result.detections,
                original_size=evidence.prepared_image.original_size,
                analysis_size=evidence.prepared_image.analysis_size,
                policy=self.attribution_policy,
            )
        except Exception as exc:
            finished = perf_counter()
            enriched = tuple(
                detection.to_dict()
                for detection in obstacle_result.detections
            )
            return RoadPushAnalysisResponse(
                request_id=request.request_id,
                sample_id=request.sample_id,
                status="partial",
                surface=surface_bundle.response,
                surface_obstacles=SurfaceObstacleResponse(
                    status="attribution_failed",
                    obstacle_count=0,
                    types=(),
                    obstacles=(),
                    components=(),
                    auto_routable=False,
                    policy_id=self.attribution_policy.policy_id,
                    error={"type": type(exc).__name__, "message": str(exc)},
                ),
                diagnostics=self._diagnostics(
                    obstacle_result,
                    enriched,
                    started,
                    after_surface,
                    after_obstacle,
                    finished,
                ),
            )
        matched = set(attribution.matched_detection_ids)
        surface_obstacles = tuple(
            detection.to_dict(
                box_analysis_xyxy=attribution.analysis_boxes[detection.detection_id]
            )
            for detection in obstacle_result.detections
            if detection.detection_id in matched
        )
        enriched_all = tuple(
            detection.to_dict(
                box_analysis_xyxy=attribution.analysis_boxes[detection.detection_id]
            )
            for detection in obstacle_result.detections
        )
        finished = perf_counter()
        return self._compose(
            request=request,
            bundle=surface_bundle,
            obstacle_result=obstacle_result,
            status=attribution.status,
            obstacles=surface_obstacles,
            components=attribution.components,
            all_detections=enriched_all,
            analysis_boxes=attribution.analysis_boxes,
            started=started,
            after_surface=after_surface,
            after_obstacle=after_obstacle,
            finished=finished,
        )

    def _compose(
        self,
        *,
        request: AnalysisRequest,
        bundle: SurfaceAnalysisBundle,
        obstacle_result: ObstacleDetectionResult,
        status: str,
        obstacles: tuple[dict[str, object], ...],
        components: tuple[dict[str, object], ...],
        all_detections: tuple[dict[str, object], ...],
        analysis_boxes: dict[int, tuple[float, float, float, float]],
        started: float,
        after_surface: float,
        after_obstacle: float,
        finished: float,
    ) -> RoadPushAnalysisResponse:
        del analysis_boxes
        types = tuple(sorted({str(obstacle["type"]) for obstacle in obstacles}))
        authoritative = (
            bundle.response.passability.handoff.result_authority == "authoritative"
        )
        return RoadPushAnalysisResponse(
            request_id=request.request_id,
            sample_id=request.sample_id,
            status="complete",
            surface=bundle.response,
            surface_obstacles=SurfaceObstacleResponse(
                status=status,
                obstacle_count=len(obstacles),
                types=types,
                obstacles=obstacles,
                components=components,
                auto_routable=bool(obstacles) and authoritative,
                policy_id=self.attribution_policy.policy_id,
            ),
            diagnostics=self._diagnostics(
                obstacle_result,
                all_detections,
                started,
                after_surface,
                after_obstacle,
                finished,
            ),
        )

    @staticmethod
    def _diagnostics(
        obstacle_result: ObstacleDetectionResult,
        all_detections: tuple[dict[str, object], ...],
        started: float,
        after_surface: float,
        after_obstacle: float,
        finished: float,
    ) -> dict[str, object]:
        return {
            "obstacle_model": {
                "version": obstacle_result.model_version,
                "sha256": obstacle_result.model_sha256,
                "device": obstacle_result.device,
            },
            "all_detection_count": len(all_detections),
            "all_detections": list(all_detections),
            "stage_timings_s": {
                "surface": after_surface - started,
                "obstacle": after_obstacle - after_surface,
                "attribution_and_response": finished - after_obstacle,
                "total": finished - started,
            },
        }


def _runtime_root() -> Path:
    value = os.environ.get("ROADPUSH_ROOT")
    if value:
        return Path(value).expanduser()
    return Path(__file__).resolve().parents[2]


def _configured_root(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()
