"""Explicit runtime dependencies for the production orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Protocol

from surface_passability.config.artifacts import ArtifactLocator, ArtifactRef
from surface_passability.config.model_profiles import (
    load_mask2former_mapillary_class_names,
    verify_mask2former_mapillary_model,
)
from surface_passability.config.settings import RuntimeSettings, get_settings
from surface_passability.contracts.image import PreparedImage
from surface_passability.contracts.semantic import SemanticEvidence
from surface_passability.policies.selection import SurfaceSelectionPolicy

class SemanticEvidenceSource(Protocol):
    def provide(self, prepared: PreparedImage) -> SemanticEvidence: ...


class VehicleModelSource(Protocol):
    def get_model(self) -> object: ...


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """All impure providers and configurable policies used by `analyze_photo`."""

    semantic_provider: SemanticEvidenceSource
    vehicle_model_provider: VehicleModelSource
    class_names: Mapping[int | str, str]
    artifact_locator: ArtifactLocator | None = None
    selection_policy: SurfaceSelectionPolicy = SurfaceSelectionPolicy()
    clock: Callable[[], float] = perf_counter


YOLO11N_SEG_SHA256 = (
    "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
)
SEMANTIC_MODEL_DIRECTORY = PurePosixPath("semantic")


def create_default_dependencies(
    *,
    requested_device: str | None = None,
    settings: RuntimeSettings | None = None,
) -> PipelineDependencies:
    """Build verified, relocatable production providers without loading models."""

    runtime = settings or get_settings()
    locator = ArtifactLocator(runtime)
    semantic_root = runtime.model_root / Path(*SEMANTIC_MODEL_DIRECTORY.parts)
    identity = verify_mask2former_mapillary_model(semantic_root)
    class_names = load_mask2former_mapillary_class_names(
        semantic_root,
        verified_identity=identity,
    )
    device = (
        os.environ.get("SURFACE_PASSABILITY_DEVICE", "cpu")
        if requested_device is None
        else requested_device
    )
    if not device.strip():
        raise ValueError("requested inference device must not be empty")
    vehicle_weights = locator.resolve_read(
        ArtifactRef(
            namespace="model",
            artifact_id="yolo11n-seg-coco",
            version="accepted-v1",
            relative_path=PurePosixPath("yolo11n-seg.pt"),
            sha256=YOLO11N_SEG_SHA256,
        )
    )
    from surface_passability.inference.semantic_cache import SemanticEvidenceProvider
    from surface_passability.inference.vehicle_instances import (
        UltralyticsVehicleModelProvider,
    )

    return PipelineDependencies(
        semantic_provider=SemanticEvidenceProvider(
            model_root=semantic_root,
            cache_root=runtime.cache_root,
            requested_device=device,
            model_identity=identity,
        ),
        vehicle_model_provider=UltralyticsVehicleModelProvider(
            vehicle_weights,
            requested_device=requested_device,
        ),
        class_names=class_names,
        artifact_locator=locator,
    )


def get_default_dependencies(
    *,
    requested_device: str | None = None,
    settings: RuntimeSettings | None = None,
) -> PipelineDependencies:
    """Return one verified, reusable dependency graph per worker/runtime."""

    runtime = settings or get_settings()
    device = (
        os.environ.get("SURFACE_PASSABILITY_DEVICE", "cpu")
        if requested_device is None
        else requested_device
    )
    if not device.strip():
        raise ValueError("requested inference device must not be empty")
    return _cached_default_dependencies(runtime, device)


@lru_cache(maxsize=8)
def _cached_default_dependencies(
    settings: RuntimeSettings,
    requested_device: str,
) -> PipelineDependencies:
    return create_default_dependencies(
        requested_device=requested_device,
        settings=settings,
    )


def reset_default_dependencies_cache() -> None:
    """Drop process-local runtime owners after an explicit reconfiguration."""

    _cached_default_dependencies.cache_clear()
