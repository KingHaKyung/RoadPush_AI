"""Public successful-response contract for one complete analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any

from surface_passability.config.artifacts import ArtifactRef
from surface_passability.contracts.boundary import BoundaryAnalysisResult
from surface_passability.contracts.image import PreparedImage
from surface_passability.contracts.passability import PassabilityResult
from surface_passability.contracts.semantic import SemanticEvidence
from surface_passability.contracts.surfaces import SurfaceSelectionResult


ANALYSIS_RESPONSE_SCHEMA_VERSION = "surface_passability_response_v1"


@dataclass(frozen=True, slots=True)
class AnalysisResponse:
    """Complete typed result returned by the surface-passability model."""

    request_id: str
    sample_id: str
    image: PreparedImage
    semantic: SemanticEvidence
    surface_selection: SurfaceSelectionResult
    boundary: BoundaryAnalysisResult
    passability: PassabilityResult
    artifacts: tuple[ArtifactRef, ...]
    stage_timings_s: dict[str, float]
    schema_version: str = ANALYSIS_RESPONSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for name, value in self.stage_timings_s.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("stage timing names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"stage timing {name!r} must be a real number")
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(
                    f"stage timing {name!r} must be finite and non-negative"
                )
            normalized[name] = seconds
        object.__setattr__(self, "stage_timings_s", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the public response without embedding pixel arrays."""

        handoff = self.passability.handoff
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "sample_id": self.sample_id,
            "image": {
                "source_uri": self.image.source_uri,
                "content_sha256": self.image.content_sha256,
                "original_size": list(self.image.original_size),
                "analysis_size": list(self.image.analysis_size),
                "exif_orientation_normalized": (
                    self.image.exif_orientation_normalized
                ),
                "resize_filter": self.image.resize_filter,
            },
            "semantic": {
                "native_array_sha256": self.semantic.native_array_sha256,
                "analysis_array_sha256": self.semantic.analysis_array_sha256,
                "cache_key": self.semantic.cache_key,
                "cache_file_sha256": self.semantic.cache_file_sha256,
                "cache_hit": self.semantic.cache_hit,
                "inference_run": self.semantic.inference_run,
                "cache_reuse_policy": self.semantic.cache_reuse_policy,
                "requested_device": self.semantic.requested_device,
                "producer_provenance": self.semantic.producer_provenance,
                "model": asdict(self.semantic.model),
            },
            "surface_selection": self.surface_selection.to_dict(),
            "boundary": {
                "policy_id": handoff.boundary_policy_id,
                "status": handoff.boundary_status,
                "selected_surface": handoff.selected_surface,
                "measurement_scope": handoff.measurement_scope,
                "measurement_authority": handoff.measurement_authority,
                "result_authority": handoff.result_authority,
                "human_review_required": handoff.human_review_required,
                "authorized_row_range": (
                    None
                    if handoff.authorized_row_range is None
                    else list(handoff.authorized_row_range)
                ),
                "observed_rows": list(handoff.observed_rows),
                "generated_rows": list(handoff.generated_rows),
                "structural_mask_sha256": handoff.structural_mask_sha256,
                "structural_area_px": handoff.structural_area_px,
                "decision": self.boundary.decision,
            },
            "passability": self.passability.record,
            "artifacts": [
                {
                    "namespace": ref.namespace,
                    "artifact_id": ref.artifact_id,
                    "version": ref.version,
                    "relative_path": ref.relative_path.as_posix(),
                    "sha256": ref.sha256,
                }
                for ref in self.artifacts
            ],
            "stage_timings_s": dict(self.stage_timings_s),
        }
