"""Explicit offline input binding and review-only result envelope."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .version import __version__

SCHEMA = "roadpush.lidar.analysis/0.1-dev"
POLICY = "max_path_min_width_m_ratio_at_same_bottleneck_section"


class ContractError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ContractError("A lowercase SHA-256 is required")
    return value


@dataclass(frozen=True)
class AnalysisRequest:
    request_id: str
    frame_id: str
    pcd_path: Path
    prediction_path: Path
    report_path: Path
    expected_pcd_sha256: str
    expected_prediction_sha256: str
    expected_report_sha256: str
    expected_checkpoint_sha256: str
    model_version: str
    source_receipt_sha256: str
    axis_evidence: dict[str, Any]
    boundary_profile: str = "v5_review"
    include_unknown_bounds: bool = False
    data_kind: str = "historical_actual_inference"

    def validate(self) -> None:
        if not all(
            isinstance(x, str) and x.strip()
            for x in (self.request_id, self.frame_id, self.model_version)
        ):
            raise ContractError("Explicit request/frame/model identity required")
        for v in (
            self.expected_pcd_sha256,
            self.expected_prediction_sha256,
            self.expected_report_sha256,
            self.expected_checkpoint_sha256,
            self.source_receipt_sha256,
        ):
            require_hash(v)
        if self.boundary_profile not in ("v5_review", "low_barrier_review"):
            raise ContractError(
                "Unsupported boundary profile; variants require explicit selection"
            )
        if type(self.include_unknown_bounds) is not bool:
            raise ContractError("Bounds flag must be bool")
        if self.data_kind not in ("historical_actual_inference", "actual_inference"):
            raise ContractError(
                "Actual prediction provenance required; synthetic tests are separate"
            )
        axis = self.axis_evidence
        if (
            not isinstance(axis, dict)
            or axis.get("metric_z_up_verified") is not True
            or not axis.get("source")
        ):
            raise ContractError(
                "Explicit metre/z-up evidence or declared project assumption required"
            )
        canonical(axis)


@dataclass(frozen=True)
class AnalysisResponse:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        canonical(self.payload)
        return deepcopy(self.payload)


def envelope(
    request: AnalysisRequest,
    *,
    spatial: dict[str, Any] | None,
    recovery: dict[str, Any] | None,
    conditional: dict[str, Any],
    diagnostics: dict[str, Any],
) -> AnalysisResponse:
    """Never promote partial hypotheses into a confirmed measurement."""
    binding = dict(
        pcd_sha256=request.expected_pcd_sha256,
        prediction_sha256=request.expected_prediction_sha256,
        report_sha256=request.expected_report_sha256,
        checkpoint_sha256=request.expected_checkpoint_sha256,
        source_receipt_sha256=request.source_receipt_sha256,
    )
    versions = dict(
        package=__version__,
        model=request.model_version,
        policy=POLICY,
        boundary_profile=request.boundary_profile,
    )
    runs = conditional.get("conditional_runs", [])
    value = dict(
        schema_version=SCHEMA,
        request_id=request.request_id,
        frame_id=request.frame_id,
        processing_status="completed",
        data_kind=request.data_kind,
        input=binding,
        versions=versions,
        passability="undetermined",
        confirmed_width_m=None,
        confirmed_ratio=None,
        result_authority="provisional_review" if runs else "none",
        measurement_scope="partial" if runs else "none",
        production_eligible=False,
        human_review_required=True,
        auto_aggregation_allowed=False,
        conditional_runs=deepcopy(runs),
        reasons=[
            conditional.get("state", "measurement_unavailable"),
            "boundary_identity_and_width_accuracy_not_verified",
        ],
        evidence=dict(
            spatial=spatial,
            boundary_recovery=recovery,
            conditional=conditional,
            diagnostics=diagnostics,
        ),
        metadata=dict(actual_capture_time=None, actual_location=None),
        computation_key=hashlib.sha256(
            canonical(
                dict(
                    input=binding,
                    versions=versions,
                    bounds=request.include_unknown_bounds,
                    axis=request.axis_evidence,
                )
            )
        ).hexdigest(),
    )
    canonical(value)
    return AnalysisResponse(value)
