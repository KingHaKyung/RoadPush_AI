from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import cv2
import numpy as np

from roadpush_ai.phase7a.extension_s1 import (
    S1AGenerationPolicy,
    freeze_v6_observation,
    generate_s1a_proposal,
    grade_s1_v4_probe_result,
    mask_sha256,
    evaluate_s1_targeted_v4c,
)
from roadpush_ai.phase7a.validation_policy import (
    BANNER_CLASS_ID,
    V4B_OCCLUDER_CLASS_IDS,
    _v4b_occluder_mask,
)
from roadpush_ai.phase7a.reference_reconstruction import (
    REFERENCE,
    _v6_targeted,
)


REPOSITORY = get_settings().project_root
DEFAULT_PHASE7A_RESULTS = (
    REPOSITORY
    / "outputs"
    / "surface_generalization"
    / "phase7_extension_track_s1_full160_v4_probe_final"
    / "results.json"
)

MeasurementScope = Literal["full", "partial", "provisional", "none"]
MeasurementAuthority = Literal[
    "allowed", "conditional_on_v6_reliability", "review_only", "denied"
]
ResultAuthority = Literal["authoritative", "provisional_review", "none"]


@dataclass(frozen=True)
class Phase7AStatusRoute:
    measurement_scope: MeasurementScope
    measurement_authority: MeasurementAuthority
    result_authority: ResultAuthority
    human_review_required: bool
    abstention_reason: str | None

    @property
    def calculation_allowed(self) -> bool:
        return self.measurement_authority != "denied"


@dataclass(frozen=True)
class Phase7BTypedNullResult:
    schema_version: str
    sample_id: str
    phase7a_policy_id: str
    phase7a_status: str
    result_authority: Literal["none"]
    selected_surface: str | None
    measurement_scope: Literal["none"]
    measurement_confidence: Literal["insufficient"]
    human_review_required: bool
    abstention_reasons: list[str]
    minimum_passable_ratio: None
    raw_minimum_passable_ratio: None
    maximum_obstructed_width_ratio: None
    obstacle_area_ratio: None
    bottleneck: None
    surface_width_m: None
    absolute_passable_width_m: None
    effective_blocked_width_m: None
    agent_width_m: None
    safety_factor: None
    required_safe_width_m: None
    required_passable_ratio: None
    required_safe_ratio: None
    passability_state: Literal["undetermined"]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase7BHandoff:
    sample_id: str
    selected_surface: str | None
    phase7a_policy_id: str
    phase7a_status: str
    measurement_scope: MeasurementScope
    measurement_authority: MeasurementAuthority
    result_authority: ResultAuthority
    human_review_required: bool
    authorized_row_range: tuple[int, int] | None
    observed_rows: tuple[int, ...]
    generated_rows: tuple[int, ...]
    structural_mask_sha256: str
    structural_area_px: int
    provenance: dict[str, Any]
    structural_mask: np.ndarray | None
    typed_null_result: Phase7BTypedNullResult | None

    def to_record(self, *, mask_path: Path | None = None) -> dict[str, Any]:
        record = {
            "sample_id": self.sample_id,
            "selected_surface": self.selected_surface,
            "phase7a_policy_id": self.phase7a_policy_id,
            "phase7a_status": self.phase7a_status,
            "measurement_scope": self.measurement_scope,
            "measurement_authority": self.measurement_authority,
            "result_authority": self.result_authority,
            "human_review_required": self.human_review_required,
            "authorized_row_range": (
                list(self.authorized_row_range) if self.authorized_row_range else None
            ),
            "observed_rows": list(self.observed_rows),
            "generated_rows": list(self.generated_rows),
            "structural_mask_sha256": self.structural_mask_sha256,
            "structural_area_px": self.structural_area_px,
            "structural_mask": str(mask_path.resolve()) if mask_path else None,
            "structural_mask_file_sha256": (
                hashlib.sha256(mask_path.read_bytes()).hexdigest() if mask_path else None
            ),
            "provenance": self.provenance,
            "typed_null_result": (
                self.typed_null_result.to_dict() if self.typed_null_result else None
            ),
        }
        return record


STATUS_ROUTES: dict[str, Phase7AStatusRoute] = {
    "selected": Phase7AStatusRoute(
        "full", "allowed", "authoritative", False, None
    ),
    "unchanged": Phase7AStatusRoute(
        "full",
        "conditional_on_v6_reliability",
        "authoritative",
        False,
        None,
    ),
    "selected_partial": Phase7AStatusRoute(
        "partial", "allowed", "authoritative", True, None
    ),
    "ambiguous_hold": Phase7AStatusRoute(
        "provisional", "review_only", "provisional_review", True, None
    ),
    "failed": Phase7AStatusRoute(
        "none", "denied", "none", False, "phase7a_failed"
    ),
}


def route_phase7a_status(status: str) -> Phase7AStatusRoute:
    try:
        return STATUS_ROUTES[status]
    except KeyError as error:
        raise ValueError(f"Unsupported Phase 7-A status: {status}") from error


def _typed_null_result(
    authoritative: Mapping[str, Any],
    route: Phase7AStatusRoute,
    *,
    selected_surface: str | None,
) -> Phase7BTypedNullResult:
    decision = authoritative["s1_decision"]
    reason = route.abstention_reason
    if reason is None:
        raise ValueError("Typed nulls are only valid for denied status routes")
    return Phase7BTypedNullResult(
        schema_version="phase7_b_result_v2",
        sample_id=str(authoritative["sample_id"]),
        phase7a_policy_id=str(decision["policy_id"]),
        phase7a_status=str(decision["s1_status"]),
        result_authority="none",
        selected_surface=selected_surface,
        measurement_scope="none",
        measurement_confidence="insufficient",
        human_review_required=route.human_review_required,
        abstention_reasons=[reason],
        minimum_passable_ratio=None,
        raw_minimum_passable_ratio=None,
        maximum_obstructed_width_ratio=None,
        obstacle_area_ratio=None,
        bottleneck=None,
        surface_width_m=None,
        absolute_passable_width_m=None,
        effective_blocked_width_m=None,
        agent_width_m=None,
        safety_factor=None,
        required_safe_width_m=None,
        required_passable_ratio=None,
        required_safe_ratio=None,
        passability_state="undetermined",
        flags=[reason],
    )


def _row_ids(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(y) for y in np.flatnonzero(np.any(mask, axis=1)))


class Phase7AAdapter:
    """Read the frozen Phase 7-A disposition and reproduce its mask exactly.

    The adapter never trusts the legacy V4c PNG as authoritative. It reconstructs
    the approved V4c+Banner+grading result from frozen inputs and verifies the
    logical mask hash stored in the final Phase 7-A result before exposing a mask
    to Phase 7-B.
    """

    def __init__(
        self,
        results_path: Path = DEFAULT_PHASE7A_RESULTS,
        reference_path: Path = REFERENCE / "results.json",
    ) -> None:
        self.results_path = Path(results_path)
        self.reference_path = Path(reference_path)
        authoritative = json.loads(self.results_path.read_text(encoding="utf-8"))
        references = json.loads(self.reference_path.read_text(encoding="utf-8"))
        self._authoritative = {row["sample_id"]: row for row in authoritative}
        self._references = {row["sample_id"]: row for row in references}
        if len(self._authoritative) != len(authoritative):
            raise ValueError("Duplicate sample_id in authoritative Phase 7-A results")
        if set(self._authoritative) - set(self._references):
            raise ValueError("Phase 7-A sample is missing from the frozen V6 reference")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self._authoritative)

    def authoritative_record(self, sample_id: str) -> dict[str, Any]:
        try:
            return self._authoritative[sample_id]
        except KeyError as error:
            raise KeyError(f"Unknown Phase 7-A sample: {sample_id}") from error

    def reconstruct(self, sample_id: str) -> Phase7BHandoff:
        authoritative = self.authoritative_record(sample_id)
        reference = self._references[sample_id]
        saved_decision = authoritative["s1_decision"]
        status = str(saved_decision["s1_status"])
        route = route_phase7a_status(status)

        v6, _v6_masks, structural, blocker = _v6_targeted(reference)
        package = freeze_v6_observation(sample_id, v6, structural.shape)
        diagnostic, generation_masks = generate_s1a_proposal(
            package,
            structural,
            blocker,
            S1AGenerationPolicy(
                stop_before_side_frame_contact=True,
                minimum_resolvable_width_px=12,
                minimum_width_uncertainty_multiplier=2.0,
            ),
        )
        obstacle = _v4b_occluder_mask(
            reference, structural.shape, include_banner=True
        )
        decision, selected = evaluate_s1_targeted_v4c(
            package, diagnostic, generation_masks, obstacle
        )
        decision["policy_id"] = f'{decision["policy_id"]}_banner32'
        decision["banner_occluder_enabled"] = True
        decision["occluder_class_ids"] = [
            *V4B_OCCLUDER_CLASS_IDS,
            BANNER_CLASS_ID,
        ]
        decision, selected = grade_s1_v4_probe_result(
            package,
            decision,
            selected,
            generation_masks,
            top_bottom_width_ratio=v6.get("envelope_fit", {}).get(
                "top_bottom_width_ratio"
            ),
        )
        effective = np.asarray(selected["effective_selected"], dtype=bool)
        actual_hash = mask_sha256(effective)
        expected_hash = str(saved_decision["effective_selected_mask_sha256"])
        if decision["s1_status"] != status:
            raise AssertionError(
                f"{sample_id}: reconstructed status {decision['s1_status']} "
                f"does not match authoritative {status}"
            )
        if actual_hash != expected_hash:
            raise AssertionError(
                f"{sample_id}: reconstructed mask hash {actual_hash} "
                f"does not match authoritative {expected_hash}"
            )

        observed_mask = effective & np.asarray(
            generation_masks["observed_seed"], dtype=bool
        )
        generated_source = np.asarray(
            selected.get("validated_extension", generation_masks["extension_only"]),
            dtype=bool,
        )
        generated_mask = effective & generated_source
        effective_rows = _row_ids(effective)
        authorized_range = (
            (effective_rows[0], effective_rows[-1])
            if route.calculation_allowed and effective_rows
            else None
        )
        null_result = (
            None
            if route.calculation_allowed
            else _typed_null_result(
                authoritative,
                route,
                selected_surface=package.selected_surface,
            )
        )
        saved_phase6 = reference.get("saved_phase6_artifacts") or {}
        provenance = {
            "source_image": authoritative["source_image"],
            "phase7a_results": str(self.results_path.resolve()),
            "phase7a_policy_id": saved_decision["policy_id"],
            "phase7a_selected_model": saved_decision.get("s1_selected_model"),
            "phase7a_failure_reasons": saved_decision.get("s1_failure_reasons", []),
            "phase7a_quality_grading_reasons": saved_decision.get(
                "quality_grading_reasons", []
            ),
            "phase7a_termination": {
                "far_upper": diagnostic.get("far_upper"),
                "near_lower": diagnostic.get("near_lower"),
                "v4c_truncation": saved_decision.get("v4c_truncation"),
            },
            "v6_policy_id": authoritative["v6_policy_id"],
            "v6_final_disposition": authoritative["v6_final_disposition"],
            "v6_subclassification": authoritative["v6_subclassification"],
            "phase6_blocker_path": saved_phase6.get("phase6_blocker_path"),
            "phase6_structural_path": saved_phase6.get("phase6_structural_path"),
            "reconstruction": "frozen_v6_to_v4c_banner32_to_v4_probe_grading",
        }
        return Phase7BHandoff(
            sample_id=sample_id,
            selected_surface=package.selected_surface,
            phase7a_policy_id=str(saved_decision["policy_id"]),
            phase7a_status=status,
            measurement_scope=route.measurement_scope,
            measurement_authority=route.measurement_authority,
            result_authority=route.result_authority,
            human_review_required=route.human_review_required,
            authorized_row_range=authorized_range,
            observed_rows=_row_ids(observed_mask),
            generated_rows=_row_ids(generated_mask),
            structural_mask_sha256=actual_hash,
            structural_area_px=int(effective.sum()),
            provenance=provenance,
            structural_mask=(effective.copy() if route.calculation_allowed else None),
            typed_null_result=null_result,
        )


def materialize_authoritative_handoffs(
    output: Path,
    *,
    adapter: Phase7AAdapter | None = None,
    sample_ids: tuple[str, ...] | None = None,
) -> Path:
    """Materialize verified masks and one audit manifest without rendering."""

    adapter = adapter or Phase7AAdapter()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7-B adapter output: {output}")
    mask_dir = output / "authoritative_phase7a_masks"
    mask_dir.mkdir(parents=True)
    records = []
    for sample_id in sample_ids or adapter.sample_ids:
        handoff = adapter.reconstruct(sample_id)
        # Failed masks are reconstructed and hash-verified above but remain
        # unavailable. Ambiguous masks are review-only and are materialized as
        # provisional candidates, never as authoritative selected geometry.
        authoritative = adapter.authoritative_record(sample_id)
        expected_hash = authoritative["s1_decision"]["effective_selected_mask_sha256"]
        if handoff.structural_mask is not None:
            mask = handoff.structural_mask
        else:
            # Reconstruct again is intentionally avoided for failed results.
            records.append(handoff.to_record(mask_path=None))
            continue
        if mask_sha256(mask) != expected_hash:
            raise AssertionError(f"{sample_id}: mask changed before materialization")
        mask_path = mask_dir / f"{sample_id}.png"
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise OSError(mask_path)
        reread = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reread is None or mask_sha256(reread > 0) != expected_hash:
            raise AssertionError(f"{sample_id}: materialized PNG logical hash mismatch")
        records.append(handoff.to_record(mask_path=mask_path))

    manifest = output / "handoffs.json"
    manifest.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append")
    arguments = parser.parse_args()
    requested = tuple(arguments.sample_id) if arguments.sample_id else None
    print(
        materialize_authoritative_handoffs(
            arguments.output,
            sample_ids=requested,
        ).resolve()
    )
