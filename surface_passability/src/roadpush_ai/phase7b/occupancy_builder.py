from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from roadpush_ai.phase7a.extension_s1 import mask_sha256
from roadpush_ai.phase7b.occupancy_policy import build_passability_occupancy_mask
from roadpush_ai.phase7b.handoff import Phase7BHandoff


REPOSITORY = get_settings().project_root
DEFAULT_RUNTIME_EVIDENCE = (
    REPOSITORY
    / "outputs"
    / "surface_generalization"
    / "phase7_targeted_runtime_probe_v2"
    / "results.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value > 0


@dataclass(frozen=True)
class ExistingOccupancyEvidence:
    sample_id: str
    selected_surface: str
    occupancy_path: Path
    occupancy_file_sha256: str
    surface_blocker_pixels: int
    occupancy_pixels: int
    instances: tuple[dict[str, Any], ...]
    semantic_inference_run: bool


class ExistingOccupancyEvidenceIndex:
    """Index only already-computed, persisted measurement evidence."""

    def __init__(self, results_path: Path = DEFAULT_RUNTIME_EVIDENCE) -> None:
        self.results_path = Path(results_path)
        rows = json.loads(self.results_path.read_text(encoding="utf-8"))
        self._rows = {str(row["sample_id"]): row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError("Duplicate sample_id in existing occupancy evidence")

    def get(self, sample_id: str) -> ExistingOccupancyEvidence | None:
        row = self._rows.get(sample_id)
        if row is None:
            return None
        occupancy_path = (
            self.results_path.parent
            / "artifacts"
            / sample_id
            / "passability_occupancy.png"
        )
        if not occupancy_path.is_file():
            raise FileNotFoundError(occupancy_path)
        return ExistingOccupancyEvidence(
            sample_id=sample_id,
            selected_surface=str(row["selection"]["selected_surface"]),
            occupancy_path=occupancy_path,
            occupancy_file_sha256=_file_sha256(occupancy_path),
            surface_blocker_pixels=int(row["surface_blocker_pixels"]),
            occupancy_pixels=int(row["passability_occupancy_pixels"]),
            instances=tuple(dict(value) for value in row["passability_instances"]),
            semantic_inference_run=bool(row["semantic_inference_run"]),
        )


@dataclass(frozen=True)
class Phase7BOccupancyResult:
    sample_id: str
    coverage_status: str
    abstention_reasons: tuple[str, ...]
    occupancy_mask: np.ndarray | None
    occupancy_mask_sha256: str | None
    occupancy_pixels: int | None
    base_blocker_pixels: int | None
    added_completion_pixels: int | None
    clipped_blocker_pixels: int | None
    instance_records: tuple[dict[str, Any], ...]
    evidence_source: str
    upstream_hashes_before: dict[str, str]
    upstream_hashes_after: dict[str, str]

    @property
    def measurement_allowed(self) -> bool:
        return self.occupancy_mask is not None and not self.abstention_reasons

    def to_record(self, *, occupancy_path: Path | None = None) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "coverage_status": self.coverage_status,
            "measurement_allowed": self.measurement_allowed,
            "abstention_reasons": list(self.abstention_reasons),
            "occupancy_mask_sha256": self.occupancy_mask_sha256,
            "occupancy_mask": (
                str(occupancy_path.resolve()) if occupancy_path is not None else None
            ),
            "occupancy_mask_file_sha256": (
                _file_sha256(occupancy_path) if occupancy_path is not None else None
            ),
            "occupancy_pixels": self.occupancy_pixels,
            "base_blocker_pixels": self.base_blocker_pixels,
            "added_completion_pixels": self.added_completion_pixels,
            "clipped_blocker_pixels": self.clipped_blocker_pixels,
            "instance_records": list(self.instance_records),
            "evidence_source": self.evidence_source,
            "upstream_hashes_before": self.upstream_hashes_before,
            "upstream_hashes_after": self.upstream_hashes_after,
        }


def build_measurement_occupancy(
    handoff: Phase7BHandoff,
    *,
    v6_final_disposition: str,
    v6_subclassification: str,
    evidence_index: ExistingOccupancyEvidenceIndex | None = None,
    passability_instances: Sequence[Mapping[str, Any]] | None = None,
) -> Phase7BOccupancyResult:
    """Copy and complete occupancy without modifying an upstream mask.

    Persisted runtime evidence and caller-provided instances are accepted only
    when they already exist. This function never runs inference or ownership.
    """

    if handoff.structural_mask is None:
        reason = (
            handoff.typed_null_result.abstention_reasons[0]
            if handoff.typed_null_result is not None
            else "phase7a_measurement_not_authorized"
        )
        return Phase7BOccupancyResult(
            sample_id=handoff.sample_id,
            coverage_status="not_authorized",
            abstention_reasons=(reason,),
            occupancy_mask=None,
            occupancy_mask_sha256=None,
            occupancy_pixels=None,
            base_blocker_pixels=None,
            added_completion_pixels=None,
            clipped_blocker_pixels=None,
            instance_records=(),
            evidence_source="none",
            upstream_hashes_before={},
            upstream_hashes_after={},
        )

    structural = np.asarray(handoff.structural_mask, dtype=bool)
    blocker_value = handoff.provenance.get("phase6_blocker_path")
    if not blocker_value:
        return Phase7BOccupancyResult(
            sample_id=handoff.sample_id,
            coverage_status="unverified",
            abstention_reasons=("occupancy_coverage_unverified",),
            occupancy_mask=None,
            occupancy_mask_sha256=None,
            occupancy_pixels=None,
            base_blocker_pixels=None,
            added_completion_pixels=None,
            clipped_blocker_pixels=None,
            instance_records=(),
            evidence_source="missing_phase6_blocker",
            upstream_hashes_before={},
            upstream_hashes_after={},
        )

    blocker_path = Path(str(blocker_value))
    blocker_file_hash_before = _file_sha256(blocker_path)
    blocker_source = _read_mask(blocker_path)
    if blocker_source.shape != structural.shape:
        raise ValueError(
            f"{handoff.sample_id}: blocker shape {blocker_source.shape} "
            f"does not match structural shape {structural.shape}"
        )
    blocker_logical_hash_before = mask_sha256(blocker_source)
    # This is the working copy required by the contract. Neither this array nor
    # later completion aliases the upstream image returned by cv2.
    working = blocker_source.copy() & structural
    base_pixels = int(working.sum())
    clipped_pixels = int(blocker_source.sum() - base_pixels)
    records: tuple[dict[str, Any], ...] = ()
    evidence_source = "frozen_phase6_blocker"

    explicit_suspect = (
        v6_final_disposition == "boundary_valid_but_blocker_suspect"
        or v6_subclassification == "blocker_coverage_suspect"
    )
    supplied_instances = list(passability_instances or [])
    persisted = evidence_index.get(handoff.sample_id) if evidence_index else None

    if supplied_instances:
        completed, completed_records = build_passability_occupancy_mask(
            structural,
            working,
            [dict(instance) for instance in supplied_instances],
        )
        working = completed
        records = tuple(completed_records)
        evidence_source = "existing_runtime_instance_ownership"
    elif persisted is not None:
        if persisted.semantic_inference_run:
            raise ValueError(
                f"{handoff.sample_id}: persisted occupancy unexpectedly ran inference"
            )
        if persisted.selected_surface != handoff.selected_surface:
            raise ValueError(
                f"{handoff.sample_id}: persisted occupancy surface mismatch"
            )
        persisted_mask = _read_mask(persisted.occupancy_path)
        if persisted_mask.shape != structural.shape:
            raise ValueError(
                f"{handoff.sample_id}: persisted occupancy shape mismatch"
            )
        if int(persisted_mask.sum()) != persisted.occupancy_pixels:
            raise AssertionError(
                f"{handoff.sample_id}: persisted occupancy pixel count changed"
            )
        working |= persisted_mask & structural
        records = persisted.instances
        evidence_source = "persisted_runtime_v2_occupancy"
    elif explicit_suspect:
        blocker_file_hash_after = _file_sha256(blocker_path)
        blocker_logical_hash_after = mask_sha256(_read_mask(blocker_path))
        if (
            blocker_file_hash_after != blocker_file_hash_before
            or blocker_logical_hash_after != blocker_logical_hash_before
        ):
            raise AssertionError(f"{handoff.sample_id}: upstream blocker was modified")
        return Phase7BOccupancyResult(
            sample_id=handoff.sample_id,
            coverage_status="unverified",
            abstention_reasons=("occupancy_coverage_unverified",),
            occupancy_mask=None,
            occupancy_mask_sha256=None,
            occupancy_pixels=None,
            base_blocker_pixels=base_pixels,
            added_completion_pixels=None,
            clipped_blocker_pixels=clipped_pixels,
            instance_records=(),
            evidence_source="explicit_blocker_coverage_suspect_without_instance_evidence",
            upstream_hashes_before={
                "phase6_blocker_file_sha256": blocker_file_hash_before,
                "phase6_blocker_mask_sha256": blocker_logical_hash_before,
                "phase7a_structural_mask_sha256": handoff.structural_mask_sha256,
            },
            upstream_hashes_after={
                "phase6_blocker_file_sha256": blocker_file_hash_after,
                "phase6_blocker_mask_sha256": blocker_logical_hash_after,
                "phase7a_structural_mask_sha256": mask_sha256(structural),
            },
        )

    working &= structural
    blocker_file_hash_after = _file_sha256(blocker_path)
    blocker_logical_hash_after = mask_sha256(_read_mask(blocker_path))
    structural_hash_after = mask_sha256(structural)
    if (
        blocker_file_hash_after != blocker_file_hash_before
        or blocker_logical_hash_after != blocker_logical_hash_before
        or structural_hash_after != handoff.structural_mask_sha256
    ):
        raise AssertionError(f"{handoff.sample_id}: an upstream mask was modified")
    occupancy_pixels = int(working.sum())
    return Phase7BOccupancyResult(
        sample_id=handoff.sample_id,
        coverage_status=(
            "completed_existing_instance_evidence"
            if evidence_source != "frozen_phase6_blocker"
            else "verified_frozen_blocker"
        ),
        abstention_reasons=(),
        occupancy_mask=working.copy(),
        occupancy_mask_sha256=mask_sha256(working),
        occupancy_pixels=occupancy_pixels,
        base_blocker_pixels=base_pixels,
        added_completion_pixels=occupancy_pixels - base_pixels,
        clipped_blocker_pixels=clipped_pixels,
        instance_records=records,
        evidence_source=evidence_source,
        upstream_hashes_before={
            "phase6_blocker_file_sha256": blocker_file_hash_before,
            "phase6_blocker_mask_sha256": blocker_logical_hash_before,
            "phase7a_structural_mask_sha256": handoff.structural_mask_sha256,
        },
        upstream_hashes_after={
            "phase6_blocker_file_sha256": blocker_file_hash_after,
            "phase6_blocker_mask_sha256": blocker_logical_hash_after,
            "phase7a_structural_mask_sha256": structural_hash_after,
        },
    )


def materialize_focused_occupancy_audit(
    output: Path,
    sample_ids: Sequence[str],
) -> Path:
    """Build focused Step-2 evidence only; no row or corridor measurement."""

    from roadpush_ai.phase7b.handoff import Phase7AAdapter

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite occupancy audit: {output}")
    mask_dir = output / "measurement_occupancy_masks"
    mask_dir.mkdir(parents=True)
    adapter = Phase7AAdapter()
    evidence_index = ExistingOccupancyEvidenceIndex()
    records = []
    for sample_id in sample_ids:
        authoritative = adapter.authoritative_record(sample_id)
        handoff = adapter.reconstruct(sample_id)
        result = build_measurement_occupancy(
            handoff,
            v6_final_disposition=authoritative["v6_final_disposition"],
            v6_subclassification=authoritative["v6_subclassification"],
            evidence_index=evidence_index,
        )
        occupancy_path = None
        if result.occupancy_mask is not None:
            occupancy_path = mask_dir / f"{sample_id}.png"
            if not cv2.imwrite(
                str(occupancy_path), result.occupancy_mask.astype(np.uint8) * 255
            ):
                raise OSError(occupancy_path)
            reread = _read_mask(occupancy_path)
            if mask_sha256(reread) != result.occupancy_mask_sha256:
                raise AssertionError(
                    f"{sample_id}: materialized occupancy logical hash mismatch"
                )
        record = result.to_record(occupancy_path=occupancy_path)
        persisted = evidence_index.get(sample_id)
        record["existing_evidence"] = (
            {
                "occupancy_path": str(persisted.occupancy_path.resolve()),
                "occupancy_file_sha256": persisted.occupancy_file_sha256,
                "surface_blocker_pixels": persisted.surface_blocker_pixels,
                "occupancy_pixels": persisted.occupancy_pixels,
                "semantic_inference_run": persisted.semantic_inference_run,
                "instances": list(persisted.instances),
            }
            if persisted is not None
            else None
        )
        records.append(record)
    destination = output / "results.json"
    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    arguments = parser.parse_args()
    print(
        materialize_focused_occupancy_audit(
            arguments.output,
            tuple(dict.fromkeys(arguments.sample_id)),
        ).resolve()
    )
