"""Build measurement occupancy from in-memory owned obstacle evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from surface_passability.contracts.passability import (
    MeasurementOccupancyResult,
    PassabilityBoundaryHandoff,
)
from surface_passability.io.hashing import hash_mask_after_uint8_cast
from surface_passability.passability.persisted_evidence import (
    PersistedOccupancyEvidence,
)


def complete_instance_occupancy(
    structural_mask: np.ndarray,
    surface_blocker_mask: np.ndarray,
    passability_instances: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Bridge each owned instance over its grounded lateral contact extent."""

    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(surface_blocker_mask, dtype=bool) & structural
    occupancy = np.zeros_like(structural)
    assigned = np.zeros_like(structural)
    records: list[dict[str, Any]] = []
    for instance_id, instance in enumerate(passability_instances, 1):
        selected = np.asarray(instance["selected_mask"], dtype=bool) & structural
        if not selected.any():
            continue
        ys, xs = np.where(selected)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        envelope = np.zeros_like(structural)
        for y in range(y0, y1 + 1):
            surface_xs = np.flatnonzero(structural[y])
            if not len(surface_xs):
                continue
            left = max(x0, int(surface_xs[0]))
            right = min(x1, int(surface_xs[-1]))
            if left <= right:
                envelope[y, left : right + 1] = True
        occupancy |= envelope
        assigned |= selected
        records.append(
            {
                "id": instance_id,
                "type": instance.get("class_name") or "unknown_obstacle",
                "kind": instance.get("kind"),
                "selected_pixels": int(selected.sum()),
                "occupancy_pixels": int(envelope.sum()),
                "contact_bbox_xyxy": [x0, y0, x1, y1],
            }
        )
    occupancy |= blocker & ~assigned
    occupancy &= structural
    return occupancy, records


def build_measurement_occupancy(
    handoff: PassabilityBoundaryHandoff,
    surface_blocker_mask: np.ndarray | None,
    passability_instances: Sequence[Mapping[str, Any]] | None = None,
    *,
    persisted_evidence: PersistedOccupancyEvidence | None = None,
    blocker_coverage_suspect: bool = False,
) -> MeasurementOccupancyResult:
    """Build verified measurement occupancy without reloading persisted files."""

    if handoff.structural_mask is None:
        reason = (
            handoff.typed_null_result.abstention_reasons[0]
            if handoff.typed_null_result is not None
            else "phase7a_measurement_not_authorized"
        )
        return MeasurementOccupancyResult(
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
    if surface_blocker_mask is None:
        return MeasurementOccupancyResult(
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

    blocker_source = np.asarray(surface_blocker_mask, dtype=bool)
    if blocker_source.shape != structural.shape:
        raise ValueError(
            f"{handoff.sample_id}: blocker shape {blocker_source.shape} "
            f"does not match structural shape {structural.shape}"
        )
    blocker_hash_before = hash_mask_after_uint8_cast(blocker_source)
    structural_hash_before = hash_mask_after_uint8_cast(structural)
    working = blocker_source.copy() & structural
    base_pixels = int(working.sum())
    clipped_pixels = int(blocker_source.sum() - base_pixels)
    records: tuple[dict[str, Any], ...] = ()
    evidence_source = "frozen_phase6_blocker"
    supplied_instances = list(passability_instances or [])

    if supplied_instances:
        completed, completed_records = complete_instance_occupancy(
            structural,
            working,
            [dict(instance) for instance in supplied_instances],
        )
        working = completed
        records = tuple(completed_records)
        evidence_source = "existing_runtime_instance_ownership"
    elif persisted_evidence is not None:
        persisted_mask = np.asarray(
            persisted_evidence.occupancy_mask,
            dtype=bool,
        )
        if persisted_mask.shape != structural.shape:
            raise ValueError(
                f"{handoff.sample_id}: persisted occupancy shape "
                f"{persisted_mask.shape} does not match structural shape "
                f"{structural.shape}"
            )
        if int(persisted_mask.sum()) != persisted_evidence.occupancy_pixels:
            raise AssertionError(
                f"{handoff.sample_id}: persisted occupancy pixel count changed"
            )
        working |= persisted_mask & structural
        records = persisted_evidence.instance_records
        evidence_source = "persisted_runtime_v2_occupancy"
    elif blocker_coverage_suspect:
        blocker_hash_after = hash_mask_after_uint8_cast(blocker_source)
        structural_hash_after = hash_mask_after_uint8_cast(structural)
        if (
            blocker_hash_after != blocker_hash_before
            or structural_hash_after != structural_hash_before
            or structural_hash_after != handoff.structural_mask_sha256
        ):
            raise AssertionError(
                f"{handoff.sample_id}: an upstream mask was modified"
            )
        return MeasurementOccupancyResult(
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
            evidence_source=(
                "explicit_blocker_coverage_suspect_without_instance_evidence"
            ),
            upstream_hashes_before={
                "surface_blocker_mask_sha256": blocker_hash_before,
                "boundary_structural_mask_sha256": structural_hash_before,
            },
            upstream_hashes_after={
                "surface_blocker_mask_sha256": blocker_hash_after,
                "boundary_structural_mask_sha256": structural_hash_after,
            },
        )

    working &= structural
    blocker_hash_after = hash_mask_after_uint8_cast(blocker_source)
    structural_hash_after = hash_mask_after_uint8_cast(structural)
    if (
        blocker_hash_after != blocker_hash_before
        or structural_hash_after != handoff.structural_mask_sha256
    ):
        raise AssertionError(f"{handoff.sample_id}: an upstream mask was modified")
    occupancy_pixels = int(working.sum())
    return MeasurementOccupancyResult(
        sample_id=handoff.sample_id,
        coverage_status=(
            "completed_existing_instance_evidence"
            if evidence_source != "frozen_phase6_blocker"
            else "verified_frozen_blocker"
        ),
        abstention_reasons=(),
        occupancy_mask=working.copy(),
        occupancy_mask_sha256=hash_mask_after_uint8_cast(working),
        occupancy_pixels=occupancy_pixels,
        base_blocker_pixels=base_pixels,
        added_completion_pixels=occupancy_pixels - base_pixels,
        clipped_blocker_pixels=clipped_pixels,
        instance_records=records,
        evidence_source=evidence_source,
        upstream_hashes_before={
            "surface_blocker_mask_sha256": blocker_hash_before,
            "boundary_structural_mask_sha256": structural_hash_before,
        },
        upstream_hashes_after={
            "surface_blocker_mask_sha256": blocker_hash_after,
            "boundary_structural_mask_sha256": structural_hash_after,
        },
    )
