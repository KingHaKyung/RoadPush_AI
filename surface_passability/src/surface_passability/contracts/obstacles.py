"""Typed obstacle-analysis contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _arrays_equivalent(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare canonical masks while accepting identical nonfinite arrays."""

    if first is second:
        return True
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    try:
        return bool(np.array_equal(first, second, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(first, second))


ObstacleFootprintKind = Literal[
    "pole_or_bollard",
    "cycle_like",
    "generic_obstacle",
]
ClearanceStrategy = Literal[
    "lower_support_or_pedestal",
    "wide_body_projected_to_ground",
    "generic_body_projected_to_ground",
]


@dataclass(frozen=True, slots=True)
class SemanticObstacleComponent:
    """One connected semantic obstacle component and its source geometry."""

    class_id: int
    component_index: int
    bbox: tuple[int, int, int, int]
    source_pixels: int
    mask: np.ndarray

    def to_public_record(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "component_index": self.component_index,
            "bbox": list(self.bbox),
            "source_pixels": self.source_pixels,
        }

    def to_legacy_record(self) -> dict[str, object]:
        record = self.to_public_record()
        record["_mask"] = self.mask
        return record


@dataclass(frozen=True, slots=True)
class ObstacleFootprint:
    """Ground-position clearance envelope for one semantic component."""

    mask: np.ndarray
    class_id: int
    source_pixels: int
    source_bbox: tuple[int, int, int, int]
    lower_band_top: int
    stable_width_px: float
    qualifying_rows: int
    selected_rows: int
    footprint_bbox: tuple[int, int, int, int]
    footprint_pixels: int
    kind: ObstacleFootprintKind
    clearance_strategy: ClearanceStrategy

    def to_legacy_metadata(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "source_pixels": self.source_pixels,
            "source_bbox": list(self.source_bbox),
            "lower_band_top": self.lower_band_top,
            "stable_width_px": self.stable_width_px,
            "qualifying_rows": self.qualifying_rows,
            "selected_rows": self.selected_rows,
            "footprint_bbox": list(self.footprint_bbox),
            "footprint_pixels": self.footprint_pixels,
            "kind": self.kind,
            "clearance_strategy": self.clearance_strategy,
        }

    def to_legacy_result(self) -> tuple[np.ndarray, dict[str, object]]:
        return self.mask, self.to_legacy_metadata()


@dataclass(frozen=True, slots=True)
class ProjectedGroundSupport:
    effective_mask: np.ndarray
    support_pixels: int
    support_fraction: float

    def to_legacy_result(self) -> tuple[np.ndarray, int, float]:
        return self.effective_mask, self.support_pixels, self.support_fraction


@dataclass(frozen=True, slots=True)
class SourceContactSupport:
    supported_columns: int
    total_columns: int
    support_ratio: float
    bottom_truncated: bool

    def to_legacy_result(self) -> tuple[int, int, float, bool]:
        return (
            self.supported_columns,
            self.total_columns,
            self.support_ratio,
            self.bottom_truncated,
        )


@dataclass(frozen=True, slots=True)
class OwnedObstacleFootprint:
    """One footprint candidate after ownership against an explicit surface."""

    selected_mask: np.ndarray
    source: ObstacleFootprint | None
    accepted: bool
    ground_support_pixels: int = 0
    ground_support_fraction: float = 0.0
    minimum_ground_support_pixels: int = 0
    source_contact_supported_columns: int = 0
    source_contact_total_columns: int = 0
    source_contact_support_fraction: float = 0.0
    source_bottom_truncated: bool = False
    rejection_reason: str | None = None

    def to_legacy_metadata(self) -> dict[str, object]:
        if self.source is None:
            return {
                "accepted": False,
                "rejection_reason": "no_footprint",
            }
        metadata = self.source.to_legacy_metadata()
        metadata.update({
            "accepted": self.accepted,
            "ground_support_pixels": self.ground_support_pixels,
            "ground_support_fraction": self.ground_support_fraction,
            "minimum_ground_support_pixels": self.minimum_ground_support_pixels,
            "source_contact_supported_columns": self.source_contact_supported_columns,
            "source_contact_total_columns": self.source_contact_total_columns,
            "source_contact_support_fraction": self.source_contact_support_fraction,
            "source_bottom_truncated": self.source_bottom_truncated,
            "rejection_reason": self.rejection_reason,
        })
        return metadata

    def to_legacy_result(self) -> tuple[np.ndarray, dict[str, object]]:
        return self.selected_mask, self.to_legacy_metadata()


@dataclass(frozen=True, slots=True)
class GroundedSilhouetteEvidence:
    candidate_present: bool
    accepted: bool
    ground_support_pixels: int
    ground_support_fraction: float
    source_bottom_truncated: bool = False

    def to_legacy_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "ground_support_pixels": self.ground_support_pixels,
            "ground_support_fraction": self.ground_support_fraction,
        }
        if self.candidate_present:
            metadata["source_bottom_truncated"] = self.source_bottom_truncated
        return metadata

    def to_legacy_result(self) -> tuple[bool, dict[str, object]]:
        return self.accepted, self.to_legacy_metadata()


@dataclass(frozen=True, slots=True)
class LowerAxisGeometry:
    angle_from_vertical_degrees: float
    center: tuple[float, float]
    vector: tuple[float, float]

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "lower_axis_angle_from_vertical_deg": self.angle_from_vertical_degrees,
            "axis_center": list(self.center),
            "axis_vector": list(self.vector),
        }


@dataclass(frozen=True, slots=True)
class HorizontalBracketEvidence:
    """Rows where an intrusion has visible source walkway on both sides."""

    rows: tuple[tuple[int, int, int], ...]
    restorable_mask: np.ndarray

    def to_legacy_result(self) -> tuple[list[tuple[int, int, int]], np.ndarray]:
        return list(self.rows), self.restorable_mask


@dataclass(frozen=True, slots=True)
class NaturalGroundFootprint:
    """Ground footprint inferred from one bracketed natural intrusion."""

    footprint_mask: np.ndarray
    source_mask: np.ndarray
    class_id: int
    class_name: str
    component_index: int
    source_pixels: int
    corridor_intrusion_pixels: int
    accepted: bool
    rejection_reason: str | None
    intrusion_bbox: tuple[int, int, int, int] | None = None
    bracketed_rows: int | None = None
    intrusion_rows: int | None = None
    bracket_fraction: float | None = None
    stable_lower_width_px: float | None = None
    silhouette_strategy: str | None = None
    overhead_vertical_run_threshold: int | None = None
    overhead_supported_columns: int | None = None
    source_silhouette_pixels: int | None = None
    footprint_bbox: tuple[int, int, int, int] | None = None
    footprint_pixels: int | None = None
    ground_ring_support_pixels: int | None = None
    ground_ring_support_fraction: float | None = None

    def to_legacy_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "source": "natural_intrusion",
            "class_id": self.class_id,
            "class_name": self.class_name,
            "component_index": self.component_index,
            "source_pixels": self.source_pixels,
            "corridor_intrusion_pixels": self.corridor_intrusion_pixels,
            "accepted": self.accepted,
        }
        if self.intrusion_bbox is None:
            metadata["rejection_reason"] = self.rejection_reason
            return metadata
        metadata.update({
            "intrusion_bbox": list(self.intrusion_bbox),
            "bracketed_rows": self.bracketed_rows,
            "intrusion_rows": self.intrusion_rows,
            "bracket_fraction": self.bracket_fraction,
        })
        if self.stable_lower_width_px is None:
            metadata["rejection_reason"] = self.rejection_reason
            return metadata
        metadata.update({
            "stable_lower_width_px": self.stable_lower_width_px,
            "silhouette_strategy": self.silhouette_strategy,
            "overhead_vertical_run_threshold": self.overhead_vertical_run_threshold,
            "overhead_supported_columns": self.overhead_supported_columns,
            "source_silhouette_pixels": self.source_silhouette_pixels,
            "footprint_bbox": list(self.footprint_bbox),
            "footprint_pixels": self.footprint_pixels,
            "ground_ring_support_pixels": self.ground_ring_support_pixels,
            "ground_ring_support_fraction": self.ground_ring_support_fraction,
        })
        metadata["rejection_reason"] = self.rejection_reason
        return metadata

    def to_legacy_result(self) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return self.footprint_mask, self.source_mask, self.to_legacy_metadata()


@dataclass(frozen=True, slots=True)
class NaturalHoleEvidence:
    """Aggregated natural-obstacle evidence recovered from corridor holes."""

    footprint_mask: np.ndarray
    source_occluder_mask: np.ndarray
    accepted_source_mask: np.ndarray
    rejected_source_mask: np.ndarray
    records: tuple[dict[str, object], ...]
    corridor_envelope_mask: np.ndarray

    def to_legacy_result(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, object]],
        np.ndarray,
    ]:
        return (
            self.footprint_mask,
            self.source_occluder_mask,
            self.accepted_source_mask,
            self.rejected_source_mask,
            list(self.records),
            self.corridor_envelope_mask,
        )


@dataclass(frozen=True, slots=True)
class VehicleDepthCut:
    """Strongest left/right bottom-profile depth discontinuity."""

    cut: int
    left_median_y: float
    right_median_y: float
    depth_difference_px: float
    left_mad_px: float
    right_mad_px: float
    score: float

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "cut": self.cut,
            "left_median_y": self.left_median_y,
            "right_median_y": self.right_median_y,
            "depth_difference_px": self.depth_difference_px,
            "left_mad_px": self.left_mad_px,
            "right_mad_px": self.right_mad_px,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class VehicleComponentAnalysis:
    """Geometry-only keep/split decision for one semantic vehicle component."""

    children: tuple[np.ndarray, ...]
    source_bbox: list[int]
    source_pixels: int
    touches_bottom: bool
    bottom_gap_px: int
    bottom_tolerance_px: int
    decision: str
    reason: str
    profile: np.ndarray
    depth_cut: VehicleDepthCut | None = None
    split_x: int | None = None
    required_depth_difference_px: int | float | None = None
    child_area_fractions: tuple[float, float] | None = None

    def to_legacy_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "source_bbox": self.source_bbox,
            "source_pixels": self.source_pixels,
            "touches_bottom": self.touches_bottom,
            "bottom_gap_px": self.bottom_gap_px,
            "bottom_tolerance_px": self.bottom_tolerance_px,
            "decision": self.decision,
            "reason": self.reason,
            "profile": self.profile.tolist(),
        }
        if self.depth_cut is not None:
            record.update(self.depth_cut.to_legacy_record())
            record["split_x"] = self.split_x
            record["required_depth_difference_px"] = (
                self.required_depth_difference_px
            )
            record["child_area_fractions"] = list(self.child_area_fractions)
        return record

    def to_legacy_result(self) -> tuple[list[np.ndarray], dict[str, object]]:
        return list(self.children), self.to_legacy_record()

    def to_trigger_legacy_record(self) -> dict[str, object]:
        record = self.to_legacy_record()
        record.pop("profile", None)
        return record


@dataclass(frozen=True, slots=True)
class VehicleTriggerComponent:
    """One semantic car component and its conditional-inference decision."""

    component_id: int
    source_bbox: tuple[int, int, int, int]
    source_pixels: int
    walkway_owned: bool
    ground_support_pixels: int
    ground_support_fraction: float
    minimum_ground_support_pixels: int
    source_contact_supported_columns: int
    source_contact_total_columns: int
    source_contact_support_fraction: float
    source_bottom_truncated: bool
    geometry_anomaly: bool
    trigger_yolo: bool
    decision: str
    geometry: VehicleComponentAnalysis
    component_mask: np.ndarray
    candidate_mask: np.ndarray
    effective_mask: np.ndarray

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "source_bbox": list(self.source_bbox),
            "source_pixels": self.source_pixels,
            "walkway_owned": self.walkway_owned,
            "ground_support_pixels": self.ground_support_pixels,
            "ground_support_fraction": self.ground_support_fraction,
            "minimum_ground_support_pixels": self.minimum_ground_support_pixels,
            "source_contact_supported_columns": self.source_contact_supported_columns,
            "source_contact_total_columns": self.source_contact_total_columns,
            "source_contact_support_fraction": self.source_contact_support_fraction,
            "source_bottom_truncated": self.source_bottom_truncated,
            "geometry_anomaly": self.geometry_anomaly,
            "trigger_yolo": self.trigger_yolo,
            "decision": self.decision,
            "geometry": self.geometry.to_trigger_legacy_record(),
            "_component": self.component_mask,
            "_candidate": self.candidate_mask,
            "_effective": self.effective_mask,
        }


@dataclass(frozen=True, slots=True)
class VehicleTriggerEvidence:
    """All eligible semantic car components and their YOLO trigger decisions."""

    components: tuple[VehicleTriggerComponent, ...]

    @property
    def requires_yolo(self) -> bool:
        return any(component.trigger_yolo for component in self.components)

    def to_legacy_result(self) -> list[dict[str, object]]:
        return [component.to_legacy_record() for component in self.components]


@dataclass(frozen=True, slots=True)
class VehicleInstance:
    """One vehicle instance emitted by the conditional segmentation model."""

    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    mask_pixels: int
    mask: np.ndarray

    def to_public_record(self) -> dict[str, object]:
        return {
            "detection_index": self.detection_index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
            "mask_pixels": self.mask_pixels,
        }

    def to_legacy_record(self) -> dict[str, object]:
        record = self.to_public_record()
        record["_mask"] = self.mask
        return record


@dataclass(frozen=True, slots=True)
class VehicleInstanceMatch:
    """Overlap evidence between one vehicle instance and a trigger component."""

    instance: VehicleInstance
    mapillary_intersection_pixels: int
    yolo_mask_overlap_fraction: float
    mapillary_component_overlap_fraction: float
    matched: bool

    def to_legacy_record(self) -> dict[str, object]:
        record = self.instance.to_public_record()
        record.update({
            "mapillary_intersection_pixels": self.mapillary_intersection_pixels,
            "yolo_mask_overlap_fraction": self.yolo_mask_overlap_fraction,
            "mapillary_component_overlap_fraction": (
                self.mapillary_component_overlap_fraction
            ),
            "matched_to_trigger_component": self.matched,
        })
        return record


@dataclass(frozen=True, slots=True)
class VehicleInstanceMatchEvidence:
    """Accepted instances plus the complete per-detection overlap audit."""

    accepted: tuple[VehicleInstance, ...]
    audit: tuple[VehicleInstanceMatch, ...]

    def to_legacy_result(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return (
            [instance.to_legacy_record() for instance in self.accepted],
            [match.to_legacy_record() for match in self.audit],
        )


@dataclass(frozen=True, slots=True)
class VehicleInstancePartition:
    """Mapillary parent pixels assigned to accepted vehicle instances."""

    children: tuple[np.ndarray, ...]

    def to_legacy_result(self) -> list[np.ndarray]:
        return list(self.children)


@dataclass(frozen=True, slots=True)
class VehicleInferenceDiagnostics:
    box_count: int
    mask_count: int

    def to_legacy_record(self) -> dict[str, int]:
        return {
            "box_count": self.box_count,
            "mask_count": self.mask_count,
        }


@dataclass(frozen=True, slots=True)
class VehicleInferenceResult:
    """Filtered vehicle instances and raw model-output counts."""

    instances: tuple[VehicleInstance, ...]
    diagnostics: VehicleInferenceDiagnostics

    def to_legacy_result(
        self,
    ) -> tuple[list[dict[str, object]], dict[str, int]]:
        return (
            [instance.to_legacy_record() for instance in self.instances],
            self.diagnostics.to_legacy_record(),
        )


@dataclass(frozen=True, slots=True)
class ObstacleObjectGeometry:
    """Surface-independent geometry attached to one semantic object."""

    component: SemanticObstacleComponent
    source_record_items: tuple[tuple[str, object], ...]
    lower_silhouette_candidate: np.ndarray
    axis_geometry: LowerAxisGeometry
    footprint_evaluated: bool
    footprint_candidate_mask: np.ndarray | None = None
    footprint: ObstacleFootprint | None = None

    def __post_init__(self) -> None:
        """Reject split-brain identity between typed and preserved records."""

        record = dict(self.source_record_items)
        expected_scalars = {
            "class_id": self.component.class_id,
            "component_index": self.component.component_index,
            "source_pixels": self.component.source_pixels,
        }
        for key, expected in expected_scalars.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"source record {key} disagrees with obstacle component"
                )
        try:
            source_bbox = tuple(record["bbox"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                "source record bbox disagrees with obstacle component"
            ) from error
        if source_bbox != self.component.bbox:
            raise ValueError(
                "source record bbox disagrees with obstacle component"
            )
        source_mask = record.get("_mask")
        masks_equal = (
            isinstance(source_mask, np.ndarray)
            and _arrays_equivalent(source_mask, self.component.mask)
        )
        if not masks_equal:
            raise ValueError(
                "source record _mask disagrees with obstacle component"
            )
        if not self.footprint_evaluated:
            if self.footprint_candidate_mask is not None or self.footprint is not None:
                raise ValueError(
                    "unevaluated obstacle cannot carry footprint evidence"
                )
        elif self.footprint is not None:
            if not (
                isinstance(self.footprint_candidate_mask, np.ndarray)
                and _arrays_equivalent(
                    self.footprint_candidate_mask,
                    self.footprint.mask,
                )
            ):
                raise ValueError(
                    "footprint candidate disagrees with typed footprint mask"
                )

    def to_legacy_record(self) -> dict[str, object]:
        record = dict(self.source_record_items)
        record["_lower_silhouette_candidate"] = self.lower_silhouette_candidate
        record["_axis_geometry"] = self.axis_geometry.to_legacy_record()
        if self.footprint_evaluated:
            record["_footprint_candidate"] = self.footprint_candidate_mask
            record["_footprint_metadata"] = (
                None
                if self.footprint is None
                else self.footprint.to_legacy_metadata()
            )
        return record

    def to_clean_legacy_record(self) -> dict[str, object]:
        """Project non-private source metadata in its frozen insertion order."""

        return {
            key: value
            for key, value in self.source_record_items
            if not key.startswith("_")
        }


@dataclass(frozen=True, slots=True)
class ObstacleGeometry:
    """Image-level obstacle identities and source geometry before ownership."""

    image: np.ndarray
    labels: np.ndarray
    all_objects: tuple[SemanticObstacleComponent, ...]
    active_objects: tuple[ObstacleObjectGeometry, ...]
    banners: tuple[SemanticObstacleComponent, ...]
    transient_source: np.ndarray
    natural_source_evidence: np.ndarray
    hazard_ids: frozenset[int]
    id2label: dict[int, str]
    yolo_detections: tuple[VehicleInstance, ...] = ()
    yolo_inference: VehicleInferenceDiagnostics | None = None
    surface_trigger_records: dict[str, VehicleTriggerEvidence] | None = None

    def to_legacy_fields(self) -> dict[str, object]:
        return {
            "image": self.image,
            "labels": self.labels,
            "all_objects": [obj.to_legacy_record() for obj in self.all_objects],
            "active_objects": [
                obj.to_legacy_record() for obj in self.active_objects
            ],
            "banners": [obj.to_legacy_record() for obj in self.banners],
            "transient_source": self.transient_source,
            "natural_source_evidence": self.natural_source_evidence,
            "hazard_ids": set(self.hazard_ids),
            "id2label": dict(self.id2label),
            "yolo_detections": [
                instance.to_legacy_record()
                for instance in self.yolo_detections
            ],
            "yolo_inference": (
                None
                if self.yolo_inference is None
                else self.yolo_inference.to_legacy_record()
            ),
            "surface_trigger_records": {
                name: evidence.to_legacy_result()
                for name, evidence in (self.surface_trigger_records or {}).items()
            },
        }


@dataclass(frozen=True, slots=True)
class SurfaceVehicleComponentEvidence:
    """Instance-match and split decision for one triggered vehicle parent."""

    source_bbox: tuple[int, int, int, int]
    matches: VehicleInstanceMatchEvidence
    split_confirmed: bool

    def to_legacy_metadata(self) -> dict[str, object]:
        _, detection_audit = self.matches.to_legacy_result()
        return {
            "source_bbox": list(self.source_bbox),
            "matched_instances": len(self.matches.accepted),
            "split_confirmed": self.split_confirmed,
            "detections": detection_audit,
        }


@dataclass(frozen=True, slots=True)
class SurfaceVehicleProjection:
    """One surface's non-destructive view of shared vehicle evidence."""

    active_objects: tuple[ObstacleObjectGeometry, ...]
    triggered: bool
    trigger_component_count: int
    inference_run: bool
    result: str
    split_confirmed_count: int
    fallback_parent_count: int
    components: tuple[SurfaceVehicleComponentEvidence, ...]
    inference: VehicleInferenceDiagnostics | None = None

    def to_legacy_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "triggered": self.triggered,
            "trigger_component_count": self.trigger_component_count,
            "inference_run": self.inference_run,
            "result": self.result,
            "split_confirmed_count": self.split_confirmed_count,
            "fallback_parent_count": self.fallback_parent_count,
            "components": [
                component.to_legacy_metadata()
                for component in self.components
            ],
            "scope": "existing_v2_car_trigger_only",
        }
        if self.inference_run:
            metadata["inference"] = (
                None
                if self.inference is None
                else self.inference.to_legacy_record()
            )
        return metadata

    def to_legacy_result(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [obj.to_legacy_record() for obj in self.active_objects],
            self.to_legacy_metadata(),
        )


@dataclass(frozen=True, slots=True)
class PassabilityObstacleInstance:
    """One accepted obstacle mask consumed by passability measurement."""

    class_id: int
    class_name: str
    kind: str
    source_mask: np.ndarray
    selected_mask: np.ndarray

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "kind": self.kind,
            "source_mask": self.source_mask,
            "selected_mask": self.selected_mask,
        }


@dataclass(frozen=True, slots=True)
class StandaloneBannerEvidence:
    """Banner rejected because no grounded nearby support was available."""

    banner: SemanticObstacleComponent
    class_name: str

    def to_legacy_record(self) -> dict[str, object]:
        return {
            **self.banner.to_public_record(),
            "class_name": self.class_name,
            "kind": "standalone_banner",
            "strategy": "context_only",
            "accepted": False,
            "rejection_reason": "no_grounded_support_connection",
        }


@dataclass(frozen=True, slots=True)
class BannerSupportMember:
    """One semantic support grouped with a banner."""

    component: SemanticObstacleComponent
    class_name: str

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "class_id": int(self.component.class_id),
            "class_name": self.class_name,
            "component_index": int(self.component.component_index),
            "bbox": list(self.component.bbox),
        }


@dataclass(frozen=True, slots=True)
class BannerSupportClusterEvidence:
    """Ground-support decision for a banner and selected support objects."""

    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    group_mode: str
    grouped_supports: tuple[BannerSupportMember, ...]
    source_mask: np.ndarray
    selected_mask: np.ndarray
    grounded: GroundedSilhouetteEvidence
    silhouette_fraction: float

    def to_legacy_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "kind": "grounded_banner_support_cluster",
            "bbox": list(self.bbox),
            "strategy": "lower_19_percent_silhouette",
            "banner_support_group_mode": self.group_mode,
            "grouped_supports": [
                support.to_legacy_record()
                for support in self.grouped_supports
            ],
            "silhouette_fraction": self.silhouette_fraction,
            "source_pixels": int(self.source_mask.sum()),
            "selected_pixels": (
                int(self.selected_mask.sum())
                if self.grounded.accepted
                else 0
            ),
            "accepted": bool(self.grounded.accepted),
            "rejection_reason": (
                None
                if self.grounded.accepted
                else "no_walkway_ground_support"
            ),
        }
        record.update(self.grounded.to_legacy_metadata())
        return record


BannerOwnershipEvidence = StandaloneBannerEvidence | BannerSupportClusterEvidence


@dataclass(frozen=True, slots=True)
class BannerSupportOwnership:
    """All banner/support effects contributed to one surface ownership pass."""

    recognized_mask: np.ndarray
    selected_mask: np.ndarray
    silhouette_selected_mask: np.ndarray
    grouped_support_ids: frozenset[tuple[int, int]]
    records: tuple[BannerOwnershipEvidence, ...]
    passability_instances: tuple[PassabilityObstacleInstance, ...]

    def to_legacy_records(self) -> list[dict[str, object]]:
        return [record.to_legacy_record() for record in self.records]

    def to_legacy_instances(self) -> list[dict[str, object]]:
        return [
            instance.to_legacy_record()
            for instance in self.passability_instances
        ]


@dataclass(frozen=True, slots=True)
class WheeledObstacleEvidence:
    """Ownership and selector result for one wheeled semantic object."""

    source: ObstacleObjectGeometry
    footprint_ownership: OwnedObstacleFootprint
    class_name: str
    strategy: str
    selector_state: str
    silhouette_angle_max_degrees: float | None
    selected_mask: np.ndarray

    def to_legacy_record(self) -> dict[str, object]:
        record = self.source.to_clean_legacy_record()
        record.update(self.footprint_ownership.to_legacy_metadata())
        record.update(self.source.axis_geometry.to_legacy_record())
        record.update({
            "class_name": self.class_name,
            "kind": "wheeled_unmanned_obstacle",
            "strategy": self.strategy,
            "wheeled_selector_state": self.selector_state,
            "wheeled_silhouette_angle_max_deg": (
                self.silhouette_angle_max_degrees
            ),
            "scoped_closure_pixels": 0,
            "selected_pixels": int(self.selected_mask.sum()),
        })
        return record

    def to_passability_instance(self) -> PassabilityObstacleInstance | None:
        if not self.selected_mask.any():
            return None
        return PassabilityObstacleInstance(
            class_id=int(self.source.component.class_id),
            class_name=self.class_name,
            kind="wheeled_unmanned_obstacle",
            source_mask=self.source.component.mask.copy(),
            selected_mask=self.selected_mask.copy(),
        )


@dataclass(frozen=True, slots=True)
class WheeledObstacleOwnership:
    """All wheeled-object contributions for one surface ownership pass."""

    recognized_mask: np.ndarray
    selected_mask: np.ndarray
    footprint_selected_mask: np.ndarray
    silhouette_selected_mask: np.ndarray
    wheeled_silhouette_selected_mask: np.ndarray
    baseline_for_natural_mask: np.ndarray
    records: tuple[WheeledObstacleEvidence, ...]
    passability_instances: tuple[PassabilityObstacleInstance, ...]

    def to_legacy_records(self) -> list[dict[str, object]]:
        return [record.to_legacy_record() for record in self.records]

    def to_legacy_instances(self) -> list[dict[str, object]]:
        return [
            instance.to_legacy_record()
            for instance in self.passability_instances
        ]


@dataclass(frozen=True, slots=True)
class GroundedObstacleOwnershipEvidence:
    """Lower-silhouette ownership decision for one non-wheeled object."""

    source: ObstacleObjectGeometry
    class_name: str
    grounding: GroundedSilhouetteEvidence
    silhouette_fraction: float

    def to_legacy_record(self) -> dict[str, object]:
        record = self.source.to_clean_legacy_record()
        record.update(self.grounding.to_legacy_metadata())
        record.update({
            "class_name": self.class_name,
            "kind": "other_grounded_obstacle",
            "strategy": "lower_19_percent_silhouette",
            "silhouette_fraction": self.silhouette_fraction,
            "accepted": bool(self.grounding.accepted),
            "selected_pixels": (
                int(self.source.lower_silhouette_candidate.sum())
                if self.grounding.accepted
                else 0
            ),
            "rejection_reason": (
                None
                if self.grounding.accepted
                else "no_walkway_ground_support"
            ),
        })
        return record

    def to_passability_instance(self) -> PassabilityObstacleInstance | None:
        if not self.grounding.accepted:
            return None
        return PassabilityObstacleInstance(
            class_id=int(self.source.component.class_id),
            class_name=self.class_name,
            kind="other_grounded_obstacle",
            source_mask=self.source.component.mask.copy(),
            selected_mask=self.source.lower_silhouette_candidate.copy(),
        )


@dataclass(frozen=True, slots=True)
class GroundedObstacleOwnership:
    """All non-wheeled grounded-obstacle contributions for one surface."""

    recognized_mask: np.ndarray
    selected_mask: np.ndarray
    silhouette_selected_mask: np.ndarray
    baseline_for_natural_mask: np.ndarray
    records: tuple[GroundedObstacleOwnershipEvidence, ...]
    passability_instances: tuple[PassabilityObstacleInstance, ...]

    def to_legacy_records(self) -> list[dict[str, object]]:
        return [record.to_legacy_record() for record in self.records]

    def to_legacy_instances(self) -> list[dict[str, object]]:
        return [
            instance.to_legacy_record()
            for instance in self.passability_instances
        ]


@dataclass(frozen=True, slots=True)
class NaturalHoleOwnershipRecord:
    """Natural-hole evidence projected into the ownership record contract."""

    source_record_items: tuple[tuple[str, object], ...]
    class_name: str
    selected_pixels: int

    def to_legacy_record(self) -> dict[str, object]:
        record = dict(self.source_record_items)
        record.update({
            "kind": "v2_1_natural_grounded_footprint",
            "strategy": "footprint",
            "class_name": self.class_name,
            "selected_pixels": self.selected_pixels,
            "natural_restoration_applied": False,
        })
        return record


@dataclass(frozen=True, slots=True)
class NaturalHoleOwnership:
    """Natural-hole mask and record contributions for one surface."""

    evidence: NaturalHoleEvidence
    records: tuple[NaturalHoleOwnershipRecord, ...]

    @property
    def recognized_mask(self) -> np.ndarray:
        return self.evidence.source_occluder_mask

    @property
    def selected_mask(self) -> np.ndarray:
        return self.evidence.footprint_mask

    @property
    def footprint_selected_mask(self) -> np.ndarray:
        return self.evidence.footprint_mask

    def to_legacy_records(self) -> list[dict[str, object]]:
        return [record.to_legacy_record() for record in self.records]


@dataclass(frozen=True, slots=True)
class TransientOccluderEvidence:
    """One semantic transient retained for recognition but not occupancy."""

    component: SemanticObstacleComponent
    class_name: str

    def to_legacy_record(self) -> dict[str, object]:
        return {
            **self.component.to_public_record(),
            "class_name": self.class_name,
            "kind": "transient_occluder",
            "strategy": "transient_rows_only",
            "accepted": False,
            "selected_pixels": 0,
        }


@dataclass(frozen=True, slots=True)
class TransientOccluderOwnership:
    """Transient recognized-mask and diagnostic-record contributions."""

    recognized_mask: np.ndarray
    records: tuple[TransientOccluderEvidence, ...]

    def to_legacy_records(self) -> list[dict[str, object]]:
        return [record.to_legacy_record() for record in self.records]


@dataclass(frozen=True, slots=True)
class SemanticHazardEvidence:
    """One full-mask semantic hazard accepted into passability occupancy."""

    component: SemanticObstacleComponent
    class_name: str

    def to_legacy_record(self) -> dict[str, object]:
        return {
            **self.component.to_public_record(),
            "class_name": self.class_name,
            "kind": "semantic_hazard",
            "strategy": "full_semantic_hazard",
            "accepted": True,
            "selected_pixels": int(self.component.mask.sum()),
        }

    def to_passability_instance(self) -> PassabilityObstacleInstance:
        return PassabilityObstacleInstance(
            class_id=int(self.component.class_id),
            class_name=self.class_name,
            kind="semantic_hazard",
            source_mask=self.component.mask.copy(),
            selected_mask=self.component.mask.copy(),
        )


@dataclass(frozen=True, slots=True)
class SemanticHazardOwnership:
    """Semantic-hazard contributions for one ownership result."""

    recognized_mask: np.ndarray
    selected_mask: np.ndarray
    records: tuple[SemanticHazardEvidence, ...]
    passability_instances: tuple[PassabilityObstacleInstance, ...]

    def to_legacy_records(self) -> list[dict[str, object]]:
        return [record.to_legacy_record() for record in self.records]

    def to_legacy_instances(self) -> list[dict[str, object]]:
        return [
            instance.to_legacy_record()
            for instance in self.passability_instances
        ]


ObstacleOwnershipRecord = (
    StandaloneBannerEvidence
    | BannerSupportClusterEvidence
    | WheeledObstacleEvidence
    | GroundedObstacleOwnershipEvidence
    | NaturalHoleOwnershipRecord
    | TransientOccluderEvidence
    | SemanticHazardEvidence
)


@dataclass(frozen=True, slots=True)
class ObstacleOwnershipPolicy:
    """Frozen policy settings reported with one ownership result."""

    wheeled_silhouette_angle_max_degrees: float | None
    banner_support_group_mode: str

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "ground_animal_transient": True,
            "wheeled_default": "footprint",
            "wheeled_exception_selector": (
                "lower_axis_angle_probe"
                if self.wheeled_silhouette_angle_max_degrees is not None
                else "deferred_not_integrated"
            ),
            "wheeled_silhouette_angle_max_deg": (
                self.wheeled_silhouette_angle_max_degrees
            ),
            "non_wheeled": "lower_19_percent_silhouette_proposal",
            "banner": "grounded_support_cluster_lower_19_percent_silhouette",
            "banner_support_group_mode": self.banner_support_group_mode,
            "global_cycle_closure": False,
            "natural_grounded_footprint": True,
            "natural_restoration": False,
            "cpu_only": True,
        }


@dataclass(frozen=True, slots=True)
class ObstacleOwnershipResult:
    """Complete surface-specific obstacle ownership and occupancy result."""

    recognized_mask: np.ndarray
    transient_mask: np.ndarray
    hazard_mask: np.ndarray
    footprint_mask: np.ndarray
    silhouette_mask: np.ndarray
    wheeled_silhouette_mask: np.ndarray
    final_mask: np.ndarray
    records: tuple[ObstacleOwnershipRecord, ...]
    passability_instances: tuple[PassabilityObstacleInstance, ...]
    vehicle_projection: SurfaceVehicleProjection
    violations: tuple[str, ...]
    strategy_overlap_pixels: int
    policy: ObstacleOwnershipPolicy

    def to_legacy_result(self) -> dict[str, object]:
        return {
            "recognized_mask": self.recognized_mask,
            "transient_mask": self.transient_mask,
            "hazard_mask": self.hazard_mask,
            "footprint_mask": self.footprint_mask,
            "silhouette_mask": self.silhouette_mask,
            "wheeled_silhouette_mask": self.wheeled_silhouette_mask,
            "final_mask": self.final_mask,
            "objects": [record.to_legacy_record() for record in self.records],
            "passability_instances": [
                instance.to_legacy_record()
                for instance in self.passability_instances
            ],
            "yolo": self.vehicle_projection.to_legacy_metadata(),
            "violations": list(self.violations),
            "strategy_overlap_pixels": self.strategy_overlap_pixels,
            "policy": self.policy.to_legacy_record(),
        }


@dataclass(frozen=True, slots=True)
class ObstacleEvidence:
    """Once-per-image geometry and the two surface ownership views."""

    geometry: ObstacleGeometry
    sidewalk: ObstacleOwnershipResult
    road: ObstacleOwnershipResult

    def to_legacy_results(self) -> dict[str, dict[str, object]]:
        """Return the current downstream blocker mapping for parity adapters."""

        return {
            "sidewalk": self.sidewalk.to_legacy_result(),
            "road": self.road.to_legacy_result(),
        }


@dataclass(frozen=True, slots=True)
class ObstacleSourceEvidence:
    """One usable semantic source supplied to surface-component selection."""

    record_index: int
    mask: np.ndarray
    area: int
    reported_bbox: object
    class_name: str | None
    kind: str | None
    strategy: str | None

    def to_legacy_record(self) -> dict[str, object]:
        """Return the exact current component-selector mapping shape."""

        return {
            "index": self.record_index,
            "mask": self.mask,
            "area": self.area,
            "bbox": self.reported_bbox,
            "class_name": self.class_name,
            "kind": self.kind,
            "strategy": self.strategy,
        }
