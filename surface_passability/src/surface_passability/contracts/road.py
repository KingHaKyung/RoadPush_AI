"""Typed contracts for Road surface construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surface_passability.contracts.surfaces import SurfaceContext


@dataclass(frozen=True, slots=True)
class RoadRepairComponentEvidence:
    """One semantic component considered for local Road inheritance."""

    source_class_id: int
    repair_class_id: int
    component_index: int
    pixels: int
    accepted_pixels: int
    ring_pixels: int
    source_support_pixels: int
    source_support_fraction: float
    component_fraction: float
    touches_frame: bool
    geometry_ok: bool
    forbidden_overlap_pixels: int
    accepted: bool

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen Road-repair diagnostic record."""

        return {
            "source_class_id": self.source_class_id,
            "repair_class_id": self.repair_class_id,
            "component_index": self.component_index,
            "pixels": self.pixels,
            "accepted_pixels": self.accepted_pixels,
            "ring_pixels": self.ring_pixels,
            "source_support_pixels": self.source_support_pixels,
            "source_support_fraction": self.source_support_fraction,
            "component_fraction": self.component_fraction,
            "touches_frame": self.touches_frame,
            "geometry_ok": self.geometry_ok,
            "forbidden_overlap_pixels": self.forbidden_overlap_pixels,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class RoadContextPolicy:
    """Serializable policy identity for Road context construction."""

    source_class_ids: tuple[int, ...]
    repair_class_ids: tuple[int, ...]
    repair_radius_fraction: float
    minimum_ring_fraction: float
    minimum_support_pixels: int
    union_seam_radius_px: int
    forbidden_class_ids: tuple[int, ...]
    trusted_sidewalk_uses_all_repaired_components: bool
    sidewalk_outer_row_span_used_as_guard: bool
    bike_lane_included: bool

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen Road context policy record."""

        return {
            "source_class_ids": list(self.source_class_ids),
            "repair_class_ids": list(self.repair_class_ids),
            "repair_radius_fraction": self.repair_radius_fraction,
            "minimum_ring_fraction": self.minimum_ring_fraction,
            "minimum_support_pixels": self.minimum_support_pixels,
            "union_seam_radius_px": self.union_seam_radius_px,
            "forbidden_class_ids": list(self.forbidden_class_ids),
            "trusted_sidewalk_uses_all_repaired_components": (
                self.trusted_sidewalk_uses_all_repaired_components
            ),
            "sidewalk_outer_row_span_used_as_guard": (
                self.sidewalk_outer_row_span_used_as_guard
            ),
            "bike_lane_included": self.bike_lane_included,
        }


@dataclass(frozen=True, slots=True)
class RoadContext:
    """Complete pre-obstacle Road surface context and construction evidence."""

    surface: SurfaceContext
    raw_road_mask: np.ndarray
    raw_service_lane_mask: np.ndarray
    raw_parking_mask: np.ndarray
    repaired_road_mask: np.ndarray
    repaired_service_lane_mask: np.ndarray
    repaired_parking_mask: np.ndarray
    accepted_repair_mask: np.ndarray
    trusted_sidewalk_guard_mask: np.ndarray
    road_forbidden_mask: np.ndarray
    surface_exclusion_mask: np.ndarray
    bike_lane_mask: np.ndarray
    repair_components: tuple[RoadRepairComponentEvidence, ...]
    policy: RoadContextPolicy

    def repair_records(self) -> list[dict[str, object]]:
        """Return ordered frozen diagnostic records for all source classes."""

        return [component.to_legacy_record() for component in self.repair_components]


@dataclass(frozen=True, slots=True)
class RoadOwnershipPolicy:
    """Policy provenance inherited from positive Road ownership."""

    positive_ownership_policy: str
    positive_ownership_before_repair: bool
    raw_road_ownership_enabled: bool
    service_lane_ownership_policy: str
    parking_ownership_policy: str
    obstacle_or_footprint_used_for_ownership: bool

    def augment_legacy_record(
        self,
        base_policy: RoadContextPolicy,
    ) -> dict[str, object]:
        """Return the frozen owned-Road policy record and key order."""

        record = base_policy.to_legacy_record()
        record.update(
            {
                "positive_ownership_policy": self.positive_ownership_policy,
                "positive_ownership_before_repair": (
                    self.positive_ownership_before_repair
                ),
                "raw_road_ownership_enabled": self.raw_road_ownership_enabled,
                "service_lane_ownership_policy": (
                    self.service_lane_ownership_policy
                ),
                "parking_ownership_policy": self.parking_ownership_policy,
                "obstacle_or_footprint_used_for_ownership": (
                    self.obstacle_or_footprint_used_for_ownership
                ),
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class OwnedRoadContextResult:
    """Road context rebuilt under positive ownership with its provenance."""

    road: RoadContext
    policy: RoadOwnershipPolicy

    def road_policy_record(self) -> dict[str, object]:
        """Return the exact frozen owned-Road policy record and key order."""

        return self.policy.augment_legacy_record(self.road.policy)


@dataclass(frozen=True, slots=True)
class RoadDrainRestorationPolicy:
    """Policy provenance for replaying an approved drain into owned Road."""

    policy_id: str
    linear_transverse_drain_enabled: bool
    linear_drain_whole_object_restoration: bool
    road_positive_ownership_enabled: bool
    ordered_stages: tuple[str, ...]

    def augment_legacy_record(
        self,
        inherited_policy_record: dict[str, object],
    ) -> dict[str, object]:
        """Return the frozen enhanced-Road policy record and key order."""

        record = dict(inherited_policy_record)
        record.update(
            {
                "enhanced_road_policy_id": self.policy_id,
                "linear_transverse_drain_enabled": (
                    self.linear_transverse_drain_enabled
                ),
                "linear_drain_whole_object_restoration": (
                    self.linear_drain_whole_object_restoration
                ),
                "road_positive_ownership_enabled": (
                    self.road_positive_ownership_enabled
                ),
                "enhanced_policy_order": list(self.ordered_stages),
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class RoadDrainRestorationResult:
    """Typed Road context and policy provenance after drain restoration."""

    road: RoadContext
    ownership_policy: RoadOwnershipPolicy
    policy: RoadDrainRestorationPolicy

    def road_policy_record(self) -> dict[str, object]:
        """Return the exact frozen Road policy record."""

        inherited = self.ownership_policy.augment_legacy_record(
            self.road.policy
        )
        return self.policy.augment_legacy_record(inherited)


@dataclass(frozen=True, slots=True)
class RoadOwnershipRouteQualityPolicy:
    """Thresholds for accepting a route as stable ownership evidence."""

    minimum_vertical_progress_fraction: float
    maximum_absolute_angle_degrees: float
    maximum_line_residual_fraction: float


@dataclass(frozen=True, slots=True)
class RouteQualityEvidence:
    """Typed route-quality decision used by Road ownership."""

    accepted: bool
    reason: str | None
    vertical_progress: float
    angle_degrees: float | None
    line_residual_px: float | None

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen route-quality record and key order."""

        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "vertical_progress": self.vertical_progress,
            "angle_deg": self.angle_degrees,
            "line_residual_px": self.line_residual_px,
        }


@dataclass(frozen=True, slots=True)
class RoadOwnershipTrackPolicy:
    """Horizontal separation policy for route-centered boundary tracks."""

    route_margin_fraction: float


@dataclass(frozen=True, slots=True)
class RawBoundaryTracks:
    """Per-row raw left/right boundary positions and curb evidence."""

    left_x: np.ndarray
    right_x: np.ndarray
    left_curb_hit: np.ndarray
    right_curb_hit: np.ndarray

    def to_legacy_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the frozen four-array result order."""

        return (
            self.left_x,
            self.right_x,
            self.left_curb_hit,
            self.right_curb_hit,
        )


@dataclass(frozen=True, slots=True)
class RoadOwnershipPersistencePolicy:
    """Thresholds for accepting and bridging a boundary track."""

    local_window_fraction: float
    minimum_local_support_fraction: float
    maximum_gap_fraction: float
    maximum_step_fraction: float
    frame_margin_fraction: float


@dataclass(frozen=True, slots=True)
class PersistentBoundaryTrack:
    """In-place stabilized boundary positions and row-level evidence."""

    raw_x: np.ndarray
    persistent_mask: np.ndarray
    interpolated_mask: np.ndarray
    reasons: tuple[str | None, ...]

    def to_legacy_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str | None]]:
        """Return the frozen result order with a fresh reasons list."""

        return (
            self.raw_x,
            self.persistent_mask,
            self.interpolated_mask,
            list(self.reasons),
        )


@dataclass(frozen=True, slots=True)
class SurfaceOwnershipProfile:
    """Complete policy for route-centered surface ownership."""

    surface_name: str
    policy_name: str
    enabled: bool
    owned_class_ids: tuple[int, ...]
    preserve_class_ids: tuple[int, ...]
    curb_class_ids: tuple[int, ...]
    sidewalk_class_ids: tuple[int, ...]
    local_window_fraction: float = 0.030
    minimum_local_support_fraction: float = 0.45
    maximum_gap_fraction: float = 0.035
    maximum_total_interpolated_rows_fraction: float = 0.075
    maximum_step_fraction: float = 0.045
    minimum_route_progress_fraction: float = 0.20
    maximum_route_angle_deg: float = 60.0
    maximum_line_residual_fraction: float = 0.090
    frame_margin_fraction: float = 0.012
    route_margin_fraction: float = 0.012

    def route_quality_policy(self) -> RoadOwnershipRouteQualityPolicy:
        """Project the route-quality thresholds used by that responsibility."""

        return RoadOwnershipRouteQualityPolicy(
            minimum_vertical_progress_fraction=(
                self.minimum_route_progress_fraction
            ),
            maximum_absolute_angle_degrees=self.maximum_route_angle_deg,
            maximum_line_residual_fraction=(
                self.maximum_line_residual_fraction
            ),
        )

    def track_policy(self) -> RoadOwnershipTrackPolicy:
        """Project the route-margin threshold used to collect raw tracks."""

        return RoadOwnershipTrackPolicy(
            route_margin_fraction=self.route_margin_fraction,
        )

    def persistence_policy(self) -> RoadOwnershipPersistencePolicy:
        """Project the thresholds used to stabilize one boundary track."""

        return RoadOwnershipPersistencePolicy(
            local_window_fraction=self.local_window_fraction,
            minimum_local_support_fraction=(
                self.minimum_local_support_fraction
            ),
            maximum_gap_fraction=self.maximum_gap_fraction,
            maximum_step_fraction=self.maximum_step_fraction,
            frame_margin_fraction=self.frame_margin_fraction,
        )


@dataclass(frozen=True, slots=True)
class SurfaceOwnershipRowEvidence:
    """One row of route-centered ownership evidence and its decision."""

    y: int
    route_x: float
    left_boundary_x: int | None
    right_boundary_x: int | None
    left_persistent: bool
    right_persistent: bool
    left_evidence_kind: str | None
    right_evidence_kind: str | None
    left_reason: str | None
    right_reason: str | None
    status: str

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen row record and key order."""

        return {
            "y": self.y,
            "route_x": self.route_x,
            "left_boundary_x": self.left_boundary_x,
            "right_boundary_x": self.right_boundary_x,
            "left_persistent": self.left_persistent,
            "right_persistent": self.right_persistent,
            "left_evidence_kind": self.left_evidence_kind,
            "right_evidence_kind": self.right_evidence_kind,
            "left_reason": self.left_reason,
            "right_reason": self.right_reason,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class IdentitySurfaceOwnershipDiagnostics:
    """Diagnostics emitted by the frozen identity/no-op ownership policy."""

    policy: str
    status: str
    source_pixels: int
    approved_removal_pixels: int

    def to_legacy_record(self) -> dict[str, object]:
        """Return the exact four-key identity diagnostic record."""

        return {
            "policy": self.policy,
            "status": self.status,
            "source_pixels": self.source_pixels,
            "approved_removal_pixels": self.approved_removal_pixels,
        }


@dataclass(frozen=True, slots=True)
class ActiveSurfaceOwnershipDiagnostics:
    """Aggregate diagnostics emitted by active route-centered ownership."""

    policy: str
    status: str
    scene_veto_reason: str | None
    route_quality: RouteQualityEvidence
    source_pixels: int
    boundary_evidence_pixels: int
    persistent_boundary_pixels: int
    left_persistent_rows: int
    right_persistent_rows: int
    interpolated_gap_rows: int
    rows_without_persistent_boundary: int
    longest_bilateral_boundary_gap_rows: int
    bilateral_boundary_gap_runs: tuple[tuple[int, int], ...]
    uncertain_pixels: int
    veto_pixels: int
    approved_removal_pixels: int
    route_corridor_removed_pixels: int
    obstacle_or_footprint_evidence_used: bool

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen active diagnostic record and key order."""

        return {
            "policy": self.policy,
            "status": self.status,
            "scene_veto_reason": self.scene_veto_reason,
            "route_quality": self.route_quality.to_legacy_record(),
            "source_pixels": self.source_pixels,
            "boundary_evidence_pixels": self.boundary_evidence_pixels,
            "persistent_boundary_pixels": self.persistent_boundary_pixels,
            "left_persistent_rows": self.left_persistent_rows,
            "right_persistent_rows": self.right_persistent_rows,
            "interpolated_gap_rows": self.interpolated_gap_rows,
            "rows_without_persistent_boundary": (
                self.rows_without_persistent_boundary
            ),
            "longest_bilateral_boundary_gap_rows": (
                self.longest_bilateral_boundary_gap_rows
            ),
            "bilateral_boundary_gap_runs": [
                [begin, end]
                for begin, end in self.bilateral_boundary_gap_runs
            ],
            "uncertain_pixels": self.uncertain_pixels,
            "veto_pixels": self.veto_pixels,
            "approved_removal_pixels": self.approved_removal_pixels,
            "route_corridor_removed_pixels": (
                self.route_corridor_removed_pixels
            ),
            "obstacle_or_footprint_evidence_used": (
                self.obstacle_or_footprint_evidence_used
            ),
        }


SurfaceOwnershipDiagnostics = (
    IdentitySurfaceOwnershipDiagnostics | ActiveSurfaceOwnershipDiagnostics
)


@dataclass(frozen=True, slots=True)
class SurfaceOwnershipResult:
    """Typed masks and evidence from route-centered surface ownership."""

    profile: SurfaceOwnershipProfile
    ownership_mask: np.ndarray
    envelope_mask: np.ndarray
    approved_removal_mask: np.ndarray
    uncertain_mask: np.ndarray
    veto_mask: np.ndarray
    boundary_evidence_mask: np.ndarray
    persistent_boundary_mask: np.ndarray
    route_corridor_mask: np.ndarray
    left_track_x: np.ndarray
    right_track_x: np.ndarray
    rows: tuple[SurfaceOwnershipRowEvidence, ...]
    diagnostics: SurfaceOwnershipDiagnostics

    def legacy_row_records(self) -> list[dict[str, object]]:
        """Return fresh frozen-format row records."""

        return [row.to_legacy_record() for row in self.rows]

    def legacy_diagnostics(self) -> dict[str, object]:
        """Return a fresh frozen-format diagnostics record."""

        return self.diagnostics.to_legacy_record()
