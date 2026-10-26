"""Typed evidence contracts for Road drainage analysis."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from surface_passability.contracts.road import RoadContext


@dataclass(frozen=True, slots=True)
class DrainComponentGeometry:
    """PCA geometry of one candidate transverse-drain component."""

    center: np.ndarray
    major_axis: np.ndarray
    minor_axis: np.ndarray
    major_span_px: float
    minor_span_px: float
    pca_elongation: float
    major_axis_angle_deg: float

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen geometry record with exact key order."""

        return {
            "center": self.center,
            "major": self.major_axis,
            "minor": self.minor_axis,
            "major_span_px": self.major_span_px,
            "minor_span_px": self.minor_span_px,
            "pca_elongation": self.pca_elongation,
            "major_axis_angle_deg": self.major_axis_angle_deg,
        }


@dataclass(frozen=True, slots=True)
class DrainRouteVectorEvidence:
    """Evidence describing how the local drain route vector was estimated."""

    status: str
    point_count: int
    local_radius_px: int | None = None

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen route-direction record with conditional radius."""

        record: dict[str, object] = {
            "status": self.status,
            "point_count": self.point_count,
        }
        if self.local_radius_px is not None:
            record["local_radius_px"] = self.local_radius_px
        return record


@dataclass(frozen=True, slots=True)
class DrainSideSupportEvidence:
    """Road-family support measured on one side of a drain component."""

    side: str
    band_pixels: int
    road_support_pixels: int
    road_support_fraction: float

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen per-side support record with exact key order."""

        return {
            "side": self.side,
            "band_pixels": self.band_pixels,
            "road_support_pixels": self.road_support_pixels,
            "road_support_fraction": self.road_support_fraction,
        }


@dataclass(frozen=True, slots=True)
class DrainCorridorEvidence:
    """Evidence for an estimated transverse-drain Road corridor."""

    status: str
    before_row_count: int
    after_row_count: int
    band_depth_px: int
    left_fit_x_per_y: tuple[float, float] | None = None
    right_fit_x_per_y: tuple[float, float] | None = None
    center_fit_x_per_y: tuple[float, float] | None = None
    route_vector_xy: tuple[float, float] | None = None

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen corridor record with conditional fit evidence."""

        record: dict[str, object] = {
            "status": self.status,
            "before_row_count": self.before_row_count,
            "after_row_count": self.after_row_count,
            "band_depth_px": self.band_depth_px,
        }
        if self.left_fit_x_per_y is not None:
            record["left_fit_x_per_y"] = list(self.left_fit_x_per_y)
            record["right_fit_x_per_y"] = list(self.right_fit_x_per_y)
            record["center_fit_x_per_y"] = list(self.center_fit_x_per_y)
            record["route_vector_xy"] = list(self.route_vector_xy)
        return record


@dataclass(frozen=True, slots=True)
class DrainFrameContactEvidence:
    """Image-frame edges touched by one drain component mask."""

    left: bool
    right: bool
    top: bool
    bottom: bool

    def to_legacy_record(self) -> dict[str, bool]:
        """Return the frozen frame-contact record with exact key order."""

        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }


@dataclass(frozen=True, slots=True)
class LinearDrainRouteDirectionEvidence:
    """Selected route-direction evidence for one drain component."""

    base: DrainRouteVectorEvidence
    selected_estimator: str
    path_route_vector_xy: tuple[float, float]
    corridor_route_vector_xy: tuple[float, float]
    corridor_edges_converge: bool

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen route-direction record with exact key order."""

        record = self.base.to_legacy_record()
        record.update(
            {
                "selected_estimator": self.selected_estimator,
                "path_route_vector_xy": list(self.path_route_vector_xy),
                "corridor_route_vector_xy": list(
                    self.corridor_route_vector_xy
                ),
                "corridor_edges_converge": self.corridor_edges_converge,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class LinearDrainConditions:
    """Ordered acceptance conditions for one drain component."""

    minimum_pixels: bool
    long_and_thin: bool
    transverse_to_route: bool
    independent_road_support_before: bool
    independent_road_support_after: bool
    corridor_estimated_from_both_sides: bool
    corridor_intersection: bool
    approved_region_avoids_hard_exclusion: bool

    def to_legacy_record(self) -> dict[str, bool]:
        """Return the frozen condition record with exact insertion order."""

        return {
            "minimum_pixels": self.minimum_pixels,
            "long_and_thin": self.long_and_thin,
            "transverse_to_route": self.transverse_to_route,
            "independent_road_support_before": (
                self.independent_road_support_before
            ),
            "independent_road_support_after": (
                self.independent_road_support_after
            ),
            "corridor_estimated_from_both_sides": (
                self.corridor_estimated_from_both_sides
            ),
            "corridor_intersection": self.corridor_intersection,
            "approved_region_avoids_hard_exclusion": (
                self.approved_region_avoids_hard_exclusion
            ),
        }


@dataclass(frozen=True, slots=True)
class LinearDrainComponentEvidence:
    """Complete decision evidence for one Catch Basin component."""

    component_index: int
    original_class_id: int
    original_class_name: str
    decision_type: str
    pixels: int
    component_fraction: float
    bbox_xywh: tuple[int, int, int, int]
    bbox_aspect_ratio: float
    geometry: DrainComponentGeometry
    estimated_route_vector_xy: tuple[float, float]
    route_direction: LinearDrainRouteDirectionEvidence
    transverse_angle_deg: float
    frame_contact: DrainFrameContactEvidence
    support_depth_px: int
    side_support: tuple[DrainSideSupportEvidence, ...]
    corridor: DrainCorridorEvidence
    corridor_pixels_in_component: int
    corridor_clipped_candidate_pixels: int
    forbidden_overlap_pixels: int
    approved_hard_exclusion_overlap_pixels: int
    approved_pixels: int
    approved_fraction_of_component: float
    whole_object_restoration_enabled: bool
    whole_object_restored_pixels: int
    already_in_production_repair_pixels: int
    conditions: LinearDrainConditions
    base_transverse_condition_passed: bool
    strong_corridor_crossing_exception_passed: bool
    accepted: bool
    decision_reasons: tuple[str, ...]

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen component diagnostic record."""

        return {
            "component_index": self.component_index,
            "original_class_id": self.original_class_id,
            "original_class_name": self.original_class_name,
            "decision_type": self.decision_type,
            "pixels": self.pixels,
            "component_fraction": self.component_fraction,
            "bbox_xywh": list(self.bbox_xywh),
            "bbox_aspect_ratio": self.bbox_aspect_ratio,
            "pca_elongation": self.geometry.pca_elongation,
            "major_span_px": self.geometry.major_span_px,
            "minor_span_px": self.geometry.minor_span_px,
            "major_axis_angle_deg": self.geometry.major_axis_angle_deg,
            "estimated_route_vector_xy": list(
                self.estimated_route_vector_xy
            ),
            "route_direction": self.route_direction.to_legacy_record(),
            "transverse_angle_deg": self.transverse_angle_deg,
            "frame_contact": self.frame_contact.to_legacy_record(),
            "support_depth_px": self.support_depth_px,
            "side_support": [
                item.to_legacy_record() for item in self.side_support
            ],
            "corridor": self.corridor.to_legacy_record(),
            "corridor_pixels_in_component": (
                self.corridor_pixels_in_component
            ),
            "corridor_clipped_candidate_pixels": (
                self.corridor_clipped_candidate_pixels
            ),
            "forbidden_overlap_pixels": self.forbidden_overlap_pixels,
            "approved_hard_exclusion_overlap_pixels": (
                self.approved_hard_exclusion_overlap_pixels
            ),
            "approved_pixels": self.approved_pixels,
            "approved_fraction_of_component": (
                self.approved_fraction_of_component
            ),
            "whole_object_restoration_enabled": (
                self.whole_object_restoration_enabled
            ),
            "whole_object_restored_pixels": (
                self.whole_object_restored_pixels
            ),
            "already_in_production_repair_pixels": (
                self.already_in_production_repair_pixels
            ),
            "conditions": self.conditions.to_legacy_record(),
            "base_transverse_condition_passed": (
                self.base_transverse_condition_passed
            ),
            "strong_corridor_crossing_exception_passed": (
                self.strong_corridor_crossing_exception_passed
            ),
            "accepted": self.accepted,
            "decision_reasons": list(self.decision_reasons),
        }


@dataclass(frozen=True, slots=True)
class LinearDrainComponentAnalysis:
    """Masks and typed evidence produced for one component."""

    corridor_mask: np.ndarray
    approved_mask: np.ndarray
    evidence: LinearDrainComponentEvidence


@dataclass(frozen=True, slots=True)
class LinearDrainPolicy:
    """Serializable accepted policy for linear transverse-drain recovery."""

    probe_only: bool
    production_policy_modified: bool
    catch_basin_class_id: int
    minimum_component_pixels: int
    minimum_pca_elongation: float
    minimum_transverse_angle_deg: float
    strong_minimum_transverse_angle_deg: float
    strong_minimum_pca_elongation: float
    strong_minimum_side_support_fraction_each: float
    strong_minimum_corridor_component_fraction: float
    maximum_minor_span_fraction: float
    maximum_component_fraction: float
    support_depth_fraction: float
    minimum_side_support_pixels: int
    minimum_side_support_fraction: float
    minimum_corridor_approved_pixels: int
    minimum_corridor_component_fraction: float
    corridor_band_depth_fraction: float
    corridor_margin_px: int
    approval_clipped_to_estimated_corridor: bool
    whole_accepted_component_restoration: bool

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen aggregate policy record."""

        return {
            "probe_only": self.probe_only,
            "production_policy_modified": self.production_policy_modified,
            "catch_basin_class_id": self.catch_basin_class_id,
            "minimum_component_pixels": self.minimum_component_pixels,
            "minimum_pca_elongation": self.minimum_pca_elongation,
            "minimum_transverse_angle_deg": (
                self.minimum_transverse_angle_deg
            ),
            "strong_corridor_exception": {
                "minimum_transverse_angle_deg": (
                    self.strong_minimum_transverse_angle_deg
                ),
                "minimum_pca_elongation": (
                    self.strong_minimum_pca_elongation
                ),
                "minimum_side_support_fraction_each": (
                    self.strong_minimum_side_support_fraction_each
                ),
                "minimum_corridor_component_fraction": (
                    self.strong_minimum_corridor_component_fraction
                ),
                "all_conditions_required": True,
            },
            "maximum_minor_span_fraction": self.maximum_minor_span_fraction,
            "maximum_component_fraction": self.maximum_component_fraction,
            "support_depth_fraction": self.support_depth_fraction,
            "minimum_side_support_pixels": self.minimum_side_support_pixels,
            "minimum_side_support_fraction": (
                self.minimum_side_support_fraction
            ),
            "minimum_corridor_approved_pixels": (
                self.minimum_corridor_approved_pixels
            ),
            "minimum_corridor_component_fraction": (
                self.minimum_corridor_component_fraction
            ),
            "corridor_band_depth_fraction": (
                self.corridor_band_depth_fraction
            ),
            "corridor_margin_px": self.corridor_margin_px,
            "approval_clipped_to_estimated_corridor": (
                self.approval_clipped_to_estimated_corridor
            ),
            "whole_accepted_component_restoration": (
                self.whole_accepted_component_restoration
            ),
            "whole_object_eligibility_uses_corridor_clipped_evidence": True,
            "hard_exclusion_applied_after_approval": True,
            "original_catch_basin_diagnostics_preserved": True,
        }


@dataclass(frozen=True, slots=True)
class LinearDrainAnalysisResult:
    """Typed aggregate result for linear transverse-drain recovery."""

    road: RoadContext
    raw_catch_basin_mask: np.ndarray
    estimated_corridor_mask: np.ndarray
    approved_linear_drain_mask: np.ndarray
    rejected_linear_drain_mask: np.ndarray
    components: tuple[LinearDrainComponentEvidence, ...]
    policy: LinearDrainPolicy

    def component_records(self) -> list[dict[str, object]]:
        """Return fresh frozen-format component diagnostics."""

        return [component.to_legacy_record() for component in self.components]

    def road_policy_record(self) -> dict[str, object]:
        """Return the frozen Road policy after drain evaluation."""

        record = self.road.policy.to_legacy_record()
        record.update(
            {
                "linear_transverse_drain_enabled": True,
                "linear_transverse_drain_probe_enabled": (
                    self.policy.probe_only
                ),
                "linear_drain_whole_object_restoration": (
                    self.policy.whole_accepted_component_restoration
                ),
                "production_catch_basin_rule_changed": (
                    self.policy.production_policy_modified
                ),
            }
        )
        return record
