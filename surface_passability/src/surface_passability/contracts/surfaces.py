"""Typed contracts for surface construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from surface_passability.contracts.boundaries import (
        AcceptedSideBranchGroup,
        SideBranchCandidate,
    )
    from surface_passability.contracts.obstacles import (
        ObstacleGeometry,
        ObstacleOwnershipResult,
    )


@dataclass(frozen=True)
class SurfaceSelectionEvidence:
    """Phase-6 viability and score evidence for one prepared surface."""

    surface: str
    source_pixels: int
    source_fraction: float
    structural_pixels: int
    structural_fraction: float
    anchor_source_fraction: float
    anchor_structural_fraction: float
    central_vertical_coverage: float
    route_progress: float
    route_point_count: int
    structural_source_support: float
    addition_fraction: float
    bilateral_curb_fraction: float
    source_quality: float
    viable: bool
    score: float
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


SurfaceSelectionState = Literal[
    "sidewalk", "road", "uncertain_both", "uncertain_none"
]


@dataclass(frozen=True)
class SurfaceSelectionResult:
    """Phase-6 selected surface and complete decision evidence."""

    input_profile: str
    state: SurfaceSelectionState
    selected_surface: str | None
    fallback: str | None
    sidewalk: SurfaceSelectionEvidence
    road: SurfaceSelectionEvidence
    score_margin: float
    reasons: tuple[str, ...]
    profile_metrics: dict

    def to_dict(self) -> dict:
        return {
            "input_profile": self.input_profile,
            "state": self.state,
            "selected_surface": self.selected_surface,
            "fallback": self.fallback,
            "score_margin": self.score_margin,
            "reasons": list(self.reasons),
            "profile_metrics": self.profile_metrics,
            "sidewalk": self.sidewalk.to_dict(),
            "road": self.road.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ObstacleSurfaceInput:
    """Minimal surface evidence required for obstacle-trigger preparation."""

    name: str
    source_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class ObstacleOwnershipSurfaceInput:
    """Surface masks required by the complete obstacle-ownership policy."""

    name: str
    source_mask: np.ndarray
    repaired_mask: np.ndarray
    structural_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class SurfaceProfile:
    """Behavioral identity and routing policy for one semantic surface."""

    name: str
    source_class_ids: tuple[int, ...]
    repair_class_ids: tuple[int, ...]
    forbidden_class_ids: tuple[int, ...]
    side_pruning_enabled: bool


@dataclass(frozen=True, slots=True)
class ConnectedSurfaceComponent:
    """One eight-connected surface component in OpenCV label order."""

    component_index: int
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "index": self.component_index,
            "mask": self.mask,
            "area": self.area,
            "bbox": list(self.bbox),
        }


@dataclass(frozen=True, slots=True)
class SurfaceForwardAlignment:
    """Forward route evidence for one obstacle-linked surface component."""

    available: bool
    accepted: bool
    forward_projection_px: float | None = None
    cosine: float | None = None
    selected_contact_center_yx: tuple[float, float] | None = None
    satellite_contact_center_yx: tuple[float, float] | None = None

    def to_legacy_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "available": self.available,
            "accepted": self.accepted,
        }
        if not self.available:
            return record
        record["forward_projection_px"] = self.forward_projection_px
        record["cosine"] = self.cosine
        if self.selected_contact_center_yx is not None:
            record["selected_contact_center_yx"] = list(
                self.selected_contact_center_yx
            )
            record["satellite_contact_center_yx"] = list(
                self.satellite_contact_center_yx
            )
        return record


@dataclass(frozen=True, slots=True)
class SurfaceRowSpans:
    """First and last surface columns for every valid image row."""

    valid_rows: np.ndarray
    left_columns: np.ndarray
    right_columns: np.ndarray

    def to_legacy_result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.valid_rows, self.left_columns, self.right_columns


@dataclass(frozen=True, slots=True)
class SurfaceComponentSelectionInput:
    """Minimum surface state required for obstacle-linked component selection."""

    structural_mask: np.ndarray
    forbidden_mask: np.ndarray
    route_start: tuple[int, int] | None
    route_path: tuple[tuple[int, int], ...]
    observed_component: np.ndarray


@dataclass(frozen=True, slots=True)
class ObstacleComponentContactMatch:
    """One obstacle authorizing contact between selected and satellite surface."""

    obstacle_record_index: int
    obstacle_class_name: str | None
    obstacle_kind: str | None
    obstacle_strategy: str | None
    obstacle_area: int
    obstacle_bbox: object
    selected_contact_pixels: int
    satellite_contact_pixels: int
    forward_alignment: SurfaceForwardAlignment

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "obstacle_record_index": self.obstacle_record_index,
            "obstacle_class_name": self.obstacle_class_name,
            "obstacle_kind": self.obstacle_kind,
            "obstacle_strategy": self.obstacle_strategy,
            "obstacle_area": self.obstacle_area,
            "obstacle_bbox": self.obstacle_bbox,
            "selected_contact_pixels": self.selected_contact_pixels,
            "satellite_contact_pixels": self.satellite_contact_pixels,
            "forward_alignment": self.forward_alignment.to_legacy_record(),
        }


@dataclass(frozen=True, slots=True)
class AcceptedSurfaceSatellite:
    """One surface component admitted by obstacle-contact evidence."""

    component_index: int
    area: int
    bbox: tuple[int, int, int, int]
    obstacle_matches: tuple[ObstacleComponentContactMatch, ...]

    def to_legacy_record(self) -> dict[str, object]:
        return {
            "candidate_component_index": self.component_index,
            "candidate_area": self.area,
            "candidate_bbox": list(self.bbox),
            "authorization": "same_obstacle_touches_selected_and_satellite",
            "obstacle_matches": [
                match.to_legacy_record()
                for match in self.obstacle_matches
            ],
        }


@dataclass(frozen=True, slots=True)
class SurfaceComponentSelection:
    """Complete obstacle-linked component selection for one surface."""

    candidate_mask: np.ndarray
    main_component_mask: np.ndarray
    selected_source_mask: np.ndarray
    accepted_satellite_mask: np.ndarray
    rejected_source_mask: np.ndarray
    span_mask: np.ndarray
    components: tuple[ConnectedSurfaceComponent, ...]
    main_component_index: int | None
    accepted_component_indices: tuple[int, ...]
    evidence: tuple[AcceptedSurfaceSatellite, ...]
    obstacle_component_count: int
    contact_radius_px: int
    empty_candidate: bool
    forbidden_crossing_rejections: int | None = None

    def to_legacy_result(self) -> dict[str, object]:
        result: dict[str, object] = {
            "candidate": self.candidate_mask,
            "main": self.main_component_mask,
            "selected_sources": self.selected_source_mask,
            "accepted_satellites": self.accepted_satellite_mask,
            "rejected_sources": self.rejected_source_mask,
            "span": self.span_mask,
            "components": [
                component.to_legacy_record()
                for component in self.components
            ],
            "main_component_index": self.main_component_index,
            "accepted_component_indices": list(
                self.accepted_component_indices
            ),
            "evidence": [item.to_legacy_record() for item in self.evidence],
            "obstacle_component_count": self.obstacle_component_count,
            "contact_radius_px": self.contact_radius_px,
            "empty_candidate": self.empty_candidate,
        }
        if self.forbidden_crossing_rejections is not None:
            result["forbidden_crossing_rejections"] = (
                self.forbidden_crossing_rejections
            )
        return result


@dataclass(frozen=True, slots=True)
class SurfaceObstacleSelectionResult:
    """One surface's obstacle ownership and connected-component selection."""

    ownership: ObstacleOwnershipResult
    component_selection: SurfaceComponentSelection


@dataclass(frozen=True, slots=True)
class SurfaceObstacleAnalysis:
    """Prepared image geometry and both surface-specific selection results."""

    geometry: ObstacleGeometry
    sidewalk: SurfaceObstacleSelectionResult
    road: SurfaceObstacleSelectionResult


@dataclass(frozen=True, slots=True)
class SurfaceRepairComponent:
    """Decision evidence for one connected repair-class component."""

    class_id: int
    class_name: str
    component_index: int
    pixels: int
    sidewalk_ring_fraction: float
    road_ring_fraction: float
    component_fraction: float
    touches_frame: bool
    catch_basin_geometry_ok: bool
    accepted: bool

    def to_diagnostic_record(self) -> dict[str, object]:
        """Return the accepted diagnostic field contract."""

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "component_index": self.component_index,
            "pixels": self.pixels,
            "sidewalk_ring_fraction": self.sidewalk_ring_fraction,
            "road_ring_fraction": self.road_ring_fraction,
            "component_fraction": self.component_fraction,
            "touches_frame": self.touches_frame,
            "catch_basin_geometry_ok": self.catch_basin_geometry_ok,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class SidewalkRepairResult:
    """Observed Sidewalk plus context-approved semantic repairs."""

    source_mask: np.ndarray
    repaired_mask: np.ndarray
    addition_mask: np.ndarray
    components: tuple[SurfaceRepairComponent, ...]


@dataclass(frozen=True, slots=True)
class CurbComponentEvidence:
    """Boundary-context evidence for one observed Sidewalk component."""

    component_id: int
    pixels: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    centroid_x: float
    centroid_y: float
    boundary_context_fraction: float
    minimum_pixels_for_image: int
    anchor: bool
    flagged_suspect: bool

    def to_diagnostic_record(self) -> dict[str, object]:
        """Return the accepted component diagnostic field contract."""

        return {
            "component_id": self.component_id,
            "pixels": self.pixels,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "boundary_context_fraction": self.boundary_context_fraction,
            "minimum_pixels_for_image": self.minimum_pixels_for_image,
            "anchor": self.anchor,
            "flagged_suspect": self.flagged_suspect,
        }


@dataclass(frozen=True, slots=True)
class CurbEvidence:
    """Disconnected-Sidewalk evidence used to exclude curb-side suspects."""

    component_labels: np.ndarray
    suspect_mask: np.ndarray
    anchor_component_id: int
    components: tuple[CurbComponentEvidence, ...]


@dataclass(frozen=True, slots=True)
class SurfaceStructureResult:
    """Morphologically continuous measurement structure for one surface."""

    structural_mask: np.ndarray
    addition_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    """Typed evidence for one requested route-start fraction."""

    requested_start_fraction: float
    start: tuple[int, int] | None
    component_mask: np.ndarray
    target_y: int
    goals: tuple[tuple[int, int], ...]
    paths: tuple[tuple[tuple[int, int], ...], ...]
    totals: tuple[float, ...]
    main_index: int | None
    angle: float | None
    vertical_progress: float
    high_confidence: bool
    quality: float

    @property
    def main_path(self) -> tuple[tuple[int, int], ...]:
        """Return the selected route path, or an empty path when none exists."""

        return self.paths[self.main_index] if self.main_index is not None else ()

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen mapping shape used by current comparators."""

        return {
            "requested_start_fraction": self.requested_start_fraction,
            "start": self.start,
            "component": self.component_mask,
            "target_y": self.target_y,
            "goals": list(self.goals),
            "paths": [list(path) for path in self.paths],
            "totals": list(self.totals),
            "main_index": self.main_index,
            "angle": self.angle,
            "vertical_progress": self.vertical_progress,
            "high_confidence": self.high_confidence,
            "quality": self.quality,
        }


@dataclass(slots=True)
class RecomputedStructuralRoute:
    """Route recomputed on a completed structural surface boundary."""

    start: tuple[int, int] | None
    component_mask: np.ndarray
    target_y: int
    goals: list[tuple[int, int]]
    paths: list[list[tuple[int, int]]]
    totals: list[float]
    main_index: int | None
    selected_path: list[tuple[int, int]]
    angle: float | None
    vertical_progress: float
    high_confidence: bool
    observed_start_reused: bool
    observed_direction_angle_degrees: float | None
    direction_constraint: str

    def to_legacy_record(self) -> dict[str, object]:
        """Return the frozen mapping shape used by parity and response code."""

        return {
            "requested_start_fraction": None,
            "start": self.start,
            "component": self.component_mask,
            "target_y": self.target_y,
            "goals": self.goals,
            "paths": self.paths,
            "totals": self.totals,
            "main_index": self.main_index,
            "angle": self.angle,
            "vertical_progress": self.vertical_progress,
            "high_confidence": self.high_confidence,
            "observed_start_reused": self.observed_start_reused,
            "observed_direction_angle_deg": (
                self.observed_direction_angle_degrees
            ),
            "direction_constraint": self.direction_constraint,
        }


@dataclass(slots=True)
class StructuralRouteStageResult:
    """Complete boundary, structural-route, and side-pruning stage result."""

    raw_boundary_mask: np.ndarray
    pruned_boundary_mask: np.ndarray
    route: RouteAttempt | RecomputedStructuralRoute
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]]
    route_displacement: dict[str, object]
    raw_candidate_mask: np.ndarray
    vetoed_candidate_mask: np.ndarray
    approved_removal_mask: np.ndarray
    candidate_rows: list[SideBranchCandidate]
    accepted_groups: list[AcceptedSideBranchGroup]
    calls: dict[str, bool]

    def route_to_legacy_record(self) -> dict[str, object]:
        return self.route.to_legacy_record()

    def candidate_rows_to_legacy_records(self) -> list[dict[str, object]]:
        return [candidate.to_legacy_record() for candidate in self.candidate_rows]

    def accepted_groups_to_legacy_records(self) -> list[dict[str, object]]:
        return [group.to_legacy_record() for group in self.accepted_groups]


@dataclass(slots=True)
class SurfaceContext:
    """Surface-local state produced before obstacle and boundary processing."""

    profile: SurfaceProfile
    source_mask: np.ndarray
    repaired_mask: np.ndarray
    structural_mask: np.ndarray
    forbidden_mask: np.ndarray
    observed_component: np.ndarray
    route: RouteAttempt
    route_path: tuple[tuple[int, int], ...]
    structural_addition_mask: np.ndarray


@dataclass(slots=True)
class SurfacePipelineContext:
    """Image-level evidence and one pre-obstacle semantic surface context."""

    image: np.ndarray
    class_map: np.ndarray
    surface: SurfaceContext

    @property
    def repaired(self) -> np.ndarray:
        return self.surface.repaired_mask

    @property
    def structure_copy(self) -> np.ndarray:
        return self.surface.structural_mask

    @property
    def structure_copy_additions(self) -> np.ndarray:
        return self.surface.structural_addition_mask

    @property
    def curb_suspect(self) -> np.ndarray:
        return self.surface.forbidden_mask

    @property
    def route(self) -> RouteAttempt:
        return self.surface.route

    @property
    def route_path(self) -> tuple[tuple[int, int], ...]:
        return self.surface.route_path

    @property
    def source_walkway(self) -> np.ndarray:
        return self.surface.source_mask

    @property
    def observed_component(self) -> np.ndarray:
        return self.surface.observed_component

