"""A-to-Z orchestration boundaries for the surface-passability model."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from surface_passability.contracts.analysis import (
    SurfaceCandidate,
    SurfaceCandidates,
    SurfaceSelection,
    SelectedSurfaceAnalysis,
)
from surface_passability.contracts.request import AnalysisRequest
from surface_passability.contracts.response import AnalysisResponse
from surface_passability.boundary.pipeline import analyze_surface_boundary
from surface_passability.boundary.line_fitting import request_robust_fit_cache
from surface_passability.passability.authority import build_passability_handoff
from surface_passability.passability.occupancy import build_measurement_occupancy
from surface_passability.passability.pipeline import compose_passability_result
from surface_passability.contracts.road import RoadContext
from surface_passability.contracts.request import InputProfile
from surface_passability.contracts.road_analysis import (
    RoadCandidateComputation,
)
from surface_passability.contracts.semantic import SemanticEvidence
from surface_passability.contracts.sidewalk import (
    SidewalkCandidateComputation,
)
from surface_passability.contracts.surfaces import (
    ObstacleOwnershipSurfaceInput,
    ObstacleSurfaceInput,
    SurfaceContext,
    SurfaceComponentSelectionInput,
    SurfaceObstacleAnalysis,
    SurfaceObstacleSelectionResult,
    SurfacePipelineContext,
    SurfaceSelectionResult,
)
from surface_passability.inference.vehicle_instances import VehicleModelProvider
from surface_passability.dependencies import (
    PipelineDependencies,
    get_default_dependencies,
)
from surface_passability.imaging.preparation import prepare_image
from surface_passability.obstacles.evidence import (
    extract_obstacle_geometry,
    prepare_obstacle_geometry,
)
from surface_passability.obstacles.taxonomy import (
    HAZARD_CLASS_IDS,
    OCCUPANCY_CLASS_IDS,
)
from surface_passability.policies.selection import SurfaceSelectionPolicy
from surface_passability.selection.surface_selector import (
    select_surface_for_input_profile,
)
from surface_passability.surfaces.road import (
    analyze_road_obstacle_components,
    evaluate_road_structural_rows,
    prepare_road_boundaries,
    prepare_road_structural_stage,
)
from surface_passability.surfaces.context import build_surface_pipeline_context
from surface_passability.surfaces.road_context import prepare_road_surface_context
from surface_passability.surfaces.sidewalk import (
    analyze_sidewalk_obstacle_components,
    evaluate_sidewalk_structural_rows,
    prepare_sidewalk_structural_stages,
)
from surface_passability.surfaces.routes import request_route_cache


def _analyze_surface_obstacles(
    image: np.ndarray,
    class_map: np.ndarray,
    class_names: Mapping[int | str, str],
    sidewalk_ownership: ObstacleOwnershipSurfaceInput,
    road_ownership: ObstacleOwnershipSurfaceInput,
    sidewalk_selection: SurfaceComponentSelectionInput,
    road_selection: SurfaceComponentSelectionInput,
    road_authorization_forbidden_mask: np.ndarray,
    vehicle_model_provider: VehicleModelProvider,
) -> SurfaceObstacleAnalysis:
    """Prepare shared geometry once and analyze both production surfaces."""

    geometry = extract_obstacle_geometry(image, class_map, class_names)
    geometry = prepare_obstacle_geometry(
        geometry,
        (
            ObstacleSurfaceInput(
                sidewalk_ownership.name,
                sidewalk_ownership.source_mask,
            ),
            ObstacleSurfaceInput(
                road_ownership.name,
                road_ownership.source_mask,
            ),
        ),
        vehicle_model_provider,
    )
    return SurfaceObstacleAnalysis(
        geometry=geometry,
        sidewalk=analyze_sidewalk_obstacle_components(
            geometry,
            sidewalk_ownership,
            class_map,
            sidewalk_selection,
        ),
        road=analyze_road_obstacle_components(
            geometry,
            road_ownership,
            class_map,
            road_selection,
            road_authorization_forbidden_mask,
        ),
    )


def _calculate_sidewalk_candidate(
    context: SurfacePipelineContext,
    obstacle_selection: SurfaceObstacleSelectionResult,
) -> SidewalkCandidateComputation:
    """Compose the approved Sidewalk structural and row calculations once."""

    ownership = obstacle_selection.ownership
    selection = obstacle_selection.component_selection
    preparation = prepare_sidewalk_structural_stages(context, selection)
    rows = evaluate_sidewalk_structural_rows(
        preparation,
        selection,
        ownership,
        context,
    )
    return SidewalkCandidateComputation(
        ownership=ownership,
        selection=selection,
        current_stage=preparation.current_stage,
        proposed_stage=preparation.proposed_stage,
        sparse_support=rows.sparse_support,
        final_evaluation=rows.final_evaluation,
    )


def _calculate_road_candidate(
    base_context: SurfacePipelineContext,
    road: RoadContext,
    obstacle_selection: SurfaceObstacleSelectionResult,
) -> RoadCandidateComputation:
    """Compose the approved Road structural and row calculations once."""

    ownership = obstacle_selection.ownership
    selection = obstacle_selection.component_selection
    preparation = prepare_road_boundaries(
        base_context,
        road,
        obstacle_selection,
    )
    stage = prepare_road_structural_stage(preparation, road)
    rows = evaluate_road_structural_rows(
        stage,
        selection,
        ownership,
        preparation.context,
    )
    return RoadCandidateComputation(
        ownership=ownership,
        selection=selection,
        context=preparation.context,
        stage=stage,
        sparse_support=rows.sparse_support,
        final_evaluation=rows.final_evaluation,
    )


def _obstacle_surface_inputs(
    surface: SurfaceContext,
) -> tuple[ObstacleOwnershipSurfaceInput, SurfaceComponentSelectionInput]:
    """Expose the two exact obstacle views required from one surface."""

    return (
        ObstacleOwnershipSurfaceInput(
            name=surface.profile.name,
            source_mask=surface.source_mask,
            repaired_mask=surface.repaired_mask,
            structural_mask=surface.structural_mask,
        ),
        SurfaceComponentSelectionInput(
            structural_mask=surface.structural_mask,
            forbidden_mask=surface.forbidden_mask,
            route_start=surface.route.start,
            route_path=surface.route_path,
            observed_component=surface.observed_component,
        ),
    )


def build_surface_candidates(
    image: np.ndarray,
    semantic: SemanticEvidence,
    class_names: Mapping[int | str, str],
    vehicle_model_provider: VehicleModelProvider,
) -> SurfaceCandidates:
    """Calculate both production surface candidates entirely in memory."""

    sidewalk_context = build_surface_pipeline_context(
        image,
        semantic.class_map,
    )
    road_preparation = prepare_road_surface_context(
        semantic.class_map,
        sidewalk_context.surface,
    )
    road = road_preparation.road
    sidewalk_ownership, sidewalk_selection = _obstacle_surface_inputs(
        sidewalk_context.surface
    )
    road_ownership, road_selection = _obstacle_surface_inputs(road.surface)
    obstacles = _analyze_surface_obstacles(
        image,
        semantic.class_map,
        class_names,
        sidewalk_ownership,
        road_ownership,
        sidewalk_selection,
        road_selection,
        road.road_forbidden_mask,
        vehicle_model_provider,
    )
    sidewalk_calculation = _calculate_sidewalk_candidate(
        sidewalk_context,
        obstacles.sidewalk,
    )
    road_calculation = _calculate_road_candidate(
        sidewalk_context,
        road,
        obstacles.road,
    )
    obstacle_silhouette = np.isin(
        semantic.class_map,
        tuple(OCCUPANCY_CLASS_IDS) + tuple(HAZARD_CLASS_IDS),
    )
    sidewalk_structural = (
        sidewalk_calculation.proposed_stage.pruned_boundary_mask
    )
    sidewalk_instances = tuple(
        sidewalk_calculation.ownership.to_legacy_result()[
            "passability_instances"
        ]
    )
    road_structural = road_calculation.final_evaluation.component_mask
    road_instances = tuple(
        road_calculation.ownership.to_legacy_result()["passability_instances"]
    )
    return SurfaceCandidates(
        sidewalk_context=sidewalk_context,
        road_context=road,
        obstacles=obstacles,
        sidewalk=SurfaceCandidate(
            name="Sidewalk",
            source_mask=sidewalk_context.surface.source_mask,
            repaired_mask=sidewalk_context.surface.repaired_mask,
            raw_structural_mask=(
                sidewalk_calculation.proposed_stage.raw_boundary_mask
            ),
            structural_mask=sidewalk_structural,
            blocker_mask=(
                sidewalk_calculation.ownership.final_mask
                & sidewalk_structural
            ),
            free_mask=sidewalk_calculation.final_evaluation.free_mask,
            obstacle_silhouette_mask=obstacle_silhouette,
            passability_instances=sidewalk_instances,
            calculation=sidewalk_calculation,
        ),
        road=SurfaceCandidate(
            name="Road",
            source_mask=road.surface.source_mask,
            repaired_mask=road.surface.repaired_mask,
            raw_structural_mask=road_calculation.stage.raw_boundary_mask,
            structural_mask=road_structural,
            blocker_mask=road_calculation.final_evaluation.blocker_mask,
            free_mask=road_calculation.final_evaluation.free_mask,
            obstacle_silhouette_mask=obstacle_silhouette,
            passability_instances=road_instances,
            calculation=road_calculation,
        ),
    )
def select_surface_candidate(
    input_profile: InputProfile,
    candidates: SurfaceCandidates,
    semantic: SemanticEvidence,
    policy: SurfaceSelectionPolicy = SurfaceSelectionPolicy(),
) -> SurfaceSelection:
    """Select one already-calculated candidate without recomputation."""

    decision = select_surface_for_input_profile(
        input_profile,
        candidates.sidewalk_context.surface,
        candidates.road_context,
        candidates.sidewalk.structural_mask.copy(),
        candidates.road.structural_mask.copy(),
        semantic.class_map,
        policy,
    )
    selected = {
        "sidewalk": candidates.sidewalk,
        "road": candidates.road,
    }.get(decision.selected_surface)
    return SurfaceSelection(decision=decision, candidate=selected)


def analyze_selected_surface(
    request: AnalysisRequest,
    selection: SurfaceSelection,
    semantic: SemanticEvidence,
) -> SelectedSurfaceAnalysis:
    """Run boundary and passability once for the selected in-memory candidate."""

    candidate = selection.candidate
    if candidate is None:
        shape = (semantic.native_size[1], semantic.native_size[0])
        structural = np.zeros(shape, dtype=bool)
        blocker = np.zeros(shape, dtype=bool)
        silhouette = np.zeros(shape, dtype=bool)
        raw_structural = None
        passability_instances: tuple[dict[str, object], ...] = ()
    else:
        shape = semantic.class_map.shape
        structural = candidate.structural_mask
        blocker = candidate.blocker_mask
        silhouette = candidate.obstacle_silhouette_mask
        raw_structural = candidate.raw_structural_mask
        passability_instances = candidate.passability_instances

    with request_robust_fit_cache():
        boundary = analyze_surface_boundary(
            request.sample_id,
            selection.decision.selected_surface,
            selection.decision.state,
            structural,
            blocker,
            semantic.class_map,
            obstacle_silhouette_mask=silhouette,
            raw_structural_mask=raw_structural,
        )
    handoff = build_passability_handoff(
        boundary,
        provenance={
            "request_id": request.request_id,
            "image_content_sha256": request.image_content_sha256,
            "semantic_cache_key": semantic.cache_key,
            "semantic_analysis_array_sha256": semantic.analysis_array_sha256,
            "surface_selection_state": selection.decision.state,
        },
    )
    straight = boundary.straight_result
    blocker_reuse = straight.get("blocker_reuse") or {}
    coverage_suspect = (
        straight.get("final_disposition")
        == "boundary_valid_but_blocker_suspect"
        or "blocker_coverage_suspect" in blocker_reuse.get("warnings", ())
    )
    occupancy = build_measurement_occupancy(
        handoff,
        candidate.blocker_mask if candidate is not None else None,
        passability_instances,
        blocker_coverage_suspect=coverage_suspect,
    )
    passability = compose_passability_result(
        handoff,
        occupancy,
        surface_width_m=request.surface_width_m,
        agent_width_m=request.agent_width_m,
        safety_factor=request.safety_factor,
    )
    return SelectedSurfaceAnalysis(
        selection=selection,
        boundary=boundary,
        passability=passability,
    )


def analyze_photo(
    request: AnalysisRequest,
    *,
    dependencies: PipelineDependencies | None = None,
) -> AnalysisResponse:
    """Run the complete artifact-free production pipeline for one photo."""

    dependencies = dependencies or get_default_dependencies()
    timings: dict[str, float] = {}
    started = _read_pipeline_clock(dependencies)

    prepared = prepare_image(request)
    checkpoint = _read_pipeline_clock(dependencies, previous=started)
    timings["image_preparation"] = checkpoint - started

    semantic = dependencies.semantic_provider.provide(prepared)
    previous = checkpoint
    checkpoint = _read_pipeline_clock(dependencies, previous=previous)
    timings["semantic_inference"] = checkpoint - previous

    with request_route_cache():
        candidates = build_surface_candidates(
            prepared.rgb,
            semantic,
            dependencies.class_names,
            dependencies.vehicle_model_provider,
        )
    previous = checkpoint
    checkpoint = _read_pipeline_clock(dependencies, previous=previous)
    timings["surface_candidates"] = checkpoint - previous

    selection = select_surface_candidate(
        request.input_profile,
        candidates,
        semantic,
        dependencies.selection_policy,
    )
    previous = checkpoint
    checkpoint = _read_pipeline_clock(dependencies, previous=previous)
    timings["surface_selection"] = checkpoint - previous

    selected = analyze_selected_surface(request, selection, semantic)
    previous = checkpoint
    checkpoint = _read_pipeline_clock(dependencies, previous=previous)
    timings["boundary_and_passability"] = checkpoint - previous
    timings["total"] = checkpoint - started

    return AnalysisResponse(
        request_id=request.request_id,
        sample_id=request.sample_id,
        image=prepared,
        semantic=semantic,
        surface_selection=selected.selection.decision,
        boundary=selected.boundary,
        passability=selected.passability,
        artifacts=(),
        stage_timings_s=timings,
    )


def _read_pipeline_clock(
    dependencies: PipelineDependencies,
    *,
    previous: float | None = None,
) -> float:
    """Reject a malformed injected clock before it can taint a response."""

    value = dependencies.clock()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("pipeline clock must return a real number")
    current = float(value)
    if not math.isfinite(current):
        raise ValueError("pipeline clock must return a finite value")
    if previous is not None and current < previous:
        raise ValueError("pipeline clock must be monotonic")
    return current
