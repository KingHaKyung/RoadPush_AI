"""Sidewalk boundary construction from the prepared surface context."""

from __future__ import annotations

import numpy as np

from surface_passability.contracts.boundaries import (
    SidewalkBoundaryMode,
    SidewalkBoundaryResult,
)
from surface_passability.contracts.surfaces import SurfacePipelineContext
from surface_passability.obstacles.taxonomy import (
    HAZARD_CLASS_IDS,
    OCCUPANCY_CLASS_IDS,
    TRANSIENT_OCCLUSION_CLASS_IDS,
)
from surface_passability.surfaces.row_completion import (
    build_surface_span_mask,
    measure_surface_row_spans,
)
from surface_passability.surfaces.routes import select_routed_surface_component
from surface_passability.surfaces.structure import (
    build_obstacle_transparent_structure,
)


def _initial_diagnostics() -> dict[str, bool]:
    """Return diagnostics in the frozen backend serialization order."""

    return {
        "repair_surface": True,
        "structure_copy": True,
        "route_component_selection_before_completion": False,
        "v2_horizontal_obstacle_bridge": False,
        "v2_1_natural_restoration": False,
        "a1_corridor_hole_restoration": False,
        "a2_row_span_completion": False,
        "all_candidate_outer_row_span": False,
    }


def construct_sidewalk_boundary(
    context: SurfacePipelineContext,
    mode: SidewalkBoundaryMode,
) -> SidewalkBoundaryResult:
    """Construct one frozen sidewalk-boundary variant without mutating context."""

    empty = np.zeros_like(context.class_map, dtype=bool)
    calls = _initial_diagnostics()

    if mode == SidewalkBoundaryMode.OBSERVED_COMPONENT:
        return SidewalkBoundaryResult(
            context.observed_component.copy(),
            empty,
            mode,
            calls,
        )

    if mode == SidewalkBoundaryMode.OBSTACLE_TRANSPARENT_COMPONENT:
        blocked_class_ids = (
            tuple(OCCUPANCY_CLASS_IDS)
            + tuple(TRANSIENT_OCCLUSION_CLASS_IDS)
            + tuple(HAZARD_CLASS_IDS)
        )
        raw_blocked = np.isin(context.class_map, blocked_class_ids)
        bridge = build_obstacle_transparent_structure(
            context.repaired,
            raw_blocked,
        )
        bridged = bridge.structural_mask
        bridged &= ~context.curb_suspect
        component = select_routed_surface_component(
            bridged,
            context.route_path,
        )
        calls["v2_horizontal_obstacle_bridge"] = True
        calls["route_component_selection_before_completion"] = True
        final = component if component.any() else bridged
        return SidewalkBoundaryResult(
            final,
            final & ~context.observed_component,
            mode,
            calls,
        )

    if mode == SidewalkBoundaryMode.ALL_CANDIDATE_ROW_SPAN:
        candidates = context.structure_copy & ~context.curb_suspect
        spans = measure_surface_row_spans(candidates)
        span_mask = build_surface_span_mask(spans, context.class_map.shape)
        span_mask &= ~context.curb_suspect
        calls["a2_row_span_completion"] = True
        calls["all_candidate_outer_row_span"] = True
        return SidewalkBoundaryResult(
            span_mask,
            span_mask & ~context.observed_component,
            mode,
            calls,
        )

    raise ValueError(f"Unsupported boundary mode: {mode}")
