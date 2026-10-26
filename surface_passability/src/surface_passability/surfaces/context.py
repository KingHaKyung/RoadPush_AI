"""Typed semantic-surface context construction."""

from __future__ import annotations

import numpy as np

from surface_passability.contracts.surfaces import (
    SurfaceContext,
    SurfacePipelineContext,
)
from surface_passability.policies.surfaces import SIDEWALK_SURFACE_PROFILE
from surface_passability.surfaces.curb_evidence import extract_curb_evidence
from surface_passability.surfaces.routes import (
    select_routed_surface_component,
    solve_surface_route,
)
from surface_passability.surfaces.sidewalk import repair_sidewalk_surface
from surface_passability.surfaces.structure import build_surface_structure


def build_sidewalk_context(class_map: np.ndarray) -> SurfaceContext:
    """Build the frozen pre-obstacle Sidewalk state in production order."""

    repair = repair_sidewalk_surface(class_map)
    source_route = solve_surface_route(repair.repaired_mask)
    source_component = select_routed_surface_component(
        repair.repaired_mask,
        source_route.main_path,
    )
    source_mask = (
        source_component
        if source_component.any()
        else repair.repaired_mask
    )

    curb = extract_curb_evidence(class_map)
    structure = build_surface_structure(repair.repaired_mask)

    route_surface = repair.repaired_mask & ~curb.suspect_mask
    if not route_surface.any():
        route_surface = repair.repaired_mask.copy()
    route = solve_surface_route(route_surface)

    visible_surface = structure.structural_mask & ~curb.suspect_mask
    if not visible_surface.any():
        visible_surface = structure.structural_mask.copy()
    observed_component = select_routed_surface_component(
        visible_surface,
        route.main_path,
    )
    if not observed_component.any():
        observed_component = visible_surface.copy()

    return SurfaceContext(
        profile=SIDEWALK_SURFACE_PROFILE,
        source_mask=source_mask,
        repaired_mask=repair.repaired_mask,
        structural_mask=structure.structural_mask,
        forbidden_mask=curb.suspect_mask,
        observed_component=observed_component,
        route=route,
        route_path=route.main_path,
        structural_addition_mask=structure.addition_mask,
    )


def build_surface_pipeline_context(
    image: np.ndarray,
    class_map: np.ndarray,
) -> SurfacePipelineContext:
    """Pair exact image evidence with one exact Sidewalk context object."""

    surface = build_sidewalk_context(class_map)
    return SurfacePipelineContext(
        image=image,
        class_map=class_map,
        surface=surface,
    )
