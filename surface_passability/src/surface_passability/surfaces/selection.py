"""Compose obstacle-source evidence with surface-component selection."""

from __future__ import annotations

import numpy as np

from surface_passability.contracts.obstacles import ObstacleOwnershipResult
from surface_passability.contracts.surfaces import (
    SurfaceComponentSelection,
    SurfaceComponentSelectionInput,
)
from surface_passability.obstacles.evidence import (
    collect_obstacle_source_evidence,
)
from surface_passability.surfaces.components import (
    select_obstacle_connected_surface_components,
)


def select_surface_components_from_obstacle_ownership(
    ownership: ObstacleOwnershipResult,
    class_map: np.ndarray,
    surface: SurfaceComponentSelectionInput,
    *,
    authorization_forbidden_mask: np.ndarray | None = None,
) -> SurfaceComponentSelection:
    """Collect source evidence and select the connected surface components."""

    sources = collect_obstacle_source_evidence(ownership, class_map)
    if authorization_forbidden_mask is None:
        return select_obstacle_connected_surface_components(surface, sources)
    return select_obstacle_connected_surface_components(
        surface,
        sources,
        authorization_forbidden_mask=authorization_forbidden_mask,
    )
