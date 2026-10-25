from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roadpush_ai import legacy_policy as legacy


@dataclass
class SharedPreprocessing:
    image: np.ndarray
    labels: np.ndarray
    repaired: np.ndarray
    structure_copy: np.ndarray
    structure_copy_additions: np.ndarray
    curb_suspect: np.ndarray
    route: dict
    route_path: list[tuple[int, int]]
    source_walkway: np.ndarray
    observed_component: np.ndarray


def load_cached_inputs(item: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load the cached Mask2Former map and its source image at V2 resolution."""
    return legacy.integrated.load_inputs(item)


def build_shared_preprocessing(image: np.ndarray, labels: np.ndarray) -> SharedPreprocessing:
    repaired, source_walkway = legacy.integrated.selected_walkway(labels)
    _, curb_suspect, _, _ = legacy.curb_probe.analyze(labels)

    structure, additions = legacy.base.structure_copy(repaired)
    route_surface = repaired & ~curb_suspect
    if not route_surface.any():
        route_surface = repaired.copy()
    route, path = legacy.outer.route_path(route_surface)

    visible_surface = structure & ~curb_suspect
    if not visible_surface.any():
        visible_surface = structure.copy()
    observed_component = legacy.outer.selected_component(visible_surface, path)
    if not observed_component.any():
        observed_component = visible_surface.copy()

    return SharedPreprocessing(
        image=image,
        labels=labels,
        repaired=repaired,
        structure_copy=structure,
        structure_copy_additions=additions,
        curb_suspect=curb_suspect,
        route=route,
        route_path=path,
        source_walkway=source_walkway,
        observed_component=observed_component,
    )

