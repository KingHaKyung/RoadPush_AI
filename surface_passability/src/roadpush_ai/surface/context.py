from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roadpush_ai import legacy_policy as legacy


@dataclass(frozen=True)
class SurfaceProfile:
    """Behavioral identity for one surface without changing frozen policy code."""

    name: str
    source_class_ids: tuple[int, ...]
    repair_class_ids: tuple[int, ...]
    forbidden_class_ids: tuple[int, ...]
    side_pruning_enabled: bool


@dataclass
class SurfaceContext:
    """Surface-local state produced before obstacle and boundary processing."""

    profile: SurfaceProfile
    source_mask: np.ndarray
    repaired_mask: np.ndarray
    structure_copy: np.ndarray
    forbidden_mask: np.ndarray
    observed_component: np.ndarray
    route: dict
    route_path: list[tuple[int, int]]
    structure_copy_additions: np.ndarray


SIDEWALK_PROFILE = SurfaceProfile(
    name="sidewalk",
    source_class_ids=(15,),
    # The frozen V2 repair and curb-suspect algorithms are policy functions,
    # not simple class-id unions. Empty tuples avoid misrepresenting them as
    # class-local rules while Phase 1 delegates to those functions verbatim.
    repair_class_ids=(),
    forbidden_class_ids=(),
    side_pruning_enabled=True,
)


@dataclass
class SurfacePipelineContext:
    """Image-level data plus the single Phase-1 sidewalk surface context.

    The compatibility properties are intentionally read-only. They let the
    frozen Test4 functions consume this envelope without editing their
    source-hashed implementations, while all sidewalk-owned arrays live in
    ``surface``.
    """

    image: np.ndarray
    labels: np.ndarray
    surface: SurfaceContext

    @property
    def repaired(self) -> np.ndarray:
        return self.surface.repaired_mask

    @property
    def structure_copy(self) -> np.ndarray:
        return self.surface.structure_copy

    @property
    def structure_copy_additions(self) -> np.ndarray:
        return self.surface.structure_copy_additions

    @property
    def curb_suspect(self) -> np.ndarray:
        return self.surface.forbidden_mask

    @property
    def route(self) -> dict:
        return self.surface.route

    @property
    def route_path(self) -> list[tuple[int, int]]:
        return self.surface.route_path

    @property
    def source_walkway(self) -> np.ndarray:
        return self.surface.source_mask

    @property
    def observed_component(self) -> np.ndarray:
        return self.surface.observed_component


def build_sidewalk_context(labels: np.ndarray) -> SurfaceContext:
    """Run the frozen sidewalk preprocessing order into a SurfaceContext."""

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

    return SurfaceContext(
        profile=SIDEWALK_PROFILE,
        source_mask=source_walkway,
        repaired_mask=repaired,
        structure_copy=structure,
        forbidden_mask=curb_suspect,
        observed_component=observed_component,
        route=route,
        route_path=path,
        structure_copy_additions=additions,
    )


def build_surface_pipeline_context(
    image: np.ndarray,
    labels: np.ndarray,
) -> SurfacePipelineContext:
    """Build the generalized envelope with sidewalk as its only current surface."""

    return SurfacePipelineContext(
        image=image,
        labels=labels,
        surface=build_sidewalk_context(labels),
    )


def build_shared_preprocessing(
    image: np.ndarray,
    labels: np.ndarray,
) -> SurfacePipelineContext:
    """Phase-1 compatibility entry point; new code uses the explicit name."""

    return build_surface_pipeline_context(image, labels)
