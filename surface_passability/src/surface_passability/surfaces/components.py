"""Connected surface components used by candidate selection."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from surface_passability.contracts.surfaces import (
    AcceptedSurfaceSatellite,
    ConnectedSurfaceComponent,
    ObstacleComponentContactMatch,
    SurfaceComponentSelection,
    SurfaceComponentSelectionInput,
    SurfaceForwardAlignment,
)
from surface_passability.contracts.obstacles import ObstacleSourceEvidence
from surface_passability.policies.surfaces import (
    SURFACE_COMPONENT_CONTACT_RADIUS_IMAGE_FRACTION,
    SURFACE_COMPONENT_MINIMUM_CONTACT_PIXELS_EACH_SIDE,
    SURFACE_COMPONENT_MINIMUM_FORWARD_PROJECTION_PX,
    SURFACE_COMPONENT_MINIMUM_PIXELS,
    SURFACE_COMPONENT_MINIMUM_ROWS,
    SURFACE_COMPONENT_MINIMUM_ROUTE_ALIGNMENT_COSINE,
)
from surface_passability.surfaces.row_completion import (
    build_surface_span_mask,
    measure_surface_row_spans,
)


def extract_connected_surface_components(
    mask: np.ndarray,
    minimum_pixels: int = 1,
) -> tuple[ConnectedSurfaceComponent, ...]:
    """Return qualifying eight-connected components in OpenCV label order."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        8,
    )
    components: list[ConnectedSurfaceComponent] = []
    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < minimum_pixels:
            continue
        components.append(
            ConnectedSurfaceComponent(
                component_index=int(component_index),
                mask=labels == component_index,
                area=area,
                bbox=tuple(
                    int(value)
                    for value in stats[component_index, :4]
                ),
            )
        )
    return tuple(components)


def select_main_surface_component(
    components: Sequence[ConnectedSurfaceComponent],
    route_start: tuple[int, int] | None,
    observed_component: np.ndarray,
) -> ConnectedSurfaceComponent:
    """Choose the route-containing component, then overlap and area."""

    if not components:
        raise RuntimeError("No structure-copy components")
    ranked: list[
        tuple[bool, int, int, ConnectedSurfaceComponent]
    ] = []
    for component in components:
        overlap = int((component.mask & observed_component).sum())
        contains_start = bool(
            route_start is not None and component.mask[route_start]
        )
        ranked.append(
            (
                contains_start,
                overlap,
                component.area,
                component,
            )
        )
    return max(ranked, key=lambda value: value[:3])[3]


def evaluate_surface_forward_alignment(
    route_path: Sequence[tuple[int, int]],
    selected_contact: np.ndarray,
    satellite_contact: np.ndarray,
) -> SurfaceForwardAlignment:
    """Measure whether a satellite contact lies forward along the route."""

    if len(route_path) < 2:
        return SurfaceForwardAlignment(available=False, accepted=False)
    start_y, start_x = route_path[0]
    end_y, end_x = route_path[-1]
    route_vector = np.asarray(
        [end_x - start_x, end_y - start_y],
        np.float64,
    )
    selected_yx = np.argwhere(selected_contact).mean(axis=0)
    satellite_yx = np.argwhere(satellite_contact).mean(axis=0)
    contact_vector = np.asarray(
        [
            satellite_yx[1] - selected_yx[1],
            satellite_yx[0] - selected_yx[0],
        ],
        np.float64,
    )
    route_norm = float(np.linalg.norm(route_vector))
    contact_norm = float(np.linalg.norm(contact_vector))
    if route_norm < 1e-6 or contact_norm < 1e-6:
        return SurfaceForwardAlignment(
            available=True,
            accepted=False,
            forward_projection_px=0.0,
            cosine=None,
        )
    unit_route = route_vector / route_norm
    forward = float(np.dot(contact_vector, unit_route))
    cosine = float(forward / contact_norm)
    return SurfaceForwardAlignment(
        available=True,
        accepted=bool(
            forward >= SURFACE_COMPONENT_MINIMUM_FORWARD_PROJECTION_PX
            and cosine >= SURFACE_COMPONENT_MINIMUM_ROUTE_ALIGNMENT_COSINE
        ),
        forward_projection_px=forward,
        cosine=cosine,
        selected_contact_center_yx=(
            float(selected_yx[0]),
            float(selected_yx[1]),
        ),
        satellite_contact_center_yx=(
            float(satellite_yx[0]),
            float(satellite_yx[1]),
        ),
    )


def surface_connection_crosses_forbidden(
    first_contact: np.ndarray,
    second_contact: np.ndarray,
    forbidden_mask: np.ndarray,
    thickness: int,
) -> bool:
    """Return whether the contact-center connection crosses forbidden pixels."""

    first_center = np.argwhere(first_contact).mean(axis=0)
    second_center = np.argwhere(second_contact).mean(axis=0)
    line = np.zeros_like(forbidden_mask, np.uint8)
    cv2.line(
        line,
        (round(first_center[1]), round(first_center[0])),
        (round(second_center[1]), round(second_center[0])),
        1,
        max(1, thickness),
    )
    return bool((line.astype(bool) & forbidden_mask).any())


def select_obstacle_connected_surface_components(
    surface: SurfaceComponentSelectionInput,
    obstacle_sources: Sequence[ObstacleSourceEvidence],
    *,
    authorization_forbidden_mask: np.ndarray | None = None,
) -> SurfaceComponentSelection:
    """Select surface satellites connected through the same obstacle source."""

    candidates = surface.structural_mask & ~surface.forbidden_mask
    components = extract_connected_surface_components(candidates)
    radius = max(
        3,
        round(
            min(candidates.shape)
            * SURFACE_COMPONENT_CONTACT_RADIUS_IMAGE_FRACTION
        ),
    )
    if not components:
        empty = np.zeros_like(candidates)
        return SurfaceComponentSelection(
            candidate_mask=candidates,
            main_component_mask=empty.copy(),
            selected_source_mask=empty.copy(),
            accepted_satellite_mask=empty.copy(),
            rejected_source_mask=empty.copy(),
            span_mask=empty.copy(),
            components=(),
            main_component_index=None,
            accepted_component_indices=(),
            evidence=(),
            obstacle_component_count=0,
            contact_radius_px=radius,
            empty_candidate=True,
            forbidden_crossing_rejections=(
                0 if authorization_forbidden_mask is not None else None
            ),
        )

    main = select_main_surface_component(
        components,
        surface.route_start,
        surface.observed_component,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    obstacle_dilated = [
        cv2.dilate(source.mask.astype(np.uint8), kernel).astype(bool)
        for source in obstacle_sources
    ]
    selected = main.mask.copy()
    accepted = {main.component_index}
    evidence: list[AcceptedSurfaceSatellite] = []
    crossing_rejections = 0
    changed = True
    while changed:
        changed = False
        for component in components:
            if component.component_index in accepted:
                continue
            ys = np.nonzero(component.mask)[0]
            if (
                component.area < SURFACE_COMPONENT_MINIMUM_PIXELS
                or len(np.unique(ys)) < SURFACE_COMPONENT_MINIMUM_ROWS
            ):
                continue
            matches: list[ObstacleComponentContactMatch] = []
            for obstacle, dilated in zip(
                obstacle_sources,
                obstacle_dilated,
            ):
                selected_contact = dilated & selected
                satellite_contact = dilated & component.mask
                selected_pixels = int(selected_contact.sum())
                satellite_pixels = int(satellite_contact.sum())
                if (
                    selected_pixels
                    < SURFACE_COMPONENT_MINIMUM_CONTACT_PIXELS_EACH_SIDE
                    or satellite_pixels
                    < SURFACE_COMPONENT_MINIMUM_CONTACT_PIXELS_EACH_SIDE
                ):
                    continue
                alignment = evaluate_surface_forward_alignment(
                    surface.route_path,
                    selected_contact,
                    satellite_contact,
                )
                if not alignment.accepted:
                    continue
                if (
                    authorization_forbidden_mask is not None
                    and surface_connection_crosses_forbidden(
                        selected_contact,
                        satellite_contact,
                        authorization_forbidden_mask,
                        radius,
                    )
                ):
                    crossing_rejections += 1
                    continue
                matches.append(
                    ObstacleComponentContactMatch(
                        obstacle_record_index=obstacle.record_index,
                        obstacle_class_name=obstacle.class_name,
                        obstacle_kind=obstacle.kind,
                        obstacle_strategy=obstacle.strategy,
                        obstacle_area=obstacle.area,
                        obstacle_bbox=obstacle.reported_bbox,
                        selected_contact_pixels=selected_pixels,
                        satellite_contact_pixels=satellite_pixels,
                        forward_alignment=alignment,
                    )
                )
            if not matches:
                continue
            selected |= component.mask
            accepted.add(component.component_index)
            evidence.append(
                AcceptedSurfaceSatellite(
                    component_index=component.component_index,
                    area=component.area,
                    bbox=component.bbox,
                    obstacle_matches=tuple(matches),
                )
            )
            changed = True
    rejected = candidates & ~selected
    spans = measure_surface_row_spans(selected)
    span = build_surface_span_mask(spans, candidates.shape)
    span &= ~surface.forbidden_mask
    return SurfaceComponentSelection(
        candidate_mask=candidates,
        main_component_mask=main.mask,
        selected_source_mask=selected,
        accepted_satellite_mask=selected & ~main.mask,
        rejected_source_mask=rejected,
        span_mask=span,
        components=components,
        main_component_index=main.component_index,
        accepted_component_indices=tuple(sorted(accepted)),
        evidence=tuple(evidence),
        obstacle_component_count=len(obstacle_sources),
        contact_radius_px=radius,
        empty_candidate=False,
        forbidden_crossing_rejections=(
            crossing_rejections
            if authorization_forbidden_mask is not None
            else None
        ),
    )
