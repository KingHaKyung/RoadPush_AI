"""Composition of approved obstacle evidence into one surface result."""

from __future__ import annotations

import numpy as np

from surface_passability.contracts.obstacles import (
    ObstacleGeometry,
    ObstacleOwnershipPolicy,
    ObstacleOwnershipResult,
)
from surface_passability.contracts.surfaces import (
    ObstacleOwnershipSurfaceInput,
    ObstacleSurfaceInput,
)
from surface_passability.obstacles.evidence import (
    project_surface_vehicle_instances,
)
from surface_passability.obstacles.ownership import (
    collect_semantic_hazard_ownership_records,
    evaluate_banner_support_ownership,
    evaluate_grounded_obstacle_ownership,
    evaluate_wheeled_obstacle_ownership,
    project_natural_hole_ownership,
    project_transient_occluder_ownership,
)
from surface_passability.obstacles.taxonomy import (
    WHEELED_OBSTACLE_CLASS_IDS,
)


def evaluate_obstacle_ownership(
    geometry: ObstacleGeometry,
    surface: ObstacleOwnershipSurfaceInput,
    *,
    wheeled_silhouette_angle_max_degrees: float | None = None,
    banner_support_group_mode: str = "nearest",
) -> ObstacleOwnershipResult:
    """Evaluate the frozen complete obstacle policy for one surface."""

    labels = geometry.labels
    vehicle_projection = project_surface_vehicle_instances(
        geometry,
        ObstacleSurfaceInput(surface.name, surface.source_mask),
    )
    hazard_ids = tuple(sorted(geometry.hazard_ids))
    recognized = geometry.transient_source | np.isin(labels, hazard_ids)
    final = np.isin(labels, tuple(sorted(geometry.hazard_ids)))
    footprint_selected = np.zeros_like(labels, dtype=bool)
    silhouette_selected = np.zeros_like(labels, dtype=bool)
    wheeled_silhouette_selected = np.zeros_like(labels, dtype=bool)
    records = []
    passability_instances = []

    banner = evaluate_banner_support_ownership(
        geometry,
        vehicle_projection.active_objects,
        surface.source_mask,
        group_mode=banner_support_group_mode,
    )
    recognized |= banner.recognized_mask
    final |= banner.selected_mask
    silhouette_selected |= banner.silhouette_selected_mask
    records.extend(banner.records)
    passability_instances.extend(banner.passability_instances)

    baseline_for_natural = np.zeros_like(labels, dtype=bool)
    grouped_support_ids = banner.grouped_support_ids
    for obj in vehicle_projection.active_objects:
        component = obj.component
        if (component.class_id, component.component_index) in grouped_support_ids:
            continue
        if component.class_id in WHEELED_OBSTACLE_CLASS_IDS:
            contribution = evaluate_wheeled_obstacle_ownership(
                geometry,
                (obj,),
                surface.source_mask,
                grouped_support_ids,
                silhouette_angle_max_degrees=(
                    wheeled_silhouette_angle_max_degrees
                ),
            )
            evidence = contribution.records[0]
            recognized |= contribution.recognized_mask
            final |= contribution.selected_mask
            footprint_selected |= contribution.footprint_selected_mask
            silhouette_selected |= contribution.silhouette_selected_mask
            wheeled_silhouette_selected |= (
                contribution.wheeled_silhouette_selected_mask
            )
            baseline_for_natural |= contribution.baseline_for_natural_mask
            records.append(evidence)
            instance = evidence.to_passability_instance()
        else:
            contribution = evaluate_grounded_obstacle_ownership(
                geometry,
                (obj,),
                surface.source_mask,
                grouped_support_ids,
            )
            evidence = contribution.records[0]
            recognized |= contribution.recognized_mask
            final |= contribution.selected_mask
            silhouette_selected |= contribution.silhouette_selected_mask
            baseline_for_natural |= contribution.baseline_for_natural_mask
            records.append(evidence)
            instance = evidence.to_passability_instance()
        if instance is not None:
            passability_instances.append(instance)

    natural = project_natural_hole_ownership(
        geometry,
        surface.repaired_mask,
        surface.structural_mask,
        surface.source_mask,
        baseline_for_natural,
    )
    recognized |= natural.recognized_mask
    final |= natural.selected_mask
    footprint_selected |= natural.footprint_selected_mask
    records.extend(natural.records)

    transient = project_transient_occluder_ownership(geometry)
    recognized |= transient.recognized_mask
    records.extend(transient.records)

    hazard_recognized, hazard_records, hazard_instances = (
        collect_semantic_hazard_ownership_records(geometry)
    )
    recognized |= hazard_recognized
    records.extend(hazard_records)
    passability_instances.extend(hazard_instances)

    overlap_pixels = int((footprint_selected & silhouette_selected).sum())
    violations = []
    if any(
        record.to_legacy_record().get("kind") == "cycle_gap_closure"
        for record in records
    ):
        violations.append("forbidden_global_cycle_closure")
    if any(
        record.to_legacy_record().get("natural_restoration_applied")
        for record in records
    ):
        violations.append("natural_restoration_leak")
    return ObstacleOwnershipResult(
        recognized_mask=recognized,
        transient_mask=geometry.transient_source,
        hazard_mask=np.isin(labels, tuple(sorted(geometry.hazard_ids))),
        footprint_mask=footprint_selected,
        silhouette_mask=silhouette_selected,
        wheeled_silhouette_mask=wheeled_silhouette_selected,
        final_mask=final,
        records=tuple(records),
        passability_instances=tuple(passability_instances),
        vehicle_projection=vehicle_projection,
        violations=tuple(violations),
        strategy_overlap_pixels=overlap_pixels,
        policy=ObstacleOwnershipPolicy(
            wheeled_silhouette_angle_max_degrees=(
                wheeled_silhouette_angle_max_degrees
            ),
            banner_support_group_mode=banner_support_group_mode,
        ),
    )
