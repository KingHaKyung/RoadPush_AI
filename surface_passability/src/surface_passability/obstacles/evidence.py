"""Image-level obstacle evidence assembled before surface ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Sequence

import numpy as np

from surface_passability.contracts.obstacles import (
    ObstacleGeometry,
    ObstacleObjectGeometry,
    ObstacleOwnershipResult,
    ObstacleSourceEvidence,
    SemanticObstacleComponent,
    SurfaceVehicleComponentEvidence,
    SurfaceVehicleProjection,
)
from surface_passability.contracts.surfaces import ObstacleSurfaceInput
from surface_passability.inference.vehicle_instances import (
    VehicleModelProvider,
    infer_vehicle_instances,
    match_vehicle_instances,
    partition_mapillary_component,
)
from surface_passability.obstacles.components import (
    extract_semantic_obstacle_components,
)
from surface_passability.obstacles.footprints import estimate_component_footprint
from surface_passability.obstacles.geometry import (
    extract_lower_silhouette,
    measure_lower_axis_geometry,
)
from surface_passability.obstacles.vehicle_trigger import (
    analyze_vehicle_trigger_components,
)
from surface_passability.obstacles.taxonomy import (
    BANNER_CLASS_ID,
    BANNER_SUPPORT_CLASS_IDS,
    CAR_CLASS_ID,
    HAZARD_CLASS_IDS,
    NATURAL_NON_WALKABLE_CLASS_IDS,
    OCCUPANCY_CLASS_IDS,
    TRANSIENT_SOURCE_CLASS_IDS,
    WHEELED_OBSTACLE_CLASS_IDS,
)


def _record_bounding_box(record: Mapping[str, object]) -> object:
    """Select the first truthy frozen source-box field."""

    return (
        record.get("bbox")
        or record.get("source_bbox")
        or record.get("component_bbox")
        or record.get("intrusion_bbox")
    )


def reconstruct_obstacle_source_mask(
    record: Mapping[str, object],
    class_map: np.ndarray,
) -> np.ndarray:
    """Reconstruct one record's semantic source inside its reported bounds."""

    mask = np.zeros_like(class_map, dtype=bool)
    box = _record_bounding_box(record)
    if not box:
        return mask
    x, y, width, height = map(int, box)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(class_map.shape[1], x + width)
    y1 = min(class_map.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return mask
    class_id = record.get("class_id")
    if class_id is not None:
        mask[y0:y1, x0:x1] |= (
            class_map[y0:y1, x0:x1] == int(class_id)
        )
    if record.get("kind") == "grounded_banner_support_cluster":
        class_ids = {
            BANNER_CLASS_ID,
            *BANNER_SUPPORT_CLASS_IDS,
        }
        mask[y0:y1, x0:x1] |= np.isin(
            class_map[y0:y1, x0:x1],
            tuple(sorted(class_ids)),
        )
    if record.get("source") == "class_agnostic_corridor_hole":
        class_ids = [int(key) for key in record.get("class_mix", {})]
        if class_ids:
            mask[y0:y1, x0:x1] |= np.isin(
                class_map[y0:y1, x0:x1],
                class_ids,
            )
    return mask


def collect_obstacle_source_evidence(
    ownership: ObstacleOwnershipResult,
    class_map: np.ndarray,
) -> tuple[ObstacleSourceEvidence, ...]:
    """Collect accepted and transient sources in original ownership order."""

    sources: list[ObstacleSourceEvidence] = []
    for index, typed_record in enumerate(ownership.records):
        record = typed_record.to_legacy_record()
        usable = bool(record.get("accepted")) or (
            record.get("kind") == "transient_occluder"
        )
        if not usable:
            continue
        mask = reconstruct_obstacle_source_mask(record, class_map)
        if not mask.any():
            continue
        sources.append(
            ObstacleSourceEvidence(
                record_index=index,
                mask=mask,
                area=int(mask.sum()),
                reported_bbox=record.get("bbox"),
                class_name=record.get("class_name"),
                kind=record.get("kind"),
                strategy=record.get("strategy"),
            )
        )
    return tuple(sources)


def enrich_obstacle_object_geometry(
    component: SemanticObstacleComponent,
    *,
    source_record: Mapping[str, object] | None = None,
) -> ObstacleObjectGeometry:
    """Attach reusable source geometry to one semantic obstacle component."""

    source_mask = component.mask
    source_record_items = tuple(
        (
            component.to_legacy_record()
            if source_record is None
            else source_record
        ).items()
    )
    lower_silhouette = extract_lower_silhouette(source_mask)
    axis_geometry = measure_lower_axis_geometry(source_mask)
    if component.class_id in WHEELED_OBSTACLE_CLASS_IDS:
        footprint = estimate_component_footprint(
            source_mask,
            component.class_id,
        )
        candidate_mask = (
            np.zeros_like(source_mask)
            if footprint is None
            else footprint.mask
        )
        return ObstacleObjectGeometry(
            component=component,
            source_record_items=source_record_items,
            lower_silhouette_candidate=lower_silhouette,
            axis_geometry=axis_geometry,
            footprint_evaluated=True,
            footprint_candidate_mask=candidate_mask,
            footprint=footprint,
        )
    return ObstacleObjectGeometry(
        component=component,
        source_record_items=source_record_items,
        lower_silhouette_candidate=lower_silhouette,
        axis_geometry=axis_geometry,
        footprint_evaluated=False,
    )


def extract_obstacle_geometry(
    image: np.ndarray,
    class_map: np.ndarray,
    class_names: Mapping[int | str, str],
) -> ObstacleGeometry:
    """Extract image-level obstacle identities and source geometry once."""

    id2label = {int(key): value for key, value in class_names.items()}
    all_objects = extract_semantic_obstacle_components(
        class_map,
        OCCUPANCY_CLASS_IDS,
    )
    active = tuple(
        component
        for component in all_objects
        if (
            component.class_id not in TRANSIENT_SOURCE_CLASS_IDS
            and component.class_id != BANNER_CLASS_ID
        )
    )
    return ObstacleGeometry(
        image=image,
        labels=class_map,
        all_objects=all_objects,
        active_objects=tuple(
            enrich_obstacle_object_geometry(component)
            for component in active
        ),
        banners=extract_semantic_obstacle_components(
            class_map,
            (BANNER_CLASS_ID,),
        ),
        transient_source=np.isin(
            class_map,
            tuple(sorted(TRANSIENT_SOURCE_CLASS_IDS)),
        ),
        natural_source_evidence=np.isin(
            class_map,
            tuple(sorted(NATURAL_NON_WALKABLE_CLASS_IDS)),
        ),
        hazard_ids=frozenset(HAZARD_CLASS_IDS),
        id2label=id2label,
        surface_trigger_records={},
    )


def prepare_obstacle_geometry(
    geometry: ObstacleGeometry,
    surfaces: Sequence[ObstacleSurfaceInput],
    model_provider: VehicleModelProvider,
) -> ObstacleGeometry:
    """Collect surface triggers and run at most one lazy vehicle inference.

    The provider owns model configuration and cross-image reuse. It is never
    consulted when no surface requests instance inference and is called once
    at most for one prepared image.
    """

    trigger_records = {
        surface.name: analyze_vehicle_trigger_components(
            geometry.labels,
            surface.source_mask,
        )
        for surface in surfaces
    }
    requires_inference = any(
        evidence.requires_yolo
        for evidence in trigger_records.values()
    )
    detections = ()
    inference = None
    if requires_inference:
        result = infer_vehicle_instances(
            model_provider.get_model(),
            geometry.image,
        )
        detections = result.instances
        inference = result.diagnostics
    return replace(
        geometry,
        yolo_detections=detections,
        yolo_inference=inference,
        surface_trigger_records=trigger_records,
    )


def project_surface_vehicle_instances(
    geometry: ObstacleGeometry,
    surface: ObstacleSurfaceInput,
) -> SurfaceVehicleProjection:
    """Project shared vehicle detections into one surface-specific object view."""

    trigger_evidence = geometry.surface_trigger_records.get(surface.name)
    trigger_records = (
        ()
        if trigger_evidence is None
        else trigger_evidence.components
    )
    triggered = tuple(
        record
        for record in trigger_records
        if record.trigger_yolo
    )
    if not triggered:
        return SurfaceVehicleProjection(
            active_objects=geometry.active_objects,
            triggered=False,
            trigger_component_count=0,
            inference_run=False,
            result="not_triggered",
            split_confirmed_count=0,
            fallback_parent_count=0,
            components=(),
        )

    triggered_masks = tuple(record.component_mask for record in triggered)
    kept = [
        obj
        for obj in geometry.active_objects
        if not (
            obj.component.class_id == CAR_CLASS_ID
            and any(
                np.array_equal(obj.component.mask, parent)
                for parent in triggered_masks
            )
        )
    ]
    component_evidence: list[SurfaceVehicleComponentEvidence] = []
    split_confirmed_count = 0
    fallback_parent_count = 0
    for trigger_record in triggered:
        parent = trigger_record.component_mask
        matches = match_vehicle_instances(parent, geometry.yolo_detections)
        split = (
            len(matches.accepted) >= 2
            and all(instance.mask.any() for instance in matches.accepted)
        )
        component_evidence.append(SurfaceVehicleComponentEvidence(
            source_bbox=trigger_record.source_bbox,
            matches=matches,
            split_confirmed=bool(split),
        ))
        if split:
            split_confirmed_count += 1
            children = partition_mapillary_component(
                parent,
                matches.accepted,
            ).children
            for child_index, (child, instance) in enumerate(
                zip(children, matches.accepted)
            ):
                source_record = {
                    "class_id": CAR_CLASS_ID,
                    "component_index": trigger_record.component_id,
                    "bbox": list(trigger_record.source_bbox),
                    "source_pixels": int(child.sum()),
                    "instance_source": "yolo_partitioned_mapillary",
                    "instance_index": child_index,
                    "yolo_detection_index": instance.detection_index,
                    "yolo_class": instance.class_name,
                    "yolo_confidence": instance.confidence,
                    "_mask": child,
                }
                kept.append(enrich_obstacle_object_geometry(
                    SemanticObstacleComponent(
                        class_id=CAR_CLASS_ID,
                        component_index=trigger_record.component_id,
                        bbox=trigger_record.source_bbox,
                        source_pixels=int(child.sum()),
                        mask=child,
                    ),
                    source_record=source_record,
                ))
        else:
            fallback_parent_count += 1
            source_record = {
                "class_id": CAR_CLASS_ID,
                "component_index": trigger_record.component_id,
                "bbox": list(trigger_record.source_bbox),
                "source_pixels": int(parent.sum()),
                "instance_source": "yolo_inconclusive_parent_fallback",
                "_mask": parent,
            }
            kept.append(enrich_obstacle_object_geometry(
                SemanticObstacleComponent(
                    class_id=CAR_CLASS_ID,
                    component_index=trigger_record.component_id,
                    bbox=trigger_record.source_bbox,
                    source_pixels=int(parent.sum()),
                    mask=parent,
                ),
                source_record=source_record,
            ))
    return SurfaceVehicleProjection(
        active_objects=tuple(kept),
        triggered=True,
        trigger_component_count=len(triggered),
        inference_run=True,
        inference=geometry.yolo_inference,
        result=(
            "split_confirmed"
            if split_confirmed_count
            else "fallback_parent"
        ),
        split_confirmed_count=split_confirmed_count,
        fallback_parent_count=fallback_parent_count,
        components=tuple(component_evidence),
    )
