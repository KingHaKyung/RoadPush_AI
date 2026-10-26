"""Surface-contact evidence for obstacle ownership."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.obstacles import (
    BannerSupportClusterEvidence,
    BannerSupportMember,
    BannerSupportOwnership,
    GroundedObstacleOwnership,
    GroundedObstacleOwnershipEvidence,
    NaturalHoleOwnership,
    NaturalHoleOwnershipRecord,
    ObstacleFootprint,
    ObstacleGeometry,
    ObstacleObjectGeometry,
    OwnedObstacleFootprint,
    PassabilityObstacleInstance,
    ProjectedGroundSupport,
    SourceContactSupport,
    StandaloneBannerEvidence,
    SemanticHazardEvidence,
    SemanticHazardOwnership,
    TransientOccluderEvidence,
    TransientOccluderOwnership,
    WheeledObstacleEvidence,
    WheeledObstacleOwnership,
)
from surface_passability.obstacles.geometry import (
    evaluate_grounded_silhouette,
    extract_lower_silhouette,
)
from surface_passability.obstacles.components import (
    extract_semantic_obstacle_components,
)
from surface_passability.obstacles.natural import collect_natural_hole_evidence
from surface_passability.obstacles.taxonomy import (
    BANNER_CLASS_ID,
    BANNER_SUPPORT_CLASS_IDS,
    TRANSIENT_SOURCE_CLASS_IDS,
    WHEELED_OBSTACLE_CLASS_IDS,
)
from surface_passability.policies.obstacles import (
    GROUND_SUPPORT_COLUMN_KERNEL_WIDTH_FRACTION,
    GROUND_SUPPORT_IMAGE_HEIGHT_FRACTION,
    GROUND_SUPPORT_MINIMUM_COLUMN_KERNEL_PX,
    GROUND_SUPPORT_MINIMUM_DEPTH_PX,
    SOURCE_CONTACT_BAND_HEIGHT_FRACTION,
    SOURCE_CONTACT_BOTTOM_TRUNCATION_MARGIN_PX,
    SOURCE_CONTACT_HORIZONTAL_RADIUS_PX,
    SOURCE_CONTACT_MINIMUM_BAND_DEPTH_PX,
    SOURCE_CONTACT_PROBE_DEPTH_PX,
    OWNERSHIP_MINIMUM_GROUND_SUPPORT_FRACTION,
    OWNERSHIP_MINIMUM_GROUND_SUPPORT_PIXELS,
    OWNERSHIP_MINIMUM_SOURCE_CONTACT_COLUMNS,
    OWNERSHIP_MINIMUM_SOURCE_CONTACT_FRACTION,
    LOWER_SILHOUETTE_HEIGHT_FRACTION,
)


def bbox_separation(
    first: list[int] | tuple[int, int, int, int],
    second: list[int] | tuple[int, int, int, int],
) -> tuple[int, int]:
    """Return horizontal and vertical pixel gaps between inclusive boxes."""

    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    first_right = first_x + first_width - 1
    first_bottom = first_y + first_height - 1
    second_right = second_x + second_width - 1
    second_bottom = second_y + second_height - 1
    dx = max(
        0,
        max(first_x, second_x) - min(first_right, second_right) - 1,
    )
    dy = max(
        0,
        max(first_y, second_y) - min(first_bottom, second_bottom) - 1,
    )
    return dx, dy


def measure_projected_ground_support(
    component_mask: np.ndarray,
    footprint_mask: np.ndarray,
    surface_mask: np.ndarray,
) -> ProjectedGroundSupport:
    """Retain projected footprint columns supported beneath the object."""

    height, width = component_mask.shape
    rows_y, columns_x = np.nonzero(component_mask)
    if not len(columns_x):
        return ProjectedGroundSupport(
            effective_mask=np.zeros_like(footprint_mask),
            support_pixels=0,
            support_fraction=0.0,
        )
    bottom = int(rows_y.max())
    depth = max(
        GROUND_SUPPORT_MINIMUM_DEPTH_PX,
        round(height * GROUND_SUPPORT_IMAGE_HEIGHT_FRACTION),
    )
    y0, y1 = max(0, bottom - 1), min(height, bottom + depth + 1)
    supported_columns = surface_mask[y0:y1].any(axis=0)
    kernel_width = max(
        GROUND_SUPPORT_MINIMUM_COLUMN_KERNEL_PX,
        round(width * GROUND_SUPPORT_COLUMN_KERNEL_WIDTH_FRACTION),
    ) | 1
    supported_columns = cv2.dilate(
        supported_columns.astype(np.uint8)[None, :],
        np.ones((1, kernel_width), np.uint8),
    )[0].astype(bool)
    effective_mask = footprint_mask & supported_columns[None, :]
    support_pixels = int(effective_mask.sum())
    support_fraction = support_pixels / max(1, int(footprint_mask.sum()))
    return ProjectedGroundSupport(
        effective_mask=effective_mask,
        support_pixels=support_pixels,
        support_fraction=support_fraction,
    )


def measure_source_contact_support(
    component_mask: np.ndarray,
    surface_mask: np.ndarray,
) -> SourceContactSupport:
    """Measure surface support immediately below original lowest contact columns."""

    height, width = component_mask.shape
    rows_y, columns_x = np.nonzero(component_mask)
    if not len(columns_x):
        return SourceContactSupport(0, 0, 0.0, False)
    bottom = int(rows_y.max())
    if bottom >= height - SOURCE_CONTACT_BOTTOM_TRUNCATION_MARGIN_PX:
        return SourceContactSupport(0, 0, 1.0, True)

    bbox_height = int(rows_y.max() - rows_y.min() + 1)
    band_top = max(
        int(rows_y.min()),
        bottom
        - max(
            SOURCE_CONTACT_MINIMUM_BAND_DEPTH_PX,
            round(bbox_height * SOURCE_CONTACT_BAND_HEIGHT_FRACTION),
        )
        + 1,
    )
    contact = component_mask.copy()
    contact[:band_top] = False
    contact_columns = np.flatnonzero(contact.any(axis=0))
    if not len(contact_columns):
        return SourceContactSupport(0, 0, 0.0, False)

    supported = 0
    for column_x in contact_columns:
        local_bottom = int(np.flatnonzero(component_mask[:, column_x]).max())
        y0 = local_bottom + 1
        y1 = min(height, local_bottom + SOURCE_CONTACT_PROBE_DEPTH_PX + 1)
        x0 = max(0, int(column_x) - SOURCE_CONTACT_HORIZONTAL_RADIUS_PX)
        x1 = min(
            width,
            int(column_x) + SOURCE_CONTACT_HORIZONTAL_RADIUS_PX + 1,
        )
        if y0 < y1 and surface_mask[y0:y1, x0:x1].any():
            supported += 1
    total = int(len(contact_columns))
    ratio = supported / max(1, total)
    return SourceContactSupport(supported, total, ratio, False)


def evaluate_footprint_ownership(
    component_mask: np.ndarray,
    footprint: ObstacleFootprint | None,
    surface_mask: np.ndarray,
) -> OwnedObstacleFootprint:
    """Accept a footprint only when projected and original contacts own it."""

    if footprint is None:
        return OwnedObstacleFootprint(
            selected_mask=np.zeros_like(component_mask),
            source=None,
            accepted=False,
            rejection_reason="no_footprint",
        )
    projected = measure_projected_ground_support(
        component_mask,
        footprint.mask,
        surface_mask,
    )
    minimum_support = max(
        OWNERSHIP_MINIMUM_GROUND_SUPPORT_PIXELS,
        round(
            footprint.mask.sum()
            * OWNERSHIP_MINIMUM_GROUND_SUPPORT_FRACTION
        ),
    )
    contact = measure_source_contact_support(component_mask, surface_mask)
    source_ok = contact.bottom_truncated or (
        contact.supported_columns
        >= OWNERSHIP_MINIMUM_SOURCE_CONTACT_COLUMNS
        and contact.support_ratio
        >= OWNERSHIP_MINIMUM_SOURCE_CONTACT_FRACTION
    )
    accepted = projected.support_pixels >= minimum_support and source_ok
    rejection_reason = (
        None
        if accepted
        else (
            "off_walkway_source_contact"
            if not source_ok
            else "no_selected_walkway_ground_support"
        )
    )
    return OwnedObstacleFootprint(
        selected_mask=(
            projected.effective_mask
            if accepted
            else np.zeros_like(component_mask)
        ),
        source=footprint,
        accepted=bool(accepted),
        ground_support_pixels=int(projected.support_pixels),
        ground_support_fraction=float(projected.support_fraction),
        minimum_ground_support_pixels=int(minimum_support),
        source_contact_supported_columns=int(contact.supported_columns),
        source_contact_total_columns=int(contact.total_columns),
        source_contact_support_fraction=float(contact.support_ratio),
        source_bottom_truncated=bool(contact.bottom_truncated),
        rejection_reason=rejection_reason,
    )


def evaluate_banner_support_ownership(
    geometry: ObstacleGeometry,
    active_objects: tuple[ObstacleObjectGeometry, ...],
    surface_mask: np.ndarray,
    *,
    group_mode: str = "nearest",
) -> BannerSupportOwnership:
    """Evaluate banner/support clusters for one explicit surface."""

    labels = geometry.labels
    recognized = np.zeros_like(labels, dtype=bool)
    selected = np.zeros_like(labels, dtype=bool)
    silhouette_selected = np.zeros_like(labels, dtype=bool)
    supports = tuple(
        obj
        for obj in active_objects
        if obj.component.class_id in BANNER_SUPPORT_CLASS_IDS
    )
    grouped_support_ids: set[tuple[int, int]] = set()
    records = []
    passability_instances = []
    for banner in geometry.banners:
        recognized |= banner.mask
        nearby: list[tuple[int, ObstacleObjectGeometry]] = []
        for support in supports:
            dx, dy = bbox_separation(banner.bbox, support.component.bbox)
            if dx <= max(6, round(labels.shape[1] * 0.025)) and dy <= max(
                8,
                round(labels.shape[0] * 0.04),
            ):
                nearby.append((dx + dy, support))
        if not nearby:
            records.append(StandaloneBannerEvidence(
                banner=banner,
                class_name=geometry.id2label[BANNER_CLASS_ID],
            ))
            continue
        if group_mode == "grounded_all":
            selected_supports = []
            for _, support in nearby:
                grounded = evaluate_grounded_silhouette(
                    support.lower_silhouette_candidate,
                    support.component.mask,
                    surface_mask,
                )
                if grounded.accepted:
                    selected_supports.append(support)
            if not selected_supports:
                records.append(StandaloneBannerEvidence(
                    banner=banner,
                    class_name=geometry.id2label[BANNER_CLASS_ID],
                ))
                continue
        elif group_mode == "nearest":
            selected_supports = [min(nearby, key=lambda value: value[0])[1]]
        else:
            raise ValueError(
                f"Unsupported banner_support_group_mode: {group_mode}"
            )

        combined = banner.mask.copy()
        for support in selected_supports:
            component = support.component
            grouped_support_ids.add((component.class_id, component.component_index))
            combined |= component.mask
        grouped_boxes = [banner.bbox] + [
            support.component.bbox
            for support in selected_supports
        ]
        group_x0 = min(box[0] for box in grouped_boxes)
        group_y0 = min(box[1] for box in grouped_boxes)
        group_x1 = max(box[0] + box[2] for box in grouped_boxes)
        group_y1 = max(box[1] + box[3] for box in grouped_boxes)
        candidate = extract_lower_silhouette(combined)
        grounded = evaluate_grounded_silhouette(
            candidate,
            combined,
            surface_mask,
        )
        banner_name = geometry.id2label[BANNER_CLASS_ID]
        support_names = [
            geometry.id2label[support.component.class_id]
            for support in selected_supports
        ]
        class_name = " + ".join([banner_name] + support_names)
        members = tuple(
            BannerSupportMember(
                component=support.component,
                class_name=support_name,
            )
            for support, support_name in zip(
                selected_supports,
                support_names,
            )
        )
        if grounded.accepted:
            selected |= candidate
            silhouette_selected |= candidate
            passability_instances.append(PassabilityObstacleInstance(
                class_id=BANNER_CLASS_ID,
                class_name=class_name,
                kind="grounded_banner_support_cluster",
                source_mask=combined.copy(),
                selected_mask=candidate.copy(),
            ))
        records.append(BannerSupportClusterEvidence(
            class_id=BANNER_CLASS_ID,
            class_name=class_name,
            bbox=(
                group_x0,
                group_y0,
                group_x1 - group_x0,
                group_y1 - group_y0,
            ),
            group_mode=group_mode,
            grouped_supports=members,
            source_mask=combined,
            selected_mask=candidate,
            grounded=grounded,
            silhouette_fraction=LOWER_SILHOUETTE_HEIGHT_FRACTION,
        ))
    return BannerSupportOwnership(
        recognized_mask=recognized,
        selected_mask=selected,
        silhouette_selected_mask=silhouette_selected,
        grouped_support_ids=frozenset(grouped_support_ids),
        records=tuple(records),
        passability_instances=tuple(passability_instances),
    )


def evaluate_wheeled_obstacle_ownership(
    geometry: ObstacleGeometry,
    active_objects: tuple[ObstacleObjectGeometry, ...],
    surface_mask: np.ndarray,
    grouped_support_ids: frozenset[tuple[int, int]],
    *,
    silhouette_angle_max_degrees: float | None = None,
) -> WheeledObstacleOwnership:
    """Evaluate wheeled obstacle ownership without changing object ordering."""

    shape = geometry.labels.shape
    recognized = np.zeros(shape, dtype=bool)
    selected = np.zeros(shape, dtype=bool)
    footprint_selected = np.zeros(shape, dtype=bool)
    silhouette_selected = np.zeros(shape, dtype=bool)
    wheeled_silhouette_selected = np.zeros(shape, dtype=bool)
    baseline_for_natural = np.zeros(shape, dtype=bool)
    records = []
    passability_instances = []
    for obj in active_objects:
        component = obj.component
        key = (component.class_id, component.component_index)
        if key in grouped_support_ids:
            continue
        if component.class_id not in WHEELED_OBSTACLE_CLASS_IDS:
            continue
        source = component.mask
        recognized |= source
        if not obj.footprint_evaluated:
            raise KeyError("_footprint_candidate")
        ownership = evaluate_footprint_ownership(
            source,
            obj.footprint,
            surface_mask,
        )
        candidate = ownership.selected_mask
        strategy = "footprint"
        silhouette_exception = bool(
            silhouette_angle_max_degrees is not None
            and obj.axis_geometry.angle_from_vertical_degrees
            <= silhouette_angle_max_degrees
            and ownership.accepted
        )
        if silhouette_exception:
            selector = "orientation_selected_silhouette"
            strategy = "selected_silhouette"
            candidate = source.copy()
        elif silhouette_angle_max_degrees is not None:
            selector = "orientation_rejected_default_footprint"
        else:
            selector = "deferred_default_footprint"
        class_name = geometry.id2label.get(
            component.class_id,
            str(component.class_id),
        )
        if candidate.any():
            selected |= candidate
            if strategy == "selected_silhouette":
                silhouette_selected |= candidate
                wheeled_silhouette_selected |= candidate
            else:
                footprint_selected |= candidate
            baseline_for_natural |= candidate
            passability_instances.append(PassabilityObstacleInstance(
                class_id=int(component.class_id),
                class_name=class_name,
                kind="wheeled_unmanned_obstacle",
                source_mask=source.copy(),
                selected_mask=candidate.copy(),
            ))
        records.append(WheeledObstacleEvidence(
            source=obj,
            footprint_ownership=ownership,
            class_name=class_name,
            strategy=strategy,
            selector_state=selector,
            silhouette_angle_max_degrees=silhouette_angle_max_degrees,
            selected_mask=candidate,
        ))
    return WheeledObstacleOwnership(
        recognized_mask=recognized,
        selected_mask=selected,
        footprint_selected_mask=footprint_selected,
        silhouette_selected_mask=silhouette_selected,
        wheeled_silhouette_selected_mask=wheeled_silhouette_selected,
        baseline_for_natural_mask=baseline_for_natural,
        records=tuple(records),
        passability_instances=tuple(passability_instances),
    )


def evaluate_grounded_obstacle_ownership(
    geometry: ObstacleGeometry,
    active_objects: tuple[ObstacleObjectGeometry, ...],
    surface_mask: np.ndarray,
    grouped_support_ids: frozenset[tuple[int, int]],
) -> GroundedObstacleOwnership:
    """Evaluate non-wheeled lower-silhouette ownership in source order."""

    shape = geometry.labels.shape
    recognized = np.zeros(shape, dtype=bool)
    selected = np.zeros(shape, dtype=bool)
    silhouette_selected = np.zeros(shape, dtype=bool)
    baseline_for_natural = np.zeros(shape, dtype=bool)
    records = []
    passability_instances = []
    for obj in active_objects:
        component = obj.component
        key = (component.class_id, component.component_index)
        if key in grouped_support_ids:
            continue
        if component.class_id in WHEELED_OBSTACLE_CLASS_IDS:
            continue
        source = component.mask
        recognized |= source
        candidate = obj.lower_silhouette_candidate
        grounding = evaluate_grounded_silhouette(
            candidate,
            source,
            surface_mask,
        )
        class_name = geometry.id2label.get(
            component.class_id,
            str(component.class_id),
        )
        if grounding.accepted:
            selected |= candidate
            silhouette_selected |= candidate
            baseline_for_natural |= candidate
            passability_instances.append(PassabilityObstacleInstance(
                class_id=int(component.class_id),
                class_name=class_name,
                kind="other_grounded_obstacle",
                source_mask=source.copy(),
                selected_mask=candidate.copy(),
            ))
        records.append(GroundedObstacleOwnershipEvidence(
            source=obj,
            class_name=class_name,
            grounding=grounding,
            silhouette_fraction=LOWER_SILHOUETTE_HEIGHT_FRACTION,
        ))
    return GroundedObstacleOwnership(
        recognized_mask=recognized,
        selected_mask=selected,
        silhouette_selected_mask=silhouette_selected,
        baseline_for_natural_mask=baseline_for_natural,
        records=tuple(records),
        passability_instances=tuple(passability_instances),
    )


def project_natural_hole_ownership(
    geometry: ObstacleGeometry,
    repaired_surface_mask: np.ndarray,
    structural_surface_mask: np.ndarray,
    source_surface_mask: np.ndarray,
    baseline_footprint_mask: np.ndarray,
) -> NaturalHoleOwnership:
    """Project approved natural-hole evidence into ownership masks and records."""

    evidence = collect_natural_hole_evidence(
        geometry.labels,
        repaired_surface_mask,
        structural_surface_mask,
        source_surface_mask,
        baseline_footprint_mask,
    )
    records = []
    for row in evidence.records:
        class_id = int(row.get("class_id", -1))
        records.append(NaturalHoleOwnershipRecord(
            source_record_items=tuple(row.items()),
            class_name=geometry.id2label.get(
                class_id,
                str(row.get("class_id")),
            ),
            selected_pixels=(
                int(row.get("footprint_pixels", 0))
                if row.get("accepted")
                else 0
            ),
        ))
    return NaturalHoleOwnership(
        evidence=evidence,
        records=tuple(records),
    )


def project_transient_occluder_ownership(
    geometry: ObstacleGeometry,
) -> TransientOccluderOwnership:
    """Project transient semantic objects into recognized-only evidence."""

    recognized = np.zeros_like(geometry.labels, dtype=bool)
    records = []
    for component in geometry.all_objects:
        if component.class_id in TRANSIENT_SOURCE_CLASS_IDS:
            recognized |= component.mask
            records.append(TransientOccluderEvidence(
                component=component,
                class_name=geometry.id2label.get(
                    component.class_id,
                    str(component.class_id),
                ),
            ))
    return TransientOccluderOwnership(
        recognized_mask=recognized,
        records=tuple(records),
    )


def project_semantic_hazard_ownership(
    geometry: ObstacleGeometry,
) -> SemanticHazardOwnership:
    """Project full semantic hazard components into occupancy evidence."""

    selected = np.isin(
        geometry.labels,
        tuple(sorted(geometry.hazard_ids)),
    )
    recognized, records, passability_instances = (
        collect_semantic_hazard_ownership_records(geometry)
    )
    return SemanticHazardOwnership(
        recognized_mask=recognized,
        selected_mask=selected,
        records=records,
        passability_instances=passability_instances,
    )


def collect_semantic_hazard_ownership_records(
    geometry: ObstacleGeometry,
) -> tuple[
    np.ndarray,
    tuple[SemanticHazardEvidence, ...],
    tuple[PassabilityObstacleInstance, ...],
]:
    """Extract hazard records and instances without recomputing policy masks."""

    components = extract_semantic_obstacle_components(
        geometry.labels,
        geometry.hazard_ids,
    )
    recognized = np.zeros_like(geometry.labels, dtype=bool)
    records = []
    passability_instances = []
    for component in components:
        recognized |= component.mask
        evidence = SemanticHazardEvidence(
            component=component,
            class_name=geometry.id2label.get(
                component.class_id,
                str(component.class_id),
            ),
        )
        records.append(evidence)
        passability_instances.append(evidence.to_passability_instance())
    return (
        recognized,
        tuple(records),
        tuple(passability_instances),
    )
