from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.occupancy.blockers import (
    BANNER_ID,
    BANNER_SUPPORT_IDS,
    LOWER_SILHOUETTE_FRACTION,
    NATURAL_IDS,
    ROOT,
    TRANSIENT_IDS,
    WEIGHTS,
    WHEELED_IDS,
    _bbox_distance,
    _clean,
    _components,
    _grounded,
    _lower_axis_geometry,
    _lower_silhouette,
)
from roadpush_ai.surface.context import SurfaceContext


@dataclass
class ObstacleGeometry:
    """Image-level obstacle identities and surface-independent source geometry."""

    image: np.ndarray
    labels: np.ndarray
    all_objects: list[dict]
    active_objects: list[dict]
    banners: list[dict]
    transient_source: np.ndarray
    natural_source_evidence: np.ndarray
    hazard_ids: set[int]
    id2label: dict[int, str]
    yolo_detections: list[dict]
    yolo_inference: dict | None
    surface_trigger_records: dict[str, list[dict]]


def _enrich_object_geometry(obj: dict) -> dict:
    enriched = dict(obj)
    source = obj["_mask"]
    enriched["_lower_silhouette_candidate"] = _lower_silhouette(source)
    enriched["_axis_geometry"] = _lower_axis_geometry(source)
    if obj["class_id"] in WHEELED_IDS:
        estimated = legacy.policy.r3.class_aware_component_footprint(
            source, obj["class_id"]
        )
        if estimated is None:
            enriched["_footprint_candidate"] = np.zeros_like(source)
            enriched["_footprint_metadata"] = None
        else:
            candidate, metadata = estimated
            enriched["_footprint_candidate"] = candidate
            enriched["_footprint_metadata"] = metadata
    return enriched


def _surface_yolo_metadata(
    geometry: ObstacleGeometry,
    surface: SurfaceContext,
) -> tuple[list[dict], dict]:
    """Build one surface's legacy car view from shared, non-destructive detections."""

    trigger_records = geometry.surface_trigger_records.get(surface.profile.name, [])
    triggered = [row for row in trigger_records if row["trigger_yolo"]]
    metadata = {
        "triggered": bool(triggered),
        "trigger_component_count": len(triggered),
        "inference_run": False,
        "result": "not_triggered",
        "split_confirmed_count": 0,
        "fallback_parent_count": 0,
        "components": [],
        "scope": "existing_v2_car_trigger_only",
    }
    if not triggered:
        return geometry.active_objects, metadata

    metadata["inference_run"] = True
    metadata["inference"] = geometry.yolo_inference

    triggered_masks = [row["_component"] for row in triggered]
    kept = [
        obj
        for obj in geometry.active_objects
        if not (
            obj["class_id"] == legacy.integrated.CAR_ID
            and any(np.array_equal(obj["_mask"], parent) for parent in triggered_masks)
        )
    ]
    for trigger_record in triggered:
        parent = trigger_record["_component"]
        matched, detection_audit = legacy.integrated.yolo_split.match_instances(
            parent, geometry.yolo_detections
        )
        split = len(matched) >= 2 and all(instance["_mask"].any() for instance in matched)
        component_meta = {
            "source_bbox": trigger_record["source_bbox"],
            "matched_instances": len(matched),
            "split_confirmed": bool(split),
            "detections": detection_audit,
        }
        if split:
            metadata["split_confirmed_count"] += 1
            children = legacy.integrated.yolo_split.partition_mapillary(parent, matched)
            for child_index, (child, instance) in enumerate(zip(children, matched)):
                kept.append(
                    _enrich_object_geometry({
                        "class_id": legacy.integrated.CAR_ID,
                        "component_index": trigger_record.get("component_id"),
                        "bbox": trigger_record["source_bbox"],
                        "source_pixels": int(child.sum()),
                        "instance_source": "yolo_partitioned_mapillary",
                        "instance_index": child_index,
                        "yolo_detection_index": instance["detection_index"],
                        "yolo_class": instance["class_name"],
                        "yolo_confidence": instance["confidence"],
                        "_mask": child,
                    })
                )
        else:
            metadata["fallback_parent_count"] += 1
            kept.append(
                _enrich_object_geometry({
                    "class_id": legacy.integrated.CAR_ID,
                    "component_index": trigger_record.get("component_id"),
                    "bbox": trigger_record["source_bbox"],
                    "source_pixels": int(parent.sum()),
                    "instance_source": "yolo_inconclusive_parent_fallback",
                    "_mask": parent,
                })
            )
        metadata["components"].append(component_meta)
    metadata["result"] = (
        "split_confirmed" if metadata["split_confirmed_count"] else "fallback_parent"
    )
    return kept, metadata


def extract_obstacle_geometry(
    image: np.ndarray,
    labels: np.ndarray,
) -> ObstacleGeometry:
    """Extract surface-independent identities and source geometry once."""

    occupancy_ids = set(legacy.policy.probe.v2.OCCUPANCY)
    hazard_ids = set(legacy.policy.probe.v2.HAZARD)
    id2label = {
        int(key): value
        for key, value in __import__("json").loads(
            (
                ROOT
                / "models"
                / "mask2former_swin_large_mapillary"
                / "config.json"
            ).read_text(encoding="utf-8")
        )["id2label"].items()
    }
    all_objects = _components(labels, occupancy_ids)
    active = [
        obj
        for obj in all_objects
        if obj["class_id"] not in TRANSIENT_IDS and obj["class_id"] != BANNER_ID
    ]
    enriched_active = [_enrich_object_geometry(obj) for obj in active]
    return ObstacleGeometry(
        image=image,
        labels=labels,
        all_objects=all_objects,
        active_objects=enriched_active,
        banners=_components(labels, {BANNER_ID}),
        transient_source=np.isin(labels, tuple(sorted(TRANSIENT_IDS))),
        natural_source_evidence=np.isin(labels, tuple(sorted(NATURAL_IDS))),
        hazard_ids=hazard_ids,
        id2label=id2label,
        yolo_detections=[],
        yolo_inference=None,
        surface_trigger_records={},
    )


def prepare_obstacle_geometry(
    geometry: ObstacleGeometry,
    surfaces: list[SurfaceContext],
    model_holder: dict,
) -> ObstacleGeometry:
    """Resolve per-surface YOLO evidence with at most one image inference.

    Semantic parents remain in ``active_objects``. Shared YOLO detections are
    attached non-destructively, and each ownership pass decides whether its own
    surface uses a split view. This preserves the frozen sidewalk trigger even
    when another surface later requests YOLO for the same image.
    """

    trigger_records = {
        surface.profile.name: legacy.integrated.trigger.analyse_car_components(
            geometry.labels, surface.source_mask
        )
        for surface in surfaces
    }
    requires_yolo = any(
        row["trigger_yolo"]
        for records in trigger_records.values()
        for row in records
    )
    detections: list[dict] = []
    inference = None
    if requires_yolo:
        if "yolo" not in model_holder:
            model_holder["yolo"] = legacy.integrated.YOLO(WEIGHTS)
        detections, inference = legacy.integrated.yolo_split.infer_instances(
            model_holder["yolo"], geometry.image
        )
    return replace(
        geometry,
        yolo_detections=detections,
        yolo_inference=inference,
        surface_trigger_records=trigger_records,
    )


def _evaluate_footprint_ownership(
    obj: dict,
    walkway: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Apply surface support to an already-extracted footprint candidate."""

    component = obj["_mask"]
    candidate = obj["_footprint_candidate"]
    source_metadata = obj["_footprint_metadata"]
    if source_metadata is None:
        return np.zeros_like(component), {
            "accepted": False,
            "rejection_reason": "no_footprint",
        }
    metadata = dict(source_metadata)
    effective, support_pixels, support_fraction = legacy.policy.ground_support(
        component, candidate, walkway
    )
    minimum_support = max(6, round(candidate.sum() * 0.04))
    source_supported, source_total, source_ratio, bottom_truncated = (
        legacy.policy.source_contact_support(component, walkway)
    )
    source_ok = bottom_truncated or (
        source_supported >= 2 and source_ratio >= 0.35
    )
    accepted = support_pixels >= minimum_support and source_ok
    metadata.update(
        {
            "accepted": bool(accepted),
            "ground_support_pixels": int(support_pixels),
            "ground_support_fraction": float(support_fraction),
            "minimum_ground_support_pixels": int(minimum_support),
            "source_contact_supported_columns": int(source_supported),
            "source_contact_total_columns": int(source_total),
            "source_contact_support_fraction": float(source_ratio),
            "source_bottom_truncated": bool(bottom_truncated),
            "rejection_reason": None
            if accepted
            else (
                "off_walkway_source_contact"
                if not source_ok
                else "no_selected_walkway_ground_support"
            ),
        }
    )
    return effective if accepted else np.zeros_like(component), metadata


def evaluate_obstacle_ownership(
    geometry: ObstacleGeometry,
    surface: SurfaceContext,
    *,
    wheeled_silhouette_angle_max_deg: float | None = None,
    banner_support_group_mode: str = "nearest",
) -> dict:
    """Evaluate frozen blocker ownership against one explicit surface context."""

    labels = geometry.labels
    walkway = surface.source_mask
    active_objects, yolo = _surface_yolo_metadata(geometry, surface)
    recognized = geometry.transient_source | np.isin(
        labels, tuple(sorted(geometry.hazard_ids))
    )
    records: list[dict] = []
    passability_instances: list[dict] = []
    final = np.isin(labels, tuple(sorted(geometry.hazard_ids)))
    footprint_selected = np.zeros_like(labels, dtype=bool)
    silhouette_selected = np.zeros_like(labels, dtype=bool)
    wheeled_silhouette_selected = np.zeros_like(labels, dtype=bool)

    supports = [
        obj for obj in active_objects if obj["class_id"] in BANNER_SUPPORT_IDS
    ]
    grouped_support_ids: set[tuple[int, int]] = set()
    for banner in geometry.banners:
        recognized |= banner["_mask"]
        nearby = []
        for support in supports:
            dx, dy = _bbox_distance(banner["bbox"], support["bbox"])
            if dx <= max(6, round(labels.shape[1] * 0.025)) and dy <= max(
                8, round(labels.shape[0] * 0.04)
            ):
                nearby.append((dx + dy, support))
        if not nearby:
            records.append(
                {
                    **_clean(banner),
                    "class_name": geometry.id2label[BANNER_ID],
                    "kind": "standalone_banner",
                    "strategy": "context_only",
                    "accepted": False,
                    "rejection_reason": "no_grounded_support_connection",
                }
            )
            continue
        if banner_support_group_mode == "grounded_all":
            selected_supports = []
            for _, support in nearby:
                support_candidate = support["_lower_silhouette_candidate"]
                grounded, _ = _grounded(
                    support_candidate, support["_mask"], walkway
                )
                if grounded:
                    selected_supports.append(support)
            if not selected_supports:
                records.append(
                    {
                        **_clean(banner),
                        "class_name": geometry.id2label[BANNER_ID],
                        "kind": "standalone_banner",
                        "strategy": "context_only",
                        "accepted": False,
                        "rejection_reason": "no_grounded_support_connection",
                    }
                )
                continue
        elif banner_support_group_mode == "nearest":
            selected_supports = [min(nearby, key=lambda value: value[0])[1]]
        else:
            raise ValueError(
                f"Unsupported banner_support_group_mode: {banner_support_group_mode}"
            )

        combined = banner["_mask"].copy()
        for support in selected_supports:
            grouped_support_ids.add((support["class_id"], support["component_index"]))
            combined |= support["_mask"]
        grouped_boxes = [banner["bbox"]] + [
            support["bbox"] for support in selected_supports
        ]
        group_x0 = min(box[0] for box in grouped_boxes)
        group_y0 = min(box[1] for box in grouped_boxes)
        group_x1 = max(box[0] + box[2] for box in grouped_boxes)
        group_y1 = max(box[1] + box[3] for box in grouped_boxes)
        candidate = _lower_silhouette(combined)
        accepted, support_meta = _grounded(candidate, combined, walkway)
        if accepted:
            final |= candidate
            silhouette_selected |= candidate
            passability_instances.append(
                {
                    "class_id": BANNER_ID,
                    "class_name": " + ".join(
                        [geometry.id2label[BANNER_ID]]
                        + [geometry.id2label[support["class_id"]] for support in selected_supports]
                    ),
                    "kind": "grounded_banner_support_cluster",
                    "source_mask": combined.copy(),
                    "selected_mask": candidate.copy(),
                }
            )
        records.append(
            {
                "class_id": BANNER_ID,
                "class_name": " + ".join(
                    [geometry.id2label[BANNER_ID]]
                    + [geometry.id2label[support["class_id"]] for support in selected_supports]
                ),
                "kind": "grounded_banner_support_cluster",
                "bbox": [
                    group_x0,
                    group_y0,
                    group_x1 - group_x0,
                    group_y1 - group_y0,
                ],
                "strategy": "lower_19_percent_silhouette",
                "banner_support_group_mode": banner_support_group_mode,
                "grouped_supports": [
                    {
                        "class_id": int(support["class_id"]),
                        "class_name": geometry.id2label[support["class_id"]],
                        "component_index": int(support["component_index"]),
                        "bbox": support["bbox"],
                    }
                    for support in selected_supports
                ],
                "silhouette_fraction": LOWER_SILHOUETTE_FRACTION,
                "source_pixels": int(combined.sum()),
                "selected_pixels": int(candidate.sum()) if accepted else 0,
                "accepted": bool(accepted),
                "rejection_reason": None if accepted else "no_walkway_ground_support",
                **support_meta,
            }
        )

    baseline_for_natural = np.zeros_like(labels, dtype=bool)
    for obj in active_objects:
        key = (obj["class_id"], obj["component_index"])
        if key in grouped_support_ids:
            continue
        source = obj["_mask"]
        recognized |= source
        class_id = obj["class_id"]
        if class_id in WHEELED_IDS:
            candidate, metadata = _evaluate_footprint_ownership(obj, walkway)
            strategy = "footprint"
            axis_geometry = obj["_axis_geometry"]
            silhouette_exception = bool(
                wheeled_silhouette_angle_max_deg is not None
                and axis_geometry["lower_axis_angle_from_vertical_deg"]
                <= wheeled_silhouette_angle_max_deg
                and metadata.get("accepted")
            )
            if silhouette_exception:
                selector = "orientation_selected_silhouette"
                strategy = "selected_silhouette"
                candidate = source.copy()
            elif wheeled_silhouette_angle_max_deg is not None:
                selector = "orientation_rejected_default_footprint"
            else:
                selector = "deferred_default_footprint"
            if candidate.any():
                final |= candidate
                if strategy == "selected_silhouette":
                    silhouette_selected |= candidate
                    wheeled_silhouette_selected |= candidate
                else:
                    footprint_selected |= candidate
                baseline_for_natural |= candidate
                passability_instances.append(
                    {
                        "class_id": int(class_id),
                        "class_name": geometry.id2label.get(class_id, str(class_id)),
                        "kind": "wheeled_unmanned_obstacle",
                        "source_mask": source.copy(),
                        "selected_mask": candidate.copy(),
                    }
                )
            record = {
                **_clean(obj),
                **metadata,
                **axis_geometry,
                "class_name": geometry.id2label.get(class_id, str(class_id)),
                "kind": "wheeled_unmanned_obstacle",
                "strategy": strategy,
                "wheeled_selector_state": selector,
                "wheeled_silhouette_angle_max_deg": wheeled_silhouette_angle_max_deg,
                "scoped_closure_pixels": 0,
                "selected_pixels": int(candidate.sum()),
            }
        else:
            candidate = obj["_lower_silhouette_candidate"]
            accepted, support_meta = _grounded(candidate, source, walkway)
            if accepted:
                final |= candidate
                silhouette_selected |= candidate
                baseline_for_natural |= candidate
                passability_instances.append(
                    {
                        "class_id": int(class_id),
                        "class_name": geometry.id2label.get(class_id, str(class_id)),
                        "kind": "other_grounded_obstacle",
                        "source_mask": source.copy(),
                        "selected_mask": candidate.copy(),
                    }
                )
            record = {
                **_clean(obj),
                **support_meta,
                "class_name": geometry.id2label.get(class_id, str(class_id)),
                "kind": "other_grounded_obstacle",
                "strategy": "lower_19_percent_silhouette",
                "silhouette_fraction": LOWER_SILHOUETTE_FRACTION,
                "accepted": bool(accepted),
                "selected_pixels": int(candidate.sum()) if accepted else 0,
                "rejection_reason": None if accepted else "no_walkway_ground_support",
            }
        records.append(record)

    natural_fp, natural_source, _, _, natural_rows, _ = (
        legacy.grounded.universal_hole_candidates(
            labels,
            surface.repaired_mask,
            surface.structure_copy,
            surface.source_mask,
            baseline_for_natural,
        )
    )
    recognized |= natural_source
    final |= natural_fp
    footprint_selected |= natural_fp
    for row in natural_rows:
        row = dict(row)
        row.update(
            {
                "kind": "v2_1_natural_grounded_footprint",
                "strategy": "footprint",
                "class_name": geometry.id2label.get(
                    int(row.get("class_id", -1)), str(row.get("class_id"))
                ),
                "selected_pixels": int(row.get("footprint_pixels", 0))
                if row.get("accepted")
                else 0,
                "natural_restoration_applied": False,
            }
        )
        records.append(row)

    for obj in geometry.all_objects:
        if obj["class_id"] in TRANSIENT_IDS:
            recognized |= obj["_mask"]
            records.append(
                {
                    **_clean(obj),
                    "class_name": geometry.id2label.get(
                        obj["class_id"], str(obj["class_id"])
                    ),
                    "kind": "transient_occluder",
                    "strategy": "transient_rows_only",
                    "accepted": False,
                    "selected_pixels": 0,
                }
            )

    for hazard in _components(labels, geometry.hazard_ids):
        recognized |= hazard["_mask"]
        records.append(
            {
                **_clean(hazard),
                "class_name": geometry.id2label.get(
                    hazard["class_id"], str(hazard["class_id"])
                ),
                "kind": "semantic_hazard",
                "strategy": "full_semantic_hazard",
                "accepted": True,
                "selected_pixels": int(hazard["_mask"].sum()),
            }
        )
        passability_instances.append(
            {
                "class_id": int(hazard["class_id"]),
                "class_name": geometry.id2label.get(
                    hazard["class_id"], str(hazard["class_id"])
                ),
                "kind": "semantic_hazard",
                "source_mask": hazard["_mask"].copy(),
                "selected_mask": hazard["_mask"].copy(),
            }
        )

    strategy_overlap_pixels = int((footprint_selected & silhouette_selected).sum())
    violations = []
    if any(row.get("kind") == "cycle_gap_closure" for row in records):
        violations.append("forbidden_global_cycle_closure")
    if any(row.get("natural_restoration_applied") for row in records):
        violations.append("natural_restoration_leak")
    return {
        "recognized_mask": recognized,
        "transient_mask": geometry.transient_source,
        "hazard_mask": np.isin(labels, tuple(sorted(geometry.hazard_ids))),
        "footprint_mask": footprint_selected,
        "silhouette_mask": silhouette_selected,
        "wheeled_silhouette_mask": wheeled_silhouette_selected,
        "final_mask": final,
        "objects": records,
        "passability_instances": passability_instances,
        "yolo": yolo,
        "violations": violations,
        "strategy_overlap_pixels": strategy_overlap_pixels,
        "policy": {
            "ground_animal_transient": True,
            "wheeled_default": "footprint",
            "wheeled_exception_selector": "lower_axis_angle_probe"
            if wheeled_silhouette_angle_max_deg is not None
            else "deferred_not_integrated",
            "wheeled_silhouette_angle_max_deg": wheeled_silhouette_angle_max_deg,
            "non_wheeled": "lower_19_percent_silhouette_proposal",
            "banner": "grounded_support_cluster_lower_19_percent_silhouette",
            "banner_support_group_mode": banner_support_group_mode,
            "global_cycle_closure": False,
            "natural_grounded_footprint": True,
            "natural_restoration": False,
            "cpu_only": True,
        },
    }
