from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import os
from pathlib import Path

import cv2
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.surface.preprocessing import SharedPreprocessing


ROOT = get_settings().project_root
WEIGHTS = ROOT / "models" / "yolo11n-seg.pt"
# V3 is a CPU deployment contract.  Assign rather than setdefault so an
# inherited caller environment cannot silently route the scoped YOLO fallback
# to a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ROADPUSH_YOLO_DEVICE"] = "cpu"

GROUND_ANIMAL_ID = 1
BANNER_ID = 32
TRANSIENT_IDS = {GROUND_ANIMAL_ID, 19, 20, 21, 22}
WHEELED_IDS = {52, 54, 55, 56, 57, 58, 59, 60, 61, 62}
BANNER_SUPPORT_IDS = {35, 45, 46, 47, 49, 50}
NATURAL_IDS = {25, 29, 30}
LOWER_SILHOUETTE_FRACTION = 0.19


def _clean(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _components(labels: np.ndarray, class_ids: set[int] | tuple[int, ...]) -> list[dict]:
    records: list[dict] = []
    minimum = max(4, round(labels.size * 0.000008))
    for class_id in sorted(class_ids):
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            (labels == class_id).astype(np.uint8), 8
        )
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < minimum:
                continue
            records.append({
                "class_id": int(class_id),
                "component_index": int(index),
                "bbox": [int(x) for x in stats[index, :4]],
                "source_pixels": int(stats[index, cv2.CC_STAT_AREA]),
                "_mask": component_labels == index,
            })
    return records


def _bbox_distance(a: list[int], b: list[int]) -> tuple[int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        max(0, max(ax, bx) - min(ax + aw - 1, bx + bw - 1) - 1),
        max(0, max(ay, by) - min(ay + ah - 1, by + bh - 1) - 1),
    )


def _lower_silhouette(source: np.ndarray, fraction: float = LOWER_SILHOUETTE_FRACTION) -> np.ndarray:
    ys, _ = np.nonzero(source)
    result = np.zeros_like(source, dtype=bool)
    if not len(ys):
        return result
    top, bottom = int(ys.min()), int(ys.max())
    band = max(1, int(np.ceil((bottom - top + 1) * fraction)))
    result[max(top, bottom - band + 1):bottom + 1] = source[max(top, bottom - band + 1):bottom + 1]
    return result


def _grounded(candidate: np.ndarray, source: np.ndarray, walkway: np.ndarray) -> tuple[bool, dict]:
    if not candidate.any():
        return False, {"ground_support_pixels": 0, "ground_support_fraction": 0.0}
    ring = cv2.dilate(candidate.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    support = int((ring & walkway).sum())
    fraction = support / max(1, int(ring.sum()))
    ys, _ = np.nonzero(source)
    bottom_truncated = bool(len(ys) and int(ys.max()) >= source.shape[0] - 2)
    accepted = bottom_truncated or support >= 6
    return accepted, {
        "ground_support_pixels": support,
        "ground_support_fraction": float(fraction),
        "source_bottom_truncated": bottom_truncated,
    }


def _owned_footprint(component: np.ndarray, class_id: int, walkway: np.ndarray) -> tuple[np.ndarray, dict]:
    estimated = legacy.policy.r3.class_aware_component_footprint(component, class_id)
    if estimated is None:
        return np.zeros_like(component), {"accepted": False, "rejection_reason": "no_footprint"}
    candidate, metadata = estimated
    effective, support_pixels, support_fraction = legacy.policy.ground_support(
        component, candidate, walkway
    )
    minimum_support = max(6, round(candidate.sum() * 0.04))
    source_supported, source_total, source_ratio, bottom_truncated = legacy.policy.source_contact_support(
        component, walkway
    )
    source_ok = bottom_truncated or (source_supported >= 2 and source_ratio >= 0.35)
    accepted = support_pixels >= minimum_support and source_ok
    metadata.update({
        "accepted": bool(accepted),
        "ground_support_pixels": int(support_pixels),
        "ground_support_fraction": float(support_fraction),
        "minimum_ground_support_pixels": int(minimum_support),
        "source_contact_supported_columns": int(source_supported),
        "source_contact_total_columns": int(source_total),
        "source_contact_support_fraction": float(source_ratio),
        "source_bottom_truncated": bool(bottom_truncated),
        "rejection_reason": None if accepted else (
            "off_walkway_source_contact" if not source_ok else "no_selected_walkway_ground_support"
        ),
    })
    return (effective if accepted else np.zeros_like(component)), metadata


def _lower_axis_geometry(component: np.ndarray) -> dict:
    """Return the historical lower-mask PCA cue used by the orientation probes."""
    ys, xs = np.nonzero(component)
    if not len(xs):
        return {
            "lower_axis_angle_from_vertical_deg": 90.0,
            "axis_center": [0.0, 0.0],
            "axis_vector": [1.0, 0.0],
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    height = y1 - y0 + 1
    lower = component.copy()
    lower[: y0 + round(height * 0.42)] = False
    lys, lxs = np.nonzero(lower)
    if len(lxs) < 6:
        return {
            "lower_axis_angle_from_vertical_deg": 90.0,
            "axis_center": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
            "axis_vector": [1.0, 0.0],
        }
    points = np.column_stack((lxs, lys)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[1] < 0:
        axis = -axis
    return {
        "lower_axis_angle_from_vertical_deg": float(
            np.degrees(np.arctan2(abs(axis[0]), abs(axis[1]) + 1e-9))
        ),
        "axis_center": [float(center[0]), float(center[1])],
        "axis_vector": [float(axis[0]), float(axis[1])],
    }


def _yolo_split_car_parents(
    shared: SharedPreprocessing,
    objects: list[dict],
    model_holder: dict,
) -> tuple[list[dict], dict]:
    """Apply only the already-audited V2 car ambiguity trigger, scoped per parent."""
    trigger_records = legacy.integrated.trigger.analyse_car_components(
        shared.labels, shared.source_walkway
    )
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
        return objects, metadata

    if "yolo" not in model_holder:
        model_holder["yolo"] = legacy.integrated.YOLO(WEIGHTS)
    detections, inference = legacy.integrated.yolo_split.infer_instances(
        model_holder["yolo"], shared.image
    )
    metadata["inference_run"] = True
    metadata["inference"] = inference

    triggered_masks = [row["_component"] for row in triggered]
    kept = [obj for obj in objects if not (
        obj["class_id"] == legacy.integrated.CAR_ID
        and any(np.array_equal(obj["_mask"], parent) for parent in triggered_masks)
    )]
    for trigger_record in triggered:
        parent = trigger_record["_component"]
        matched, detection_audit = legacy.integrated.yolo_split.match_instances(parent, detections)
        split = len(matched) >= 2 and all(instance["_mask"].any() for instance in matched)
        component_meta = {
            "source_bbox": trigger_record["source_bbox"],
            "matched_instances": len(matched),
            "split_confirmed": bool(split),
            "detections": detection_audit,
        }
        if split:
            metadata["split_confirmed_count"] += 1
            for child_index, (child, instance) in enumerate(zip(
                legacy.integrated.yolo_split.partition_mapillary(parent, matched), matched
            )):
                kept.append({
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
        else:
            metadata["fallback_parent_count"] += 1
            kept.append({
                "class_id": legacy.integrated.CAR_ID,
                "component_index": trigger_record.get("component_id"),
                "bbox": trigger_record["source_bbox"],
                "source_pixels": int(parent.sum()),
                "instance_source": "yolo_inconclusive_parent_fallback",
                "_mask": parent,
            })
        metadata["components"].append(component_meta)
    metadata["result"] = "split_confirmed" if metadata["split_confirmed_count"] else "fallback_parent"
    return kept, metadata


def build_v3_blockers(
    shared: SharedPreprocessing,
    model_holder: dict,
    *,
    wheeled_silhouette_angle_max_deg: float | None = None,
    banner_support_group_mode: str = "nearest",
) -> dict:
    labels = shared.labels
    occupancy_ids = set(legacy.policy.probe.v2.OCCUPANCY)
    hazard_ids = set(legacy.policy.probe.v2.HAZARD)
    id2label = {
        int(key): value for key, value in __import__("json").loads(
            (ROOT / "models" / "mask2former_swin_large_mapillary" / "config.json").read_text(encoding="utf-8")
        )["id2label"].items()
    }

    all_objects = _components(labels, occupancy_ids)
    transient_source = np.isin(labels, tuple(sorted(TRANSIENT_IDS)))
    recognized = transient_source | np.isin(labels, tuple(sorted(hazard_ids)))
    records: list[dict] = []
    final = np.isin(labels, tuple(sorted(hazard_ids)))
    footprint_selected = np.zeros_like(labels, dtype=bool)
    silhouette_selected = np.zeros_like(labels, dtype=bool)
    wheeled_silhouette_selected = np.zeros_like(labels, dtype=bool)

    active = [obj for obj in all_objects if obj["class_id"] not in TRANSIENT_IDS and obj["class_id"] != BANNER_ID]
    active, yolo = _yolo_split_car_parents(shared, active, model_holder)

    banners = _components(labels, {BANNER_ID})
    supports = [obj for obj in active if obj["class_id"] in BANNER_SUPPORT_IDS]
    grouped_support_ids: set[tuple[int, int]] = set()
    for banner in banners:
        recognized |= banner["_mask"]
        nearby = []
        for support in supports:
            dx, dy = _bbox_distance(banner["bbox"], support["bbox"])
            if dx <= max(6, round(labels.shape[1] * 0.025)) and dy <= max(8, round(labels.shape[0] * 0.04)):
                nearby.append((dx + dy, support))
        if not nearby:
            records.append({
                **_clean(banner), "class_name": id2label[BANNER_ID],
                "kind": "standalone_banner", "strategy": "context_only",
                "accepted": False, "rejection_reason": "no_grounded_support_connection",
            })
            continue
        if banner_support_group_mode == "grounded_all":
            selected_supports = []
            for _, support in nearby:
                support_candidate = _lower_silhouette(support["_mask"])
                grounded, _ = _grounded(
                    support_candidate, support["_mask"], shared.source_walkway
                )
                if grounded:
                    selected_supports.append(support)
            if not selected_supports:
                records.append({
                    **_clean(banner), "class_name": id2label[BANNER_ID],
                    "kind": "standalone_banner", "strategy": "context_only",
                    "accepted": False,
                    "rejection_reason": "no_grounded_support_connection",
                })
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
        grouped_boxes = [banner["bbox"]] + [support["bbox"] for support in selected_supports]
        group_x0 = min(box[0] for box in grouped_boxes)
        group_y0 = min(box[1] for box in grouped_boxes)
        group_x1 = max(box[0] + box[2] for box in grouped_boxes)
        group_y1 = max(box[1] + box[3] for box in grouped_boxes)
        candidate = _lower_silhouette(combined)
        accepted, support_meta = _grounded(candidate, combined, shared.source_walkway)
        if accepted:
            final |= candidate
            silhouette_selected |= candidate
        records.append({
            "class_id": BANNER_ID,
            "class_name": " + ".join(
                [id2label[BANNER_ID]]
                + [id2label[support["class_id"]] for support in selected_supports]
            ),
            "kind": "grounded_banner_support_cluster",
            "bbox": [
                group_x0, group_y0, group_x1 - group_x0, group_y1 - group_y0,
            ],
            "strategy": "lower_19_percent_silhouette",
            "banner_support_group_mode": banner_support_group_mode,
            "grouped_supports": [
                {
                    "class_id": int(support["class_id"]),
                    "class_name": id2label[support["class_id"]],
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
        })

    baseline_for_natural = np.zeros_like(labels, dtype=bool)
    for obj in active:
        key = (obj["class_id"], obj["component_index"])
        if key in grouped_support_ids:
            continue
        source = obj["_mask"]
        recognized |= source
        class_id = obj["class_id"]
        if class_id in WHEELED_IDS:
            candidate, metadata = _owned_footprint(source, class_id, shared.source_walkway)
            strategy = "footprint"
            geometry = _lower_axis_geometry(source)
            silhouette_exception = bool(
                class_id in WHEELED_IDS
                and wheeled_silhouette_angle_max_deg is not None
                and geometry["lower_axis_angle_from_vertical_deg"]
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
                # The production contract still leaves the automatic selector open.
                selector = "deferred_default_footprint"
            if candidate.any():
                final |= candidate
                if strategy == "selected_silhouette":
                    silhouette_selected |= candidate
                    wheeled_silhouette_selected |= candidate
                else:
                    footprint_selected |= candidate
                baseline_for_natural |= candidate
            record = {
                **_clean(obj), **metadata, **geometry,
                "class_name": id2label.get(class_id, str(class_id)),
                "kind": "wheeled_unmanned_obstacle",
                "strategy": strategy,
                "wheeled_selector_state": selector,
                "wheeled_silhouette_angle_max_deg": wheeled_silhouette_angle_max_deg,
                "scoped_closure_pixels": 0,
                "selected_pixels": int(candidate.sum()),
            }
        else:
            candidate = _lower_silhouette(source)
            accepted, support_meta = _grounded(candidate, source, shared.source_walkway)
            if accepted:
                final |= candidate
                silhouette_selected |= candidate
                baseline_for_natural |= candidate
            record = {
                **_clean(obj), **support_meta,
                "class_name": id2label.get(class_id, str(class_id)),
                "kind": "other_grounded_obstacle",
                "strategy": "lower_19_percent_silhouette",
                "silhouette_fraction": LOWER_SILHOUETTE_FRACTION,
                "accepted": bool(accepted),
                "selected_pixels": int(candidate.sum()) if accepted else 0,
                "rejection_reason": None if accepted else "no_walkway_ground_support",
            }
        records.append(record)

    # Preserve V2.1 natural grounded-footprint evidence without ever invoking
    # the natural-source bridge/restoration path.
    natural_fp, natural_source, _, _, natural_rows, _ = legacy.grounded.universal_hole_candidates(
        labels,
        shared.repaired,
        shared.structure_copy,
        shared.source_walkway,
        baseline_for_natural,
    )
    recognized |= natural_source
    final |= natural_fp
    footprint_selected |= natural_fp
    for row in natural_rows:
        row = dict(row)
        row.update({
            "kind": "v2_1_natural_grounded_footprint",
            "strategy": "footprint",
            "class_name": id2label.get(int(row.get("class_id", -1)), str(row.get("class_id"))),
            "selected_pixels": int(row.get("footprint_pixels", 0)) if row.get("accepted") else 0,
            "natural_restoration_applied": False,
        })
        records.append(row)

    for obj in all_objects:
        if obj["class_id"] in TRANSIENT_IDS:
            recognized |= obj["_mask"]
            records.append({
                **_clean(obj),
                "class_name": id2label.get(obj["class_id"], str(obj["class_id"])),
                "kind": "transient_occluder",
                "strategy": "transient_rows_only",
                "accepted": False,
                "selected_pixels": 0,
            })

    for hazard in _components(labels, hazard_ids):
        recognized |= hazard["_mask"]
        records.append({
            **_clean(hazard),
            "class_name": id2label.get(hazard["class_id"], str(hazard["class_id"])),
            "kind": "semantic_hazard",
            "strategy": "full_semantic_hazard",
            "accepted": True,
            "selected_pixels": int(hazard["_mask"].sum()),
        })

    # Different physical objects can overlap in image projection. That is a
    # diagnostic, not an inconsistent strategy assignment and never authorizes
    # merging their instances.
    strategy_overlap_pixels = int((footprint_selected & silhouette_selected).sum())
    violations = []
    if any(r.get("kind") == "cycle_gap_closure" for r in records):
        violations.append("forbidden_global_cycle_closure")
    if any(r.get("natural_restoration_applied") for r in records):
        violations.append("natural_restoration_leak")

    return {
        "recognized_mask": recognized,
        "transient_mask": transient_source,
        "hazard_mask": np.isin(labels, tuple(sorted(hazard_ids))),
        "footprint_mask": footprint_selected,
        "silhouette_mask": silhouette_selected,
        "wheeled_silhouette_mask": wheeled_silhouette_selected,
        "final_mask": final,
        "objects": records,
        "yolo": yolo,
        "violations": violations,
        "strategy_overlap_pixels": strategy_overlap_pixels,
        "policy": {
            "ground_animal_transient": True,
            "wheeled_default": "footprint",
            "wheeled_exception_selector": (
                "lower_axis_angle_probe"
                if wheeled_silhouette_angle_max_deg is not None else "deferred_not_integrated"
            ),
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

