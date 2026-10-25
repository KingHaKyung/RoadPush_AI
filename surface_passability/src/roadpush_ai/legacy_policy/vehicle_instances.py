from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = get_settings().project_root
OUT = HERE / "yolo_vehicle_instance_probe_2_v1"
MANIFEST = HERE / "model_audit_40" / "audit_manifest.json"
TRIGGER_ROOT = HERE / "vehicle_yolo_trigger_audit_40_v1"
WEIGHTS = ROOT / "models" / "yolo11n-seg.pt"
SAMPLES = ("IMG_4548", "commons_04")

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "work" / "cache" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / "work" / "cache" / "matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


CAR_ID = 55
VEHICLE_NAMES = {"car", "truck", "bus"}
COLORS = [(245, 85, 75), (55, 175, 255), (175, 95, 245), (250, 190, 45)]



from roadpush_ai.legacy_policy import clearance_policy as policy
from roadpush_ai.legacy_policy import vehicle_geometry as geometry


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.7) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def load_inputs(item: dict) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(ROOT / item["relative_path"]) as source:
        size = geometry.fit(source.width, source.height)
        image = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    native = cv2.imread(str(ROOT / item["class_map_path"]), cv2.IMREAD_UNCHANGED)
    if native is None:
        raise FileNotFoundError(item["class_map_path"])
    labels = cv2.resize(native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return image, labels


def selected_walkway(labels: np.ndarray) -> np.ndarray:
    repaired, _, _ = policy.probe.v2.repair_surface(labels)
    _, path = policy.probe.outer.route_path(repaired)
    walkway = policy.probe.outer.selected_component(repaired, path)
    return walkway if walkway.any() else repaired


def rasterize_polygon(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.new("L", (shape[1], shape[0]), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon([tuple(map(float, point)) for point in points], fill=255)
    return np.asarray(mask) > 0


def infer_instances(model: YOLO, image: np.ndarray) -> tuple[list[dict], dict]:
    result = model.predict(
        image,
        device=os.environ.get("ROADPUSH_YOLO_DEVICE", "0"),
        imgsz=640,
        conf=0.25,
        verbose=False,
    )[0]
    instances: list[dict] = []
    if result.boxes is None:
        return instances, {"box_count": 0, "mask_count": 0}
    mask_polygons = result.masks.xy if result.masks is not None else []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        name = str(model.names[class_id])
        if name not in VEHICLE_NAMES:
            continue
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        mask = np.zeros(image.shape[:2], dtype=bool)
        if index < len(mask_polygons):
            mask = rasterize_polygon(mask_polygons[index], image.shape[:2])
        instances.append({
            "detection_index": index,
            "class_id": class_id,
            "class_name": name,
            "confidence": float(box.conf.item()),
            "xyxy": xyxy,
            "mask_pixels": int(mask.sum()),
            "_mask": mask,
        })
    return instances, {
        "box_count": int(len(result.boxes)),
        "mask_count": 0 if result.masks is None else int(len(result.masks.xy)),
    }


def trigger_component(labels: np.ndarray, sample_id: str) -> tuple[np.ndarray, dict]:
    trigger = json.loads((TRIGGER_ROOT / "artifacts" / sample_id / "result.json").read_text(encoding="utf-8"))
    record = next(component for component in trigger["components"] if component["trigger_yolo"])
    count, component_labels, _, _ = cv2.connectedComponentsWithStats((labels == CAR_ID).astype(np.uint8), 8)
    component_id = int(record["component_id"])
    if component_id <= 0 or component_id >= count:
        raise RuntimeError(f"Invalid cached component id {component_id} for {sample_id}")
    return component_labels == component_id, record


def match_instances(component: np.ndarray, instances: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    audit: list[dict] = []
    component_pixels = max(1, int(component.sum()))
    for instance in instances:
        mask = instance["_mask"]
        intersection = int((mask & component).sum())
        yolo_fraction = intersection / max(1, int(mask.sum()))
        component_fraction = intersection / component_pixels
        matched = intersection >= 25 and (yolo_fraction >= 0.05 or component_fraction >= 0.01)
        row = {
            **{key: value for key, value in instance.items() if not key.startswith("_")},
            "mapillary_intersection_pixels": intersection,
            "yolo_mask_overlap_fraction": yolo_fraction,
            "mapillary_component_overlap_fraction": component_fraction,
            "matched_to_trigger_component": matched,
        }
        audit.append(row)
        if matched:
            accepted.append(instance)
    return accepted, audit


def partition_mapillary(component: np.ndarray, instances: list[dict]) -> list[np.ndarray]:
    distances = []
    for instance in instances:
        seed = instance["_mask"] & component
        outside_seed = (~seed).astype(np.uint8)
        distances.append(cv2.distanceTransform(outside_seed, cv2.DIST_L2, 3))
    winner = np.argmin(np.stack(distances, axis=0), axis=0)
    return [component & (winner == index) for index in range(len(instances))]


def owned_footprint(component: np.ndarray, walkway: np.ndarray) -> tuple[np.ndarray, dict]:
    estimated = policy.r3.class_aware_component_footprint(component, CAR_ID)
    if estimated is None:
        return np.zeros_like(component), {"accepted": False, "reason": "no_footprint"}
    candidate, metadata = estimated
    effective, support_pixels, support_fraction = policy.ground_support(component, candidate, walkway)
    minimum_support = max(6, round(candidate.sum() * 0.04))
    source_supported, source_total, source_ratio, bottom_truncated = policy.source_contact_support(
        component, walkway
    )
    source_ok = bottom_truncated or (source_supported >= 2 and source_ratio >= 0.35)
    accepted = support_pixels >= minimum_support and source_ok
    return (effective if accepted else np.zeros_like(component)), {
        **metadata,
        "accepted": bool(accepted),
        "ground_support_pixels": int(support_pixels),
        "ground_support_fraction": float(support_fraction),
        "minimum_ground_support_pixels": int(minimum_support),
        "source_contact_supported_columns": int(source_supported),
        "source_contact_total_columns": int(source_total),
        "source_contact_support_fraction": float(source_ratio),
        "source_bottom_truncated": bool(bottom_truncated),
        "reason": None if accepted else ("off_walkway_source_contact" if not source_ok else "no_ground_support"),
    }


def process(model: YOLO, item: dict) -> dict:
    image, labels = load_inputs(item)
    walkway = selected_walkway(labels)
    parent, trigger_record = trigger_component(labels, item["sample_id"])
    detections, inference = infer_instances(model, image)
    matched, detection_audit = match_instances(parent, detections)
    split_confirmed = len(matched) >= 2 and all(instance["_mask"].any() for instance in matched)

    parent_footprint, parent_meta = owned_footprint(parent, walkway)
    children: list[np.ndarray] = []
    child_rows: list[dict] = []
    split_footprint = parent_footprint.copy()
    if split_confirmed:
        children = partition_mapillary(parent, matched)
        split_footprint = np.zeros_like(parent)
        for index, (child, instance) in enumerate(zip(children, matched)):
            footprint, metadata = owned_footprint(child, walkway)
            split_footprint |= footprint
            child_rows.append({
                "instance_index": index,
                "yolo_detection_index": instance["detection_index"],
                "yolo_class": instance["class_name"],
                "yolo_confidence": instance["confidence"],
                "assigned_mapillary_pixels": int(child.sum()),
                "footprint_pixels": int(footprint.sum()),
                **metadata,
            })

    yolo_view = image.copy()
    for index, instance in enumerate(detections):
        yolo_view = paint(yolo_view, instance["_mask"], COLORS[index % len(COLORS)], 0.7)
    matched_view = image.copy()
    for index, instance in enumerate(matched):
        matched_view = paint(matched_view, instance["_mask"], COLORS[index % len(COLORS)], 0.72)
    partition_view = image.copy()
    for index, child in enumerate(children):
        partition_view = paint(partition_view, child, COLORS[index % len(COLORS)], 0.72)
    if not children:
        partition_view = paint(partition_view, parent, (160, 160, 160), 0.65)
    parent_view = paint(image, parent_footprint, (255, 190, 35), 0.82)
    split_view = image.copy()
    if split_confirmed:
        for index, (child, row) in enumerate(zip(children, child_rows)):
            footprint, _ = owned_footprint(child, walkway)
            split_view = paint(split_view, footprint, COLORS[index % len(COLORS)], 0.82)
    else:
        split_view = paint(split_view, split_footprint, (160, 160, 160), 0.75)

    artifact = OUT / "artifacts" / item["sample_id"]
    artifact.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = [image, yolo_view, matched_view, partition_view, parent_view, split_view]
    titles = [
        "1 Original",
        "2 YOLO11n-seg vehicle instances",
        "3 Instances matched to triggered Mapillary component",
        "4 Mapillary Car pixels assigned per YOLO instance",
        "5 Existing parent footprint",
        "6 Per-instance Mapillary footprints",
    ]
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    status = "split_confirmed" if split_confirmed else "fallback_parent"
    fig.suptitle(
        f"{item['sample_id']} | YOLO-guided Mapillary instance split | {status} | matched {len(matched)}",
        fontsize=16,
    )
    fig.savefig(artifact / "summary.png", dpi=115)
    plt.close(fig)

    payload = {
        "sample_id": item["sample_id"],
        "status": status,
        "model": "Ultralytics YOLO11n-seg COCO",
        "weights": str(WEIGHTS.relative_to(ROOT)).replace("\\", "/"),
        "imgsz": 640,
        "confidence_threshold": 0.25,
        "device": "cuda:0",
        "trigger_component": trigger_record,
        "inference": inference,
        "vehicle_detections": detection_audit,
        "matched_vehicle_instances": len(matched),
        "parent_mapillary_pixels": int(parent.sum()),
        "assigned_mapillary_pixels": [int(child.sum()) for child in children],
        "assignment_conserves_parent_pixels": bool(
            not children or sum(int(child.sum()) for child in children) == int(parent.sum())
        ),
        "parent_footprint": parent_meta,
        "per_instance_footprints": child_rows,
        "parent_footprint_pixels": int(parent_footprint.sum()),
        "split_footprint_union_pixels": int(split_footprint.sum()),
        "width_basis": "Mapillary Car pixels partitioned by YOLO instance masks",
        "fallback": "keep_parent_mapillary_component_when_fewer_than_two_instances_match",
        "v2_modified": False,
    }
    (artifact / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in manifest["samples"]}
    model = YOLO(WEIGHTS)
    results = [process(model, by_id[sample_id]) for sample_id in SAMPLES]

    sections = []
    for result in results:
        sample_id = result["sample_id"]
        sections.append(
            f'<section><h2>{sample_id} <span>{result["status"]}</span></h2>'
            f'<p>matched YOLO vehicles {result["matched_vehicle_instances"]} · '
            f'Mapillary pixels {result["parent_mapillary_pixels"]} → '
            f'{result["assigned_mapillary_pixels"]} · footprint '
            f'{result["parent_footprint_pixels"]} → {result["split_footprint_union_pixels"]}</p>'
            f'<a href="artifacts/{sample_id}/result.json">result.json</a>'
            f'<img src="artifacts/{sample_id}/summary.png"></section>'
        )
    html = (
        '<!doctype html><meta charset="utf-8"><title>YOLO vehicle instance probe 2 v1</title>'
        '<style>body{font:16px system-ui;background:#101722;color:#eef;max-width:1500px;margin:auto}'
        'section{background:#192331;margin:22px;padding:18px;border-radius:12px}'
        'span{color:#55e5dc}img{width:100%;height:auto}a{color:#8df}</style>'
        '<h1>YOLO-guided Mapillary vehicle split: two triggered samples</h1>'
        '<p>YOLO supplies instance identity only. Final footprint geometry is recalculated from partitioned '
        'Mapillary Car pixels. Fewer than two matched instances causes a parent-mask fallback.</p>'
        + ''.join(sections)
    )
    (OUT / "review.html").write_text(html, encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "expected_samples": len(SAMPLES),
        "processed_samples": len(results),
        "all_artifacts_present": all(
            (OUT / "artifacts" / result["sample_id"] / "summary.png").exists() for result in results
        ),
        "new_yolo_inference": True,
        "mapillary_inference_reused": True,
        "width_basis_is_mapillary_pixels": True,
        "existing_outputs_overwritten": False,
        "v2_modified": False,
        "status_by_sample": {result["sample_id"]: result["status"] for result in results},
    }
    (OUT / "completion_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
