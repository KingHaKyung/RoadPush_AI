from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = get_settings().project_root
OUT = HERE / "vehicle_yolo_trigger_audit_40_v1"
MANIFEST = HERE / "model_audit_40" / "audit_manifest.json"
CAR_ID = 55



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


def analyse_car_components(labels: np.ndarray, walkway: np.ndarray) -> list[dict]:
    binary = (labels == CAR_ID).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    records: list[dict] = []
    minimum_area = max(20, round(labels.size * 0.00015))

    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if area < minimum_area:
            continue
        component = component_labels == component_id
        estimated = policy.r3.class_aware_component_footprint(component, CAR_ID)
        if estimated is None:
            continue
        candidate, _ = estimated
        effective, support_pixels, support_fraction = policy.ground_support(component, candidate, walkway)
        minimum_support = max(6, round(candidate.sum() * 0.04))
        source_supported, source_total, source_ratio, bottom_truncated = policy.source_contact_support(
            component, walkway
        )
        source_contact_ok = bottom_truncated or (source_supported >= 2 and source_ratio >= 0.35)
        walkway_owned = support_pixels >= minimum_support and source_contact_ok

        _, evidence = geometry.analyse_component(component, [x, y, width, height], labels.shape[0])
        anomaly = evidence["decision"] == "split_candidate"
        trigger = walkway_owned and anomaly
        if not walkway_owned:
            decision = "skip_off_walkway"
        elif trigger:
            decision = "call_yolo"
        else:
            decision = "keep_mapillary"

        evidence.pop("profile", None)
        records.append({
            "component_id": component_id,
            "source_bbox": [x, y, width, height],
            "source_pixels": int(area),
            "walkway_owned": bool(walkway_owned),
            "ground_support_pixels": int(support_pixels),
            "ground_support_fraction": float(support_fraction),
            "minimum_ground_support_pixels": int(minimum_support),
            "source_contact_supported_columns": int(source_supported),
            "source_contact_total_columns": int(source_total),
            "source_contact_support_fraction": float(source_ratio),
            "source_bottom_truncated": bool(bottom_truncated),
            "geometry_anomaly": bool(anomaly),
            "trigger_yolo": bool(trigger),
            "decision": decision,
            "geometry": evidence,
            "_component": component,
            "_candidate": candidate,
            "_effective": effective,
        })
    return records


def serializable(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def process(item: dict) -> dict:
    image, labels = load_inputs(item)
    walkway = selected_walkway(labels)
    records = analyse_car_components(labels, walkway)
    trigger_records = [record for record in records if record["trigger_yolo"]]

    raw_view = paint(image, labels == CAR_ID, (230, 65, 75), 0.65)
    ownership_view = image.copy()
    decision_view = image.copy()
    for record in records:
        component = record["_component"]
        ownership_color = (45, 205, 120) if record["walkway_owned"] else (150, 150, 150)
        ownership_view = paint(ownership_view, component, ownership_color, 0.72)
        x, y, width, height = record["source_bbox"]
        if record["trigger_yolo"]:
            color = (30, 225, 255)
            thickness = 4
        elif record["walkway_owned"]:
            color = (80, 190, 255)
            thickness = 3
        else:
            color = (165, 165, 165)
            thickness = 2
        cv2.rectangle(decision_view, (x, y), (x + width - 1, y + height - 1), color, thickness)
        if record["trigger_yolo"] and "split_x" in record["geometry"]:
            split_x = int(record["geometry"]["split_x"])
            cv2.line(decision_view, (split_x, y), (split_x, y + height - 1), color, 3)

    artifact = OUT / "artifacts" / item["sample_id"]
    artifact.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [image, raw_view, ownership_view, decision_view]
    titles = [
        "1 Original",
        "2 Cached Mapillary Car pixels",
        "3 V2 ownership (green=walkway, gray=off-walkway)",
        "4 YOLO trigger (cyan=call, blue=keep, gray=skip)",
    ]
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"{item['sample_id']} | automatic YOLO trigger audit | "
        f"{'CALL YOLO' if trigger_records else 'NO YOLO'}",
        fontsize=16,
    )
    fig.savefig(artifact / "summary.png", dpi=115)
    plt.close(fig)

    payload = {
        "sample_id": item["sample_id"],
        "order": item["order"],
        "split": item["split"],
        "trigger_yolo": bool(trigger_records),
        "trigger_component_count": len(trigger_records),
        "car_component_count": len(records),
        "walkway_owned_car_component_count": sum(record["walkway_owned"] for record in records),
        "components": [serializable(record) for record in records],
        "new_mapillary_inference": False,
        "new_yolo_inference": False,
        "mask_or_width_modified": False,
        "fallback": "keep_original_mapillary_parent_mask",
    }
    (artifact / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_review(results: list[dict]) -> None:
    sections = []
    for result in results:
        status = "CALL YOLO" if result["trigger_yolo"] else "NO YOLO"
        css_class = "call" if result["trigger_yolo"] else "skip"
        sample_id = result["sample_id"]
        sections.append(
            f'<section class="{css_class}"><h2>{result["order"]:02d}. {sample_id} '
            f'<span>{status}</span></h2><p>Car components {result["car_component_count"]} · '
            f'walkway-owned {result["walkway_owned_car_component_count"]} · '
            f'trigger components {result["trigger_component_count"]}</p>'
            f'<a href="artifacts/{sample_id}/result.json">result.json</a>'
            f'<img src="artifacts/{sample_id}/summary.png"></section>'
        )
    call_count = sum(result["trigger_yolo"] for result in results)
    html = (
        '<!doctype html><meta charset="utf-8"><title>Vehicle YOLO trigger audit 40 v1</title>'
        '<style>body{font:16px system-ui;background:#101722;color:#eef;max-width:1500px;margin:auto}'
        'section{background:#192331;margin:22px;padding:18px;border-radius:12px}'
        'section.call{border:3px solid #34dbe8}section.call span{color:#48e9ef}'
        'section.skip span{color:#aeb8c4}img{width:100%;height:auto}a{color:#8df}</style>'
        '<h1>Vehicle YOLO trigger audit 40 v1</h1>'
        f'<p>YOLO calls proposed: {call_count}/40 ({call_count / 40:.1%}). '
        'This is a cached semantic-map audit only: no YOLO inference, no mask split, no width change.</p>'
        '<p>Order: V2 walkway ownership filter → merged-depth anomaly test → conditional YOLO call. '
        'If YOLO later fails to confirm two vehicles, the original Mapillary parent mask remains the fallback.</p>'
        + ''.join(sections)
    )
    (OUT / "review.html").write_text(html, encoding="utf-8")


def write_tables(results: list[dict]) -> None:
    with (OUT / "trigger_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "order", "sample_id", "split", "trigger_yolo", "trigger_component_count",
            "car_component_count", "walkway_owned_car_component_count",
        ])
        writer.writeheader()
        writer.writerows({key: result[key] for key in writer.fieldnames} for result in results)
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    results = [process(item) for item in items]
    write_review(results)
    write_tables(results)

    triggered = [result["sample_id"] for result in results if result["trigger_yolo"]]
    validation = {
        "expected_samples": 40,
        "processed_samples": len(results),
        "all_artifacts_present": all(
            (OUT / "artifacts" / result["sample_id"] / "summary.png").exists() for result in results
        ),
        "v2_walkway_ownership_filter_applied_before_trigger": True,
        "geometry_used_as_trigger_only": True,
        "new_mapillary_inference": False,
        "new_yolo_inference": False,
        "mask_or_width_modified": False,
        "existing_outputs_overwritten": False,
        "trigger_count": len(triggered),
        "trigger_rate": len(triggered) / max(1, len(results)),
        "triggered_samples": triggered,
    }
    (OUT / "completion_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(
        "# Vehicle YOLO trigger audit 40 v1\n\n"
        "- Cached Mapillary argmax maps only; no new segmentation inference\n"
        "- V2 walkway ownership is checked before any geometry anomaly\n"
        "- Geometry is used only to decide whether YOLO should be called\n"
        "- No mask split and no clearance-width modification in this audit\n"
        "- Safety fallback: retain the original Mapillary parent component\n"
        f"- Proposed YOLO calls: {len(triggered)}/40\n"
        f"- Triggered samples: {', '.join(triggered) if triggered else 'none'}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
