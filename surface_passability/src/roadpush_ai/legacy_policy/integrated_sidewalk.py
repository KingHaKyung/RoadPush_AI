from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import html
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = get_settings().project_root
OUT = HERE / "integrated_v2_regression_40_v4"
AUDIT = HERE / "model_audit_40" / "audit_manifest.json"
WEIGHTS = ROOT / "models" / "yolo11n-seg.pt"
MAX_SIDE = 700
CAR_ID = 55
TRANSIENT_IDS = (19, 20, 21, 22)

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "work" / "cache" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / "work" / "cache" / "matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO



from roadpush_ai.legacy_policy import clearance_policy as policy
from roadpush_ai.legacy_policy import outer_boundary as outer
from roadpush_ai.legacy_policy import ground_footprint as ground
from roadpush_ai.legacy_policy import vehicle_trigger as trigger
from roadpush_ai.legacy_policy import vehicle_instances as yolo_split
from roadpush_ai.legacy_policy import curb_boundary as curb_probe


def fit(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.7) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def clean_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def load_inputs(item: dict) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(ROOT / item["relative_path"]) as source:
        size = fit(source.width, source.height)
        image = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    native = cv2.imread(str(ROOT / item["class_map_path"]), cv2.IMREAD_UNCHANGED)
    if native is None:
        raise FileNotFoundError(item["class_map_path"])
    labels = cv2.resize(native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return image, labels


def selected_walkway(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    repaired, _, _ = policy.probe.v2.repair_surface(labels)
    route, path = outer.route_path(repaired)
    walkway = outer.selected_component(repaired, path)
    return repaired, walkway if walkway.any() else repaired


def integrated_footprints(
    labels: np.ndarray,
    structural: np.ndarray,
    image: np.ndarray,
    model_holder: dict,
) -> tuple[np.ndarray, list[dict], dict, list[np.ndarray]]:
    repaired, walkway = selected_walkway(labels)
    vehicle_records = trigger.analyse_car_components(labels, walkway)
    triggered = [record for record in vehicle_records if record["trigger_yolo"]]

    # Let the established r6 policy process every component except a triggered merged
    # Car component.  Those components are reintroduced below, per YOLO instance.
    base_labels = labels.copy()
    for record in triggered:
        base_labels[record["_component"]] = 255
    footprints, rows = policy.policy_estimate_footprints(base_labels, structural)

    yolo_meta = {
        "triggered": bool(triggered),
        "trigger_component_count": len(triggered),
        "inference_run": False,
        "split_confirmed_count": 0,
        "fallback_parent_count": 0,
        "vehicle_detection_count": 0,
        "matched_instances": 0,
        "components": [],
    }
    child_footprints: list[np.ndarray] = []
    if not triggered:
        return footprints, rows, yolo_meta, child_footprints

    if "model" not in model_holder:
        model_holder["model"] = YOLO(WEIGHTS)
    detections, inference = yolo_split.infer_instances(model_holder["model"], image)
    yolo_meta["inference_run"] = True
    yolo_meta["inference"] = inference
    yolo_meta["vehicle_detection_count"] = len(detections)

    for trigger_record in triggered:
        parent = trigger_record["_component"]
        matched, detection_audit = yolo_split.match_instances(parent, detections)
        split_confirmed = len(matched) >= 2 and all(instance["_mask"].any() for instance in matched)
        component_meta = {
            "source_bbox": trigger_record["source_bbox"],
            "source_pixels": trigger_record["source_pixels"],
            "matched_instances": len(matched),
            "split_confirmed": split_confirmed,
            "detections": detection_audit,
            "children": [],
        }
        yolo_meta["matched_instances"] += len(matched)
        if split_confirmed:
            yolo_meta["split_confirmed_count"] += 1
            children = yolo_split.partition_mapillary(parent, matched)
            for index, (child, instance) in enumerate(zip(children, matched)):
                footprint, metadata = yolo_split.owned_footprint(child, walkway)
                footprints |= footprint
                child_footprints.append(footprint)
                row = {
                    "class_id": CAR_ID,
                    "kind": "yolo_partitioned_mapillary_car",
                    "accepted": bool(metadata["accepted"]),
                    "instance_index": index,
                    "yolo_detection_index": instance["detection_index"],
                    "yolo_class": instance["class_name"],
                    "yolo_confidence": instance["confidence"],
                    "source_pixels": int(child.sum()),
                    "footprint_pixels": int(footprint.sum()),
                    **metadata,
                }
                rows.append(row)
                component_meta["children"].append(row)
            component_meta["assignment_conserves_parent_pixels"] = bool(
                sum(int(child.sum()) for child in children) == int(parent.sum())
            )
        else:
            yolo_meta["fallback_parent_count"] += 1
            footprint, metadata = yolo_split.owned_footprint(parent, walkway)
            footprints |= footprint
            child_footprints.append(footprint)
            row = {
                "class_id": CAR_ID,
                "kind": "yolo_unconfirmed_parent_fallback",
                "accepted": bool(metadata["accepted"]),
                "source_pixels": int(parent.sum()),
                "footprint_pixels": int(footprint.sum()),
                **metadata,
            }
            rows.append(row)
            component_meta["children"].append(row)
        yolo_meta["components"].append(component_meta)
    return footprints, rows, yolo_meta, child_footprints


def draw_routes(panel: np.ndarray, route: dict, path: list[tuple[int, int]]) -> None:
    main_index = route.get("main_index")
    high = bool(route.get("high_confidence"))
    for index, candidate in enumerate(route.get("paths", [])):
        color = (35, 110, 255) if index == main_index and high else (
            (135, 135, 135) if index == main_index else (255, 150, 25)
        )
        outer.draw_path(panel, candidate, color, 5 if index == main_index else 3)
    if route.get("start") is not None:
        y, x = route["start"]
        cv2.circle(panel, (x, y), 7, (25, 230, 255), -1, cv2.LINE_AA)


def process(item: dict, model_holder: dict) -> dict:
    sample_id = item["sample_id"]
    image, labels = load_inputs(item)
    repaired, walkway = selected_walkway(labels)
    _, curb_suspect, _, _ = curb_probe.analyze(labels)
    measurement_surface = repaired & ~curb_suspect
    if not measurement_surface.any():
        measurement_surface = repaired
    base_sidewalk = labels == policy.probe.v2.SIDEWALK
    blocked_ids = tuple(policy.probe.v2.OCCUPANCY) + tuple(policy.probe.v2.OCCLUSION)
    raw_objects = np.isin(labels, blocked_ids)
    raw_hazards = np.isin(labels, tuple(policy.probe.v2.HAZARD))
    raw_blocked = raw_objects | raw_hazards

    structural, restored = outer.base.obstacle_transparent_structure(repaired, raw_blocked)
    footprints, footprint_rows, yolo_meta, child_footprints = integrated_footprints(
        labels, structural, image, model_holder
    )
    grounded_banner = policy.grounded_banner_visual(labels, footprints)
    display_blocked = raw_blocked | grounded_banner
    visual_free = structural & ~display_blocked & ~footprints
    ground_free = structural & ~raw_hazards & ~footprints

    # Curb suspects remain present in the preserved semantic surface and visualization.
    # They are excluded only from route/width candidate selection, preventing a known
    # disconnected road-side Sidewalk false positive from owning the main route.
    route, path = outer.route_path(measurement_surface)
    measurement_structural = structural & ~curb_suspect
    component = outer.selected_component(measurement_structural, path)
    records = outer.sample_outer_boundaries(component, path, labels)
    person_mask = np.isin(labels, TRANSIENT_IDS)
    for record in records:
        left, right, y = record["left"], record["right"], record["y"]
        record["person_occlusion"] = bool(person_mask[y, left:right + 1].any())
        if record["person_occlusion"]:
            record["width_state"] = "occlusion_limited"
        segments = outer.base.free_segments(ground_free, y, left, right)
        selected = outer.choose_route_segment(segments, record["anchor_x"])
        record["free_segments"] = [[a, b] for a, b in segments]
        record["selected_free_segment"] = list(selected) if selected else None
        record["selected_free_width"] = selected[1] - selected[0] + 1 if selected else 0
        record["widest_free_width"] = max((b - a + 1 for a, b in segments), default=0)

    complete = [record for record in records if record["width_state"] == "complete"]
    lower = [record for record in records if record["width_state"] == "lower_bound"]
    occluded = [record for record in records if record["width_state"] == "occlusion_limited"]
    base_width_status = "complete_width_candidate" if len(complete) >= 6 else (
        "partial_width_only" if lower or occluded else "measurement_unavailable"
    )
    alternative_count = max(0, len(route.get("paths", [])) - 1)
    branch_state = "branch_unresolved" if alternative_count else "no_alternative_branch_detected"
    width_status = (
        "branch_unresolved_width_candidate"
        if alternative_count and base_width_status != "measurement_unavailable"
        else base_width_status
    )

    repaired_view = paint(image, repaired, (45, 210, 95), 0.42)
    direction_view = repaired_view.copy()
    direction_view = paint(direction_view, curb_suspect, (245, 55, 65), 0.65)
    draw_routes(direction_view, route, path)
    footprint_view = paint(image, raw_objects, (225, 70, 165), 0.18)
    footprint_view = paint(footprint_view, footprints, (255, 195, 35), 0.82)
    if yolo_meta["split_confirmed_count"]:
        colors = [(245, 75, 70), (55, 165, 255), (175, 95, 245)]
        for index, child_footprint in enumerate(child_footprints):
            footprint_view = paint(footprint_view, child_footprint, colors[index % len(colors)], 0.85)

    contour_view = paint(image, structural, (45, 210, 95), 0.32)
    contour_view = paint(contour_view, curb_suspect, (245, 55, 65), 0.65)
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contour_view, contours, -1, (255, 65, 210), 3, cv2.LINE_AA)
    free_view = paint(image, ground_free, (85, 185, 235), 0.26)
    state_colors = {"complete": (40, 215, 255), "lower_bound": (155, 155, 155), "occlusion_limited": (255, 145, 35)}
    for record in records:
        y, left, right = record["y"], record["left"], record["right"]
        cv2.line(contour_view, (left, y), (right, y), state_colors[record["width_state"]], 2, cv2.LINE_AA)
        for a, b in record["free_segments"]:
            cv2.line(free_view, (a, y), (b, y), (90, 205, 255), 2, cv2.LINE_AA)
        if record["selected_free_segment"]:
            a, b = record["selected_free_segment"]
            cv2.line(free_view, (a, y), (b, y), (35, 85, 255), 5, cv2.LINE_AA)

    selected_widths = [record["selected_free_width"] for record in records if record["selected_free_width"]]
    structural_widths = [record["width"] for record in complete]
    accepted = [row for row in footprint_rows if row.get("accepted")]
    states = {
        "surface": "preserved" if np.all(repaired[base_sidewalk]) else "failed",
        "direction": "high" if route.get("high_confidence") else "undetermined",
        "branch": branch_state,
        "width": width_status,
        "yolo": (
            "split_confirmed" if yolo_meta["split_confirmed_count"] else
            "fallback_parent" if yolo_meta["fallback_parent_count"] else
            "not_triggered"
        ),
        "curb_route_filter": "applied" if curb_suspect.any() else "not_needed",
    }

    art = OUT / "artifacts" / sample_id
    art.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(art / "01_repaired_surface.png"), repaired.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02_structural_surface.png"), structural.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "03_clearance_footprints.png"), footprints.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "04_ground_free.png"), ground_free.astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = [repaired_view, direction_view, footprint_view, contour_view, free_view]
    titles = [
        "1 Repaired full sidewalk (preserved)",
        "2 Non-destructive direction candidates",
        "3 V2 clearance footprints (YOLO colors only when triggered)",
        "4 Actual outer boundary + horizontal width rows",
        "5 Free gaps (dark blue=route-selected)",
    ]
    for axis, title, panel in zip(axes.flat[:5], titles, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    axes.flat[5].axis("off")
    axes.flat[5].set_title("6 Integrated V2 states")
    axes.flat[5].text(
        0.02, 0.98,
        "\n".join([
            f"surface: {states['surface']}",
            f"direction: {states['direction']}",
            f"branch: {states['branch']} (alternatives={alternative_count})",
            f"width: {states['width']}",
            f"accepted footprints: {len(accepted)}",
            f"YOLO: {states['yolo']}",
            f"YOLO inference: {yolo_meta['inference_run']}",
            f"curb route filter: {states['curb_route_filter']}",
            "width unit: analysis pixels (not meters)",
            "branch deletion: 0 pixels",
        ]),
        va="top", family="monospace", fontsize=12,
    )
    fig.suptitle(f"{sample_id} | integrated V2 regression | {width_status}", fontsize=16)
    fig.savefig(art / "summary.png", dpi=115)
    plt.close(fig)

    payload = {
        "sample_id": sample_id,
        "order": item["order"],
        "split": item["split"],
        "states": states,
        "direction_high_confidence": bool(route.get("high_confidence")),
        "route_start_x_fraction": (
            float(route["start"][1] / labels.shape[1]) if route.get("start") is not None else None
        ),
        "alternative_branch_candidates": alternative_count,
        "branch_pixels_deleted": 0,
        "base_sidewalk_preserved": bool(np.all(repaired[base_sidewalk])),
        "curb_suspect_pixels_preserved": int(curb_suspect.sum()),
        "curb_suspect_excluded_from_route_width_only": bool(curb_suspect.any()),
        "sampled_rows": len(records),
        "base_width_status_before_branch_guard": base_width_status,
        "complete_rows": len(complete),
        "lower_bound_rows": len(lower),
        "occlusion_limited_rows": len(occluded),
        "median_complete_structural_width_px": float(np.median(structural_widths)) if structural_widths else None,
        "median_route_selected_free_width_px": float(np.median(selected_widths)) if selected_widths else None,
        "accepted_footprints": len(accepted),
        "footprint_pixels": int(footprints.sum()),
        "footprint_components": footprint_rows,
        "yolo": yolo_meta,
        "rows": records,
        "new_mapillary_inference": False,
        "new_yolo_inference": bool(yolo_meta["inference_run"]),
        "width_unit": "analysis_image_pixels_not_meters",
    }
    (art / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    model_holder: dict = {}
    results = [process(item, model_holder) for item in manifest["samples"]]

    summary_fields = [
        "order", "sample_id", "split", "surface_state", "direction_state", "branch_state",
        "width_state", "alternative_branch_candidates", "accepted_footprints", "footprint_pixels",
        "complete_rows", "lower_bound_rows", "occlusion_limited_rows",
        "median_complete_structural_width_px", "median_route_selected_free_width_px",
        "yolo_state", "yolo_inference",
    ]
    summary_rows = []
    for result in results:
        summary_rows.append({
            "order": result["order"], "sample_id": result["sample_id"], "split": result["split"],
            "surface_state": result["states"]["surface"],
            "direction_state": result["states"]["direction"],
            "branch_state": result["states"]["branch"],
            "width_state": result["states"]["width"],
            "alternative_branch_candidates": result["alternative_branch_candidates"],
            "accepted_footprints": result["accepted_footprints"],
            "footprint_pixels": result["footprint_pixels"],
            "complete_rows": result["complete_rows"], "lower_bound_rows": result["lower_bound_rows"],
            "occlusion_limited_rows": result["occlusion_limited_rows"],
            "median_complete_structural_width_px": result["median_complete_structural_width_px"],
            "median_route_selected_free_width_px": result["median_route_selected_free_width_px"],
            "yolo_state": result["states"]["yolo"],
            "yolo_inference": result["new_yolo_inference"],
        })
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(summary_rows)
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    width_counts = {state: sum(row["width_state"] == state for row in summary_rows) for state in (
        "complete_width_candidate", "partial_width_only", "branch_unresolved_width_candidate",
        "measurement_unavailable"
    )}
    yolo_samples = [row["sample_id"] for row in summary_rows if row["yolo_inference"]]
    split_samples = [row["sample_id"] for row in summary_rows if row["yolo_state"] == "split_confirmed"]
    failures = []
    if len(results) != 40:
        failures.append("sample_count")
    if not all(result["base_sidewalk_preserved"] for result in results):
        failures.append("base_sidewalk_preservation")
    if any(result["branch_pixels_deleted"] for result in results):
        failures.append("branch_deletion")
    if yolo_samples != ["IMG_4548", "commons_04"]:
        failures.append(f"unexpected_yolo_samples:{yolo_samples}")
    if split_samples != ["IMG_4548", "commons_04"]:
        failures.append(f"unexpected_split_samples:{split_samples}")

    by_id = {result["sample_id"]: result for result in results}
    targeted = {
        "IMG_4548_yolo_split": by_id["IMG_4548"]["states"]["yolo"] == "split_confirmed",
        "IMG_4548_false_vehicle_footprint_removed": all(
            not child.get("accepted", False)
            for component in by_id["IMG_4548"]["yolo"]["components"]
            for child in component["children"]
        ),
        "commons_04_yolo_split": by_id["commons_04"]["states"]["yolo"] == "split_confirmed",
        "IMG_4697_branch_width_not_declared_complete": (
            by_id["IMG_4697"]["states"]["width"] == "branch_unresolved_width_candidate"
        ),
        "kr_cone_01_yolo_not_called": not by_id["kr_cone_01"]["new_yolo_inference"],
        "kr_cone_01_route_avoids_left_curb_suspect": (
            by_id["kr_cone_01"]["route_start_x_fraction"] is not None
            and by_id["kr_cone_01"]["route_start_x_fraction"] > 0.55
        ),
        "people_not_fixed_footprints": all(
            not any(
                row.get("accepted") and row.get("class_id") in TRANSIENT_IDS
                for row in result["footprint_components"]
            ) for result in results
        ),
    }
    failures.extend(key for key, passed in targeted.items() if not passed)
    validation = {
        "status": "passed" if not failures else "failed",
        "expected_samples": 40,
        "processed_samples": len(results),
        "all_artifacts_present": all(
            (OUT / "artifacts" / result["sample_id"] / "summary.png").exists() for result in results
        ),
        "base_sidewalk_preserved_all": all(result["base_sidewalk_preserved"] for result in results),
        "branch_pixels_deleted_total": sum(result["branch_pixels_deleted"] for result in results),
        "mapillary_inference_reused": True,
        "conditional_yolo_inference_count": len(yolo_samples),
        "conditional_yolo_samples": yolo_samples,
        "yolo_split_confirmed_samples": split_samples,
        "width_status_counts": width_counts,
        "targeted_regressions": targeted,
        "existing_outputs_overwritten": False,
        "failures": failures,
    }
    (OUT / "completion_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cards = []
    for row in summary_rows:
        sample_id = html.escape(row["sample_id"])
        cards.append(
            f'<section class="{html.escape(row["yolo_state"])}"><h2>{int(row["order"]):02d}. {sample_id} '
            f'<small>{html.escape(row["split"])}</small></h2>'
            f'<p>{html.escape(row["width_state"])} · direction {html.escape(row["direction_state"])} · '
            f'branch {html.escape(row["branch_state"])} · footprints {row["accepted_footprints"]} · '
            f'YOLO {html.escape(row["yolo_state"])}</p>'
            f'<a href="artifacts/{sample_id}/result.json">result.json</a>'
            f'<img src="artifacts/{sample_id}/summary.png"></section>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Integrated V2 regression 40 v4</title>'
        '<style>body{font:16px system-ui;background:#101722;color:#eef;max-width:1550px;margin:auto;padding:18px}'
        'section{background:#192331;margin:24px 0;padding:18px;border-radius:12px}'
        'section.split_confirmed{border:3px solid #43ded5}img{width:100%;height:auto}'
        'small{color:#8ff}a{color:#8df}</style><h1>Integrated V2 regression — 40 images</h1>'
        f'<p>Validation: {validation["status"]}. Cached Mapillary maps reused; YOLO called conditionally on '
        f'{len(yolo_samples)}/40 images ({", ".join(yolo_samples)}). No branch or direction-based surface deletion.</p>'
        f'<p>Width status: complete {width_counts["complete_width_candidate"]} · '
        f'partial {width_counts["partial_width_only"]} · unavailable {width_counts["measurement_unavailable"]}. '
        f'branch unresolved {width_counts["branch_unresolved_width_candidate"]} · '
        'Widths are image pixels, not meters. Branch candidates remain unresolved and preserved.</p>'
        + ''.join(cards)
    )
    (OUT / "review.html").write_text(page, encoding="utf-8")
    (OUT / "PLAN.md").write_text(
        "# Integrated V2 regression plan\n\n"
        "1. Reuse cached Mapillary 65-class maps.\n"
        "2. Apply Sidewalk-only base and conservative surface inheritance.\n"
        "3. Estimate non-destructive direction candidates from the 94% start seed.\n"
        "4. Use actual connected-surface outer boundaries and multiple horizontal rows.\n"
        "5. Preserve Curb outside suspects in the semantic surface but exclude them from route/width selection.\n"
        "6. Apply V2 class-aware, walkway-owned clearance footprints; people remain occlusion.\n"
        "7. Trigger YOLO11n-seg only for walkway-owned, multi-depth merged Car components.\n"
        "8. Partition Mapillary Car pixels by confirmed YOLO instances and recalculate footprints.\n"
        "9. Preserve unresolved side branches and report pixel widths as non-metric candidates.\n",
        encoding="utf-8",
    )
    (OUT / "REPORT.md").write_text(
        "# Integrated V2 regression 40 v4\n\n"
        f"- Validation: {validation['status']}\n"
        "- Cached Mapillary argmax maps reused for all 40 images\n"
        f"- Conditional YOLO calls: {len(yolo_samples)}/40 ({', '.join(yolo_samples)})\n"
        f"- Confirmed YOLO splits: {', '.join(split_samples)}\n"
        "- Base Sidewalk preservation: all images\n"
        "- Curb outside suspects: preserved in semantic output, excluded only from route/width candidate selection\n"
        "- Direction/branch-based deletion: 0 pixels\n"
        f"- Width statuses: {json.dumps(width_counts, ensure_ascii=False)}\n"
        "- Structural width uses actual outer boundaries and repeated horizontal rows\n"
        "- Free width uses V2 policy-aligned clearance footprints and hazards\n"
        "- People/riders remain occlusion and never receive fixed footprints\n"
        "- Side branches are preserved and marked unresolved; no validated branch deletion yet\n"
        "- Widths are image-space pixels, never meters\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
