from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import html
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = get_settings().project_root
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "model_audit_40" / "audit_manifest.json"
OUT = HERE / "ground_footprint_probe_40_r1"
MAX_SIDE = 700
POLE_ID = 45
CYCLE_IDS = (52, 57)
SUMMARY_LABEL = "stable lower-span footprint probe"
FOOTPRINT_PANEL_LABEL = "3 Ground footprints (yellow) from stable lower spans"
WIDTH_PANEL_LABEL = "6 Width free space (footprints + hazards removed)"


def conditional_visual_obstacles(labels: np.ndarray, footprints: np.ndarray) -> np.ndarray:
    return np.zeros_like(labels, dtype=bool)



from roadpush_ai.legacy_policy import outer_boundary as outer
base = outer.base
v2 = outer.v2


def fit(w: int, h: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def component_footprint(component: np.ndarray, class_id: int) -> tuple[np.ndarray, dict] | None:
    """Estimate a ground patch from a stable wide span in the component's lower band.

    The absolute widest row is intentionally not used: handlebars, arms, signs, and
    one-row segmentation spikes may be wider than the true ground contact.  A high
    percentile of repeated lower-band spans is used instead, and its vertical run
    is retained so a pedestal remains a 2-D patch rather than a one-row contact.
    """
    image_h, image_w = component.shape
    ys, xs = np.nonzero(component)
    minimum = max(4, round(component.size * 0.000008))
    if len(xs) < minimum:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    object_h = y1 - y0 + 1
    object_w = x1 - x0 + 1
    band_depth = max(5, round(object_h * 0.34))
    band_top = max(y0, y1 - band_depth + 1)

    row_stats = []
    for y in range(band_top, y1 + 1):
        row_x = np.flatnonzero(component[y])
        if not len(row_x):
            continue
        row_stats.append((y, int(row_x[0]), int(row_x[-1]), int(row_x[-1] - row_x[0] + 1)))
    if not row_stats:
        return None

    widths = np.asarray([row[3] for row in row_stats], np.float32)
    stable_width = max(1.0, float(np.percentile(widths, 85)))
    broad = [row for row in row_stats if row[3] >= max(1.0, stable_width * 0.65)]
    if not broad:
        broad = [max(row_stats, key=lambda row: row[3])]

    # Ignore an isolated upper flare unless it continues toward the component bottom.
    max_gap = max(2, round(image_h * 0.004))
    groups: list[list[tuple[int, int, int, int]]] = []
    for row in broad:
        if not groups or row[0] - groups[-1][-1][0] > max_gap:
            groups.append([row])
        else:
            groups[-1].append(row)
    chosen = min(groups, key=lambda group: (y1 - group[-1][0], -len(group)))

    left = int(round(np.percentile([row[1] for row in chosen], 10)))
    right = int(round(np.percentile([row[2] for row in chosen], 90)))
    top = int(chosen[0][0])
    bottom = y1

    # A small margin represents clearance and also closes ragged last-row labels.
    margin_x = max(2, round(image_w * 0.004), round((right - left + 1) * 0.08))
    margin_y = max(2, round(image_h * 0.004))
    left = max(0, left - margin_x)
    right = min(image_w - 1, right + margin_x)
    top = max(0, top - margin_y)
    bottom = min(image_h - 1, bottom + margin_y)

    footprint = np.zeros_like(component, dtype=np.uint8)
    # Rounded rectangle preserves multi-row square pedestals but avoids sharp noise corners.
    cv2.rectangle(footprint, (left, top), (right, bottom), 1, cv2.FILLED)
    radius = min(max(1, margin_y), max(1, (bottom - top + 1) // 3), max(1, (right - left + 1) // 5))
    if radius > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, kernel)

    return footprint.astype(bool), {
        "class_id": int(class_id),
        "source_pixels": int(component.sum()),
        "source_bbox": [x0, y0, object_w, object_h],
        "lower_band_top": int(band_top),
        "stable_width_px": float(stable_width),
        "qualifying_rows": len(broad),
        "selected_rows": len(chosen),
        "footprint_bbox": [left, top, right - left + 1, bottom - top + 1],
        "footprint_pixels": int(footprint.sum()),
        "kind": "pole_or_bollard" if class_id == POLE_ID else (
            "cycle_like" if class_id in CYCLE_IDS else "generic_obstacle"
        ),
    }


def estimate_footprints(labels: np.ndarray, structural: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    object_ids = tuple(v2.OCCUPANCY) + tuple(v2.OCCLUSION)
    footprints = np.zeros_like(labels, dtype=bool)
    rows: list[dict] = []
    near_structure = cv2.dilate(
        structural.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)

    for class_id in object_ids:
        binary = (labels == class_id).astype(np.uint8)
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_index in range(1, count):
            if int(stats[component_index, cv2.CC_STAT_AREA]) < 4:
                continue
            component = component_labels == component_index
            estimated = component_footprint(component, class_id)
            if estimated is None:
                continue
            footprint, metadata = estimated
            relevant_pixels = int((footprint & near_structure).sum())
            accepted = relevant_pixels > 0
            if accepted:
                footprints |= footprint
            metadata.update({
                "component_index": component_index,
                "near_structural_pixels": relevant_pixels,
                "accepted": accepted,
            })
            rows.append(metadata)
    return footprints, rows


def draw_records(image: np.ndarray, records: list[dict], selected_color=(35, 85, 255)) -> None:
    for record in records:
        y = record["y"]
        for left, right in record["free_segments"]:
            cv2.line(image, (left, y), (right, y), (90, 205, 255), 2, cv2.LINE_AA)
        if record["selected_free_segment"]:
            left, right = record["selected_free_segment"]
            cv2.line(image, (left, y), (right, y), selected_color, 5, cv2.LINE_AA)


def process(item: dict) -> dict:
    sample_id = item["sample_id"]
    with Image.open(ROOT / item["relative_path"]) as source:
        size = fit(source.width, source.height)
        image = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    native_labels = cv2.imread(str(ROOT / item["class_map_path"]), cv2.IMREAD_UNCHANGED)
    if native_labels is None:
        raise FileNotFoundError(item["class_map_path"])

    repaired_native, _, _ = v2.repair_surface(native_labels)
    labels = cv2.resize(native_labels.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    repaired = cv2.resize(repaired_native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    obstacle_ids = tuple(v2.OCCUPANCY) + tuple(v2.OCCLUSION)
    raw_objects = np.isin(labels, obstacle_ids)
    raw_hazards = np.isin(labels, tuple(v2.HAZARD))
    raw_blocked = raw_objects | raw_hazards

    structural, restored = base.obstacle_transparent_structure(repaired, raw_blocked)
    footprints, footprint_rows = estimate_footprints(labels, structural)
    conditional_visual = conditional_visual_obstacles(labels, footprints)
    display_blocked = raw_blocked | conditional_visual
    # Display mask keeps complete visible silhouettes unpainted.  Width uses ground patches.
    visual_free = structural & ~display_blocked & ~footprints
    ground_free = structural & ~raw_hazards & ~footprints

    route, path = outer.route_path(repaired)
    component = outer.selected_component(structural, path)
    records = outer.sample_outer_boundaries(component, path, labels)
    for record in records:
        segments = base.free_segments(ground_free, record["y"], record["left"], record["right"])
        selected = outer.choose_route_segment(segments, record["anchor_x"])
        record["free_segments"] = [[left, right] for left, right in segments]
        record["selected_free_segment"] = list(selected) if selected else None
        record["selected_free_width"] = selected[1] - selected[0] + 1 if selected else 0
        record["widest_free_width"] = max((right - left + 1 for left, right in segments), default=0)

    complete = [record for record in records if record["width_state"] == "complete"]
    lower = [record for record in records if record["width_state"] == "lower_bound"]
    occluded = [record for record in records if record["width_state"] == "occlusion_limited"]
    status = "complete_width_candidate" if len(complete) >= 6 else (
        "partial_width_only" if lower or occluded else "measurement_unavailable"
    )

    raw_view = paint(image, display_blocked, (235, 65, 75), 0.62)
    footprint_view = paint(image, raw_objects, (225, 70, 165), 0.20)
    footprint_view = paint(footprint_view, footprints, (255, 195, 35), 0.82)
    structure_view = paint(image, repaired, (45, 210, 95), 0.38)
    structure_view = paint(structure_view, restored, (255, 145, 35), 0.82)
    visual_view = paint(image, visual_free, (45, 210, 95), 0.48)
    ground_view = paint(image, ground_free, (85, 185, 235), 0.25)
    draw_records(ground_view, records)

    art = OUT / "artifacts" / sample_id
    art.mkdir(parents=True)
    cv2.imwrite(str(art / "01_structural_surface.png"), structural.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02_raw_blocked.png"), display_blocked.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "03_ground_footprints.png"), footprints.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "04_visual_free.png"), visual_free.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "05_ground_free.png"), ground_free.astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [paint(image, repaired, (45, 210, 95), 0.42), raw_view, footprint_view,
              structure_view, visual_view, ground_view]
    titles = [
        "1 Repaired sidewalk source",
        "2 Full obstacle silhouettes (red, still excluded visually)",
        FOOTPRINT_PANEL_LABEL,
        "4 Structural sidewalk (orange=temporary restoration)",
        "5 Visual free space (all silhouettes removed)",
        WIDTH_PANEL_LABEL,
    ]
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{sample_id} | {SUMMARY_LABEL} | {status}", fontsize=16)
    fig.savefig(art / "summary.png", dpi=115)
    plt.close(fig)

    accepted = [row for row in footprint_rows if row["accepted"]]
    selected_widths = [record["selected_free_width"] for record in records if record["selected_free_width"]]
    payload = {
        "sample_id": sample_id,
        "order": item["order"],
        "split": item["split"],
        "status": status,
        "sampled_rows": len(records),
        "complete_rows": len(complete),
        "lower_bound_rows": len(lower),
        "occlusion_limited_rows": len(occluded),
        "accepted_footprints": len(accepted),
        "pole_footprints": sum(row["accepted"] and row["class_id"] == POLE_ID for row in footprint_rows),
        "cycle_footprints": sum(row["accepted"] and row["class_id"] in CYCLE_IDS for row in footprint_rows),
        "footprint_pixels": int(footprints.sum()),
        "median_route_selected_ground_width_px": float(np.median(selected_widths)) if selected_widths else None,
        "footprint_components": footprint_rows,
        "rows": records,
        "new_model_inference": False,
        "branch_removal": False,
        "straight_boundary_fit": False,
        "full_silhouettes_excluded_from_visual_free": True,
        "ground_width_uses_footprints_not_full_silhouettes": True,
    }
    (art / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = [process(item) for item in manifest["samples"]]

    summary_fields = [key for key in rows[0] if key not in ("rows", "footprint_components")]
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in summary_fields} for row in rows])

    counts = {status: sum(row["status"] == status for row in rows) for status in (
        "complete_width_candidate", "partial_width_only", "measurement_unavailable"
    )}
    cards = []
    for row in rows:
        sample_id = html.escape(row["sample_id"])
        width = row["median_route_selected_ground_width_px"]
        width_text = "n/a" if width is None else f"{width:.1f}px"
        cards.append(
            f'<section><h2>{row["order"]:02d}. {sample_id} <small>{row["split"]}</small></h2>'
            f'<p>{row["status"]} · footprints {row["accepted_footprints"]} '
            f'(pole {row["pole_footprints"]}, cycle {row["cycle_footprints"]}) · '
            f'ground-width median {width_text}</p>'
            f'<a href="artifacts/{sample_id}/summary.png"><img src="artifacts/{sample_id}/summary.png"></a></section>'
        )
    review = (
        '<!doctype html><meta charset="utf-8"><title>Ground footprint probe 40 r1</title>'
        '<style>body{font:16px system-ui;background:#10151d;color:#eef;max-width:1500px;margin:auto;padding:18px}'
        'section{background:#19212c;margin:24px 0;padding:18px;border-radius:12px}'
        'img{width:100%;height:auto}small{color:#8ff}</style>'
        '<h1>Stable lower-span ground-footprint probe — 40 cached images</h1>'
        '<p>Visible obstacle silhouettes remain excluded from the green visualization. Width computation uses '
        '2-D ground footprints estimated from repeated wide spans in each object lower band. The absolute widest '
        'row is not used. No branch removal and no new model inference.</p>'
        f'<p>Complete {counts["complete_width_candidate"]} · Partial {counts["partial_width_only"]} · '
        f'Unavailable {counts["measurement_unavailable"]}</p>'
        + "".join(cards)
    )
    (OUT / "review.html").write_text(review, encoding="utf-8")

    validation = {
        "expected_samples": 40,
        "processed_samples": len(rows),
        "all_artifacts_present": all(
            (OUT / "artifacts" / row["sample_id"] / "summary.png").exists() for row in rows
        ),
        "new_model_inference": False,
        "existing_outputs_overwritten": False,
        "full_silhouettes_excluded_from_visual_free": True,
        "ground_width_uses_stable_lower_span_footprints": True,
        "absolute_widest_row_used": False,
        "branch_removal": False,
        "status_counts": counts,
    }
    (OUT / "completion_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(
        "# Ground footprint probe 40 r1\n\n"
        "- Reused 40 cached native argmax maps; no model inference\n"
        "- Full visible obstacle silhouettes remain excluded in the visual free-space mask\n"
        "- Width calculation replaces vertical object silhouettes with 2-D ground footprints\n"
        "- Footprints use a repeated high-percentile width in the lower 34% of each object component\n"
        "- Multi-row lower spans preserve square lamp-post pedestals\n"
        "- One-row maximum-width spikes are not used\n"
        "- Hazards remain fully excluded\n"
        "- No branch removal, boundary fitting, or destructive edit\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "samples": len(rows), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
