from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = get_settings().project_root
OUT = Path(__file__).resolve().parent / "curb_boundary_probe_40_v1"
INPUTS = ROOT / "outputs" / "roi_connectivity_benchmark" / "input_manifest.json"
RAW_ROOT = ROOT / "outputs" / "roi_connectivity_benchmark" / "raw" / "mask2former_mapillary_swin_l"

SIDEWALK = 15
BOUNDARY_CONTEXT = (2, 9, 10, 13, 14)  # Curb, Curb Cut, Parking, Road, Service Lane
MIN_COMPONENT_IMAGE_FRACTION = 500 / (1050 * 1400)
MIN_BOUNDARY_CONTEXT_FRACTION = 0.50
MAX_PANEL_DIMENSION = 700


def tint(photo: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    base = np.asarray(photo).astype(np.float32)
    layer = np.empty_like(base)
    layer[:] = color
    result = base.copy()
    result[mask] = base[mask] * (1.0 - alpha) + layer[mask] * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def fit(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_PANEL_DIMENSION / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def analyze(class_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, object]]]:
    sidewalk = (class_map == SIDEWALK).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(sidewalk, 8)
    if count == 1:
        return labels, np.zeros(class_map.shape, dtype=bool), 0, []

    anchor_id = max(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    radius = max(5, round(min(class_map.shape) / 70))
    minimum_pixels = max(100, round(class_map.size * MIN_COMPONENT_IMAGE_FRACTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    suspect = np.zeros(class_map.shape, dtype=bool)
    rows: list[dict[str, object]] = []

    for component_id in range(1, count):
        component = (labels == component_id).astype(np.uint8)
        ring = (cv2.dilate(component, kernel, iterations=1) > 0) & (component == 0)
        ring_pixels = int(ring.sum())
        context_pixels = int((np.isin(class_map, BOUNDARY_CONTEXT) & ring).sum())
        context_fraction = context_pixels / ring_pixels if ring_pixels else 0.0
        pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        flagged = (
            component_id != anchor_id
            and pixels >= minimum_pixels
            and context_fraction >= MIN_BOUNDARY_CONTEXT_FRACTION
        )
        if flagged:
            suspect |= component.astype(bool)
        rows.append({
            "component_id": component_id,
            "pixels": pixels,
            "bbox_x": int(stats[component_id, cv2.CC_STAT_LEFT]),
            "bbox_y": int(stats[component_id, cv2.CC_STAT_TOP]),
            "bbox_width": int(stats[component_id, cv2.CC_STAT_WIDTH]),
            "bbox_height": int(stats[component_id, cv2.CC_STAT_HEIGHT]),
            "centroid_x": float(centroids[component_id, 0]),
            "centroid_y": float(centroids[component_id, 1]),
            "boundary_context_fraction": context_fraction,
            "minimum_pixels_for_image": minimum_pixels,
            "anchor": component_id == anchor_id,
            "flagged_suspect": flagged,
        })
    return labels, suspect, anchor_id, rows


def make_comparison(
    original: Image.Image,
    class_map: np.ndarray,
    labels: np.ndarray,
    suspect: np.ndarray,
    anchor_id: int,
    sample_id: str,
    output: Path,
) -> None:
    size = fit(original.width, original.height)
    photo = original.resize(size, Image.Resampling.LANCZOS)
    small_class = cv2.resize(class_map.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    small_labels = cv2.resize(labels.astype(np.int32), size, interpolation=cv2.INTER_NEAREST)
    small_suspect = cv2.resize(suspect.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)

    context = tint(photo, small_class == 15, (35, 215, 90), 0.48)
    context = tint(context, small_class == 13, (240, 80, 70), 0.58)
    context = tint(context, small_class == 2, (55, 120, 255), 0.78)
    context = tint(context, small_class == 9, (30, 225, 235), 0.68)
    components = tint(photo, small_labels == anchor_id, (35, 215, 90), 0.55) if anchor_id else photo.copy()
    components = tint(components, (small_labels > 0) & (small_labels != anchor_id), (250, 205, 45), 0.65)
    diagnostic = tint(photo, small_class == SIDEWALK, (35, 200, 85), 0.38)
    diagnostic = tint(diagnostic, small_suspect, (245, 55, 65), 0.78)

    views = [photo, context, components, diagnostic]
    titles = [
        "Original",
        "Raw context | green sidewalk, red road, blue curb, cyan curb cut",
        "Components | green largest anchor, yellow disconnected",
        "Boundary suspect | red is REVIEW ONLY, not removed",
    ]
    margin, gap, title_h = 24, 18, 30
    canvas = Image.new(
        "RGB",
        (margin * 2 + size[0] * 2 + gap, margin * 2 + (size[1] + title_h) * 2 + gap + 24),
        (18, 24, 33),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, 8), f"{sample_id} | deterministic curb-boundary probe (no deletion)", fill="white", font=font)
    for index, (view, title) in enumerate(zip(views, titles)):
        col, row = index % 2, index // 2
        x = margin + col * (size[0] + gap)
        y = margin + 22 + row * (size[1] + title_h + gap)
        draw.text((x, y), title, fill="white", font=font)
        canvas.paste(view, (x, y + title_h))
    canvas.save(output, quality=92, optimize=True)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT}")
    assets = OUT / "assets"
    masks = OUT / "suspect_masks"
    assets.mkdir(parents=True)
    masks.mkdir(parents=True)

    items = json.loads(INPUTS.read_text(encoding="utf-8"))
    summary_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for source_order, item in enumerate(items, start=1):
        sample_id = item["sample_id"]
        original = Image.open(ROOT / item["relative_path"]).convert("RGB")
        class_map = cv2.imread(str(RAW_ROOT / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        if class_map is None or class_map.shape != (original.height, original.width):
            raise ValueError(f"Missing class map or source mismatch: {sample_id}")
        labels, suspect, anchor_id, rows = analyze(class_map)
        for row in rows:
            row["sample_id"] = sample_id
            component_rows.append(row)
        Image.fromarray((suspect.astype(np.uint8) * 255), mode="L").save(masks / f"{sample_id}.png", optimize=True)
        make_comparison(original, class_map, labels, suspect, anchor_id, sample_id, assets / f"{sample_id}.jpg")
        sidewalk_pixels = int((class_map == SIDEWALK).sum())
        suspect_pixels = int(suspect.sum())
        summary_rows.append({
            "sample_id": sample_id,
            "source_order": source_order,
            "split": item["split"],
            "sidewalk_pixels": sidewalk_pixels,
            "suspect_pixels": suspect_pixels,
            "suspect_fraction_of_image": suspect_pixels / class_map.size,
            "suspect_fraction_of_sidewalk": suspect_pixels / sidewalk_pixels if sidewalk_pixels else 0.0,
            "sidewalk_component_count": int(labels.max()),
            "flagged_component_count": sum(bool(row["flagged_suspect"]) for row in rows),
            "anchor_component_id": anchor_id,
        })

    summary_rows.sort(key=lambda row: (-float(row["suspect_fraction_of_sidewalk"]), int(row["source_order"])))
    for review_order, row in enumerate(summary_rows, start=1):
        row["review_order"] = review_order
    summary_fields = [
        "sample_id", "review_order", "source_order", "split", "sidewalk_pixels", "suspect_pixels",
        "suspect_fraction_of_image", "suspect_fraction_of_sidewalk", "sidewalk_component_count",
        "flagged_component_count", "anchor_component_id",
    ]
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(summary_rows)
    component_fields = [
        "sample_id", "component_id", "pixels", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "centroid_x", "centroid_y", "boundary_context_fraction", "minimum_pixels_for_image",
        "anchor", "flagged_suspect",
    ]
    with (OUT / "components.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=component_fields)
        writer.writeheader(); writer.writerows(component_rows)

    cards = []
    for row in summary_rows:
        sample_id = str(row["sample_id"])
        cards.append(
            f'<section><h2>{int(row["review_order"]):02d}. {html.escape(sample_id)} '
            f'<small>{html.escape(str(row["split"]))} · original order {int(row["source_order"]):02d}</small></h2>'
            f'<p>Suspect {float(row["suspect_fraction_of_sidewalk"]):.2%} of Sidewalk · '
            f'{float(row["suspect_fraction_of_image"]):.2%} of image · '
            f'{int(row["flagged_component_count"])} flagged / {int(row["sidewalk_component_count"])} components</p>'
            f'<a href="assets/{html.escape(sample_id)}.jpg"><img src="assets/{html.escape(sample_id)}.jpg"></a></section>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Curb boundary diagnostic — 40 images</title>'
        '<style>body{font:15px system-ui;background:#10151d;color:#eef;max-width:1550px;margin:auto}'
        'section{background:#19212c;margin:22px;padding:18px;border-radius:12px}img{width:100%;height:auto}'
        'small{color:#8fd}p{line-height:1.5}.warning{background:#422b23;padding:14px;border-radius:10px}</style>'
        '<h1>Deterministic curb-boundary diagnostic — 40 images</h1>'
        '<p>No VLM, no new model inference, and no pixel deletion. Cached class maps were processed by one fixed rule.</p>'
        '<p class="warning"><b>Red means review candidate, not confirmed error.</b> The largest Sidewalk component is green. '
        'Disconnected components are yellow, and only sufficiently large components surrounded by at least 50% '
        'Curb/Curb Cut/Road/Parking/Service Lane context become red.</p>'
        + "".join(cards)
    )
    (OUT / "review.html").write_text(page, encoding="utf-8")

    manifest = {
        "status": "diagnostic_complete_no_deletion",
        "sample_count": len(items),
        "new_model_inference": False,
        "vlm_used": False,
        "source": "cached native-size Mapillary argmax class maps",
        "rule": {
            "anchor": "largest connected Sidewalk component",
            "minimum_component_image_fraction": MIN_COMPONENT_IMAGE_FRACTION,
            "minimum_boundary_context_fraction": MIN_BOUNDARY_CONTEXT_FRACTION,
            "boundary_context_class_ids": list(BOUNDARY_CONTEXT),
        },
        "warning": "Curb and Curb Cut can be misclassified. Red candidates require visual review and are not removed.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
