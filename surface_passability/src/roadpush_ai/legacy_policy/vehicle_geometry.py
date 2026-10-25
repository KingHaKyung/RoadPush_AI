from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = get_settings().project_root
OUT = HERE / "vehicle_geometry_split_probe_v2"
MANIFEST = HERE / "model_audit_40" / "audit_manifest.json"
SAMPLES = (
    "commons_04",
    "IMG_4548",
    "IMG_4364",
    "kr_cone_01",
    "commons_01",
    "commons_06",
    "MP_SEL_PN000695",
    "kr_signboard_01",
)
CAR_ID = 55
MAX_SIDE = 900



from roadpush_ai.legacy_policy import class_aware_clearance as r3


def fit(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.72) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    window = max(5, int(window) | 1)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.asarray([np.median(padded[index:index + window]) for index in range(len(values))])


def bottom_profile(component: np.ndarray, x0: int, width: int) -> np.ndarray:
    profile = np.full(width, np.nan, np.float32)
    for local_x in range(width):
        ys = np.flatnonzero(component[:, x0 + local_x])
        if len(ys):
            profile[local_x] = float(ys.max())
    valid = np.isfinite(profile)
    if not valid.any():
        return np.zeros(width, np.float32)
    positions = np.arange(width)
    profile = np.interp(positions, positions[valid], profile[valid]).astype(np.float32)
    return smooth_profile(profile, max(7, round(width * 0.015)))


def best_depth_cut(profile: np.ndarray) -> dict | None:
    width = len(profile)
    start, stop = max(8, round(width * 0.18)), min(width - 8, round(width * 0.82))
    if stop <= start:
        return None
    best = None
    for cut in range(start, stop):
        left, right = profile[:cut], profile[cut:]
        left_median, right_median = float(np.median(left)), float(np.median(right))
        left_mad = float(np.median(np.abs(left - left_median)))
        right_mad = float(np.median(np.abs(right - right_median)))
        difference = abs(left_median - right_median)
        score = difference / max(1.0, 1.0 + left_mad + right_mad)
        candidate = {
            "cut": cut,
            "left_median_y": left_median,
            "right_median_y": right_median,
            "depth_difference_px": difference,
            "left_mad_px": left_mad,
            "right_mad_px": right_mad,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def component_footprint(component: np.ndarray) -> np.ndarray:
    estimated = r3.class_aware_component_footprint(component, CAR_ID)
    return np.zeros_like(component) if estimated is None else estimated[0]


def analyse_component(component: np.ndarray, bbox: list[int], image_height: int) -> tuple[list[np.ndarray], dict]:
    x, y, width, height = bbox
    profile = bottom_profile(component, x, width)
    candidate = best_depth_cut(profile)
    bottom_gap = max(0, image_height - (y + height))
    bottom_tolerance = max(3, round(image_height * 0.06))
    touches_bottom = bottom_gap <= bottom_tolerance
    record = {
        "source_bbox": bbox,
        "source_pixels": int(component.sum()),
        "touches_bottom": touches_bottom,
        "bottom_gap_px": bottom_gap,
        "bottom_tolerance_px": bottom_tolerance,
        "decision": "keep_parent",
        "reason": "no_reliable_multi_depth_evidence",
        "profile": profile.tolist(),
    }
    if candidate is None:
        return [component], record
    record.update(candidate)
    threshold = max(24, image_height * 0.17)
    split_x = x + int(candidate["cut"])
    left = component.copy()
    left[:, split_x:] = False
    right = component.copy()
    right[:, :split_x] = False
    parent_pixels = max(1, int(component.sum()))
    child_fractions = [float(left.sum()) / parent_pixels, float(right.sum()) / parent_pixels]
    record["split_x"] = split_x
    record["required_depth_difference_px"] = threshold
    record["child_area_fractions"] = child_fractions
    strong_depth_step = candidate["depth_difference_px"] >= threshold
    meaningful_children = min(child_fractions) >= 0.06
    if touches_bottom and strong_depth_step and meaningful_children:
        record["decision"] = "split_candidate"
        record["reason"] = "bottom_truncated_component_with_two_strong_depth_levels"
        return [left, right], record
    if not touches_bottom:
        record["reason"] = "not_bottom_truncated"
    elif not strong_depth_step:
        record["reason"] = "depth_step_below_threshold"
    else:
        record["reason"] = "child_too_small"
    return [component], record


def process(item: dict) -> dict:
    with Image.open(ROOT / item["relative_path"]) as source:
        size = fit(source.width, source.height)
        image = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    native = cv2.imread(str(ROOT / item["class_map_path"]), cv2.IMREAD_UNCHANGED)
    if native is None:
        raise FileNotFoundError(item["class_map_path"])
    labels = cv2.resize(native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    car_mask = labels == CAR_ID
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(car_mask.astype(np.uint8), 8)

    original_footprints = np.zeros_like(car_mask)
    proposed_footprints = np.zeros_like(car_mask)
    proposed_instances: list[np.ndarray] = []
    records = []
    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if area < max(20, round(labels.size * 0.00015)):
            continue
        component = component_labels == component_id
        original_footprints |= component_footprint(component)
        children, record = analyse_component(component, [x, y, width, height], labels.shape[0])
        record["component_id"] = component_id
        records.append(record)
        for child in children:
            proposed_instances.append(child)
            proposed_footprints |= component_footprint(child)

    raw_view = paint(image, car_mask, (235, 65, 75), 0.65)
    component_view = image.copy()
    palette = [(245, 95, 70), (80, 185, 255), (170, 95, 245), (245, 195, 50), (60, 210, 135)]
    for index, component_id in enumerate(range(1, count)):
        component_view = paint(component_view, component_labels == component_id, palette[index % len(palette)], 0.72)

    evidence_view = raw_view.copy()
    for record in records:
        x, y, width, height = record["source_bbox"]
        color = (40, 225, 255) if record["decision"] == "split_candidate" else (190, 190, 190)
        cv2.rectangle(evidence_view, (x, y), (x + width - 1, y + height - 1), color, 3)
        if "split_x" in record:
            cv2.line(evidence_view, (record["split_x"], y), (record["split_x"], y + height - 1), color, 4)

    split_view = image.copy()
    for index, instance in enumerate(proposed_instances):
        split_view = paint(split_view, instance, palette[index % len(palette)], 0.72)
    original_view = paint(image, original_footprints, (255, 195, 35), 0.82)
    proposed_view = image.copy()
    for index, instance in enumerate(proposed_instances):
        proposed_view = paint(proposed_view, component_footprint(instance), palette[index % len(palette)], 0.82)

    split_count = sum(record["decision"] == "split_candidate" for record in records)
    artifact = OUT / "artifacts" / item["sample_id"]
    artifact.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [image, raw_view, component_view, evidence_view, original_view, proposed_view]
    titles = [
        "1 Original",
        "2 Raw semantic Car pixels",
        "3 Connected Car components",
        "4 Depth evidence (cyan=split, gray=keep)",
        "5 Current parent footprint",
        "6 Proposed per-depth footprints",
    ]
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{item['sample_id']} | vehicle geometry split probe | split candidates {split_count}", fontsize=16)
    fig.savefig(artifact / "summary.png", dpi=115)
    plt.close(fig)

    payload = {
        "sample_id": item["sample_id"],
        "split_candidates": split_count,
        "connected_components_considered": len(records),
        "components": records,
        "new_model_inference": False,
        "source": "cached Mapillary semantic argmax",
        "safety_fallback": "keep_parent_when_uncertain",
    }
    (artifact / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    by_id = {item["sample_id"]: item for item in items}
    results = [process(by_id[sample_id]) for sample_id in SAMPLES]

    sections = []
    for result in results:
        sample_id = result["sample_id"]
        sections.append(
            f'<section><h2>{sample_id}</h2><p>split candidates: {result["split_candidates"]} · '
            f'components: {result["connected_components_considered"]}</p>'
            f'<a href="artifacts/{sample_id}/result.json">result.json</a>'
            f'<img src="artifacts/{sample_id}/summary.png"></section>'
        )
    html = (
        '<!doctype html><meta charset="utf-8"><title>Vehicle geometry split probe v2</title>'
        '<style>body{font:16px system-ui;background:#101722;color:#eef;max-width:1500px;margin:auto}'
        'section{background:#192331;margin:22px;padding:18px;border-radius:12px}img{width:100%;height:auto}'
        'a{color:#8df}</style><h1>Vehicle geometry split probe v2</h1>'
        '<p>Geometry-only diagnostic. No model inference, no deletion, and no V2 integration. '
        'Uncertain components keep their parent mask.</p>' + ''.join(sections)
    )
    (OUT / "review.html").write_text(html, encoding="utf-8")
    validation = {
        "expected_samples": len(SAMPLES),
        "processed_samples": len(results),
        "all_artifacts_present": all((OUT / "artifacts" / r["sample_id"] / "summary.png").exists() for r in results),
        "new_model_inference": False,
        "v2_modified": False,
        "existing_outputs_overwritten": False,
        "split_samples": [r["sample_id"] for r in results if r["split_candidates"]],
    }
    (OUT / "completion_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
