from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import colorsys
import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = get_settings().project_root
HERE = Path(__file__).resolve().parent
OUT = HERE / "pilot_8_v2"
RAW_ROOT = ROOT / "outputs" / "roi_connectivity_benchmark" / "raw" / "mask2former_mapillary_swin_l"
INPUTS = ROOT / "outputs" / "roi_connectivity_benchmark" / "input_manifest.json"
CONFIG = ROOT / "models" / "mask2former_swin_large_mapillary" / "config.json"

CORE = ["IMG_4697", "IMG_4364", "MP_SEL_SUR_006472", "kr_bicycle_01"]
SUPPLEMENT = ["commons_01", "IMG_4282", "kr_case_candidate_04", "kr_bollard_01"]
IDS = CORE + SUPPLEMENT
SIDEWALK = 15
REPAIR = {
    24: ("Lane Marking - General", 0.65, 0.15),
    36: ("Catch Basin", 0.75, 0.10),
    41: ("Manhole", 0.45, 0.30),
}
CONNECTION = (8, 9, 23)
BOUNDARY = (2, 3, 4, 5, 6, 17)
BOUNDARY_CONTEXT = (2, 9, 10, 13, 14)
FIXED_OCCUPANCY = (33, 34, 35, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51)
MOVING_OCCUPANCY = (1, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62)
OCCUPANCY = FIXED_OCCUPANCY + MOVING_OCCUPANCY
OCCLUSION = (19, 20, 21, 22)
HAZARD = (12, 26, 28, 31, 43)
NATURAL_NONWALK = (25, 29, 30)
EXCLUDED_GROUND = (7, 10, 11, 13, 14)
STRUCTURE_CONTEXT = (16, 18)
CONTEXT_ONLY = (0, 27, 32, 37, 53, 63, 64)
CONDITIONAL_GROUNDED_OCCUPANCY = (32,)
WIDTH_IGNORED_TRANSIENT = (19, 20, 21, 22)
ANALYSIS_MAX = 700


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_color(class_id: int) -> tuple[int, int, int]:
    if class_id == SIDEWALK:
        return 45, 210, 95
    hue = (class_id * 0.618033988749895) % 1.0
    saturation = 0.62 + 0.22 * ((class_id % 3) / 2)
    value = 0.78 + 0.18 * ((class_id % 2))
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def fit(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, ANALYSIS_MAX / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def overlay(image: np.ndarray, mask: np.ndarray, color=(35, 215, 90), alpha=0.55) -> np.ndarray:
    result = image.astype(np.float32).copy()
    layer = np.empty_like(result)
    layer[:] = color
    result[mask] = result[mask] * (1.0 - alpha) + layer[mask] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def save_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def repair_surface(class_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    base = class_map == SIDEWALK
    repaired = base.copy()
    additions = np.zeros_like(base)
    rows: list[dict[str, object]] = []
    height, width = class_map.shape
    radius = max(3, round(min(height, width) / 420))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    for class_id, (name, min_sidewalk, max_road) in REPAIR.items():
        binary = (class_map == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            pad = radius + 1
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
            local_labels = class_map[y0:y1, x0:x1]
            component = np.zeros(local_labels.shape, dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, 0, 0] -= x0
            shifted[:, 0, 1] -= y0
            cv2.drawContours(component, [shifted], -1, 1, cv2.FILLED)
            component &= (local_labels == class_id).astype(np.uint8)
            pixels = int(component.sum())
            if not pixels:
                continue
            ring = (cv2.dilate(component, kernel) > 0) & (component == 0) & (local_labels != class_id)
            ring_pixels = int(ring.sum())
            sidewalk_fraction = float(((local_labels == SIDEWALK) & ring).sum() / ring_pixels) if ring_pixels else 0.0
            road_fraction = float((np.isin(local_labels, (10, 13, 14)) & ring).sum() / ring_pixels) if ring_pixels else 1.0
            component_fraction = pixels / class_map.size
            touches_frame = x == 0 or y == 0 or x + w == width or y + h == height
            catch_basin_geometry_ok = class_id != 36 or (
                component_fraction <= 0.004 and not touches_frame
            )
            accepted = (
                sidewalk_fraction >= min_sidewalk
                and road_fraction <= max_road
                and catch_basin_geometry_ok
            )
            if accepted:
                accepted_mask = component.astype(bool)
                repaired[y0:y1, x0:x1][accepted_mask] = True
                additions[y0:y1, x0:x1][accepted_mask] = True
            rows.append({
                "class_id": class_id, "class_name": name, "component_index": index,
                "pixels": pixels, "sidewalk_ring_fraction": sidewalk_fraction,
                "road_ring_fraction": road_fraction, "component_fraction": component_fraction,
                "touches_frame": touches_frame, "catch_basin_geometry_ok": catch_basin_geometry_ok,
                "accepted": accepted,
            })
    return repaired, additions, rows


def boundary_suspect(class_map: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    sidewalk = (class_map == SIDEWALK).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(sidewalk, 8)
    if count == 1:
        return np.zeros_like(sidewalk, dtype=bool), []
    anchor = max(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    radius = max(5, round(min(class_map.shape) / 70))
    minimum = max(100, round(class_map.size * 500 / (1050 * 1400)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    suspect = np.zeros_like(sidewalk, dtype=bool)
    rows = []
    for component_id in range(1, count):
        component = (labels == component_id).astype(np.uint8)
        ring = (cv2.dilate(component, kernel) > 0) & (component == 0)
        ring_pixels = int(ring.sum())
        fraction = float((np.isin(class_map, BOUNDARY_CONTEXT) & ring).sum() / ring_pixels) if ring_pixels else 0.0
        pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        flagged = component_id != anchor and pixels >= minimum and fraction >= 0.50
        if flagged:
            suspect |= component.astype(bool)
        rows.append({"component_id": component_id, "pixels": pixels, "context_fraction": fraction,
                     "anchor": component_id == anchor, "flagged": flagged})
    return suspect, rows


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    work = (mask.astype(np.uint8) * 255)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton > 0


def longest_run(row: np.ndarray) -> int:
    padded = np.pad(row.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def direction_diagnostics(repaired_native: np.ndarray) -> dict[str, object]:
    size = fit(repaired_native.shape[1], repaired_native.shape[0])
    mask = cv2.resize(repaired_native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    skeleton = morphological_skeleton(mask)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - skeleton.astype(np.uint8)
    endpoints = skeleton & (neighbor_count == 1)
    junctions = skeleton & (neighbor_count >= 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    bottom_start = max(0, mask.shape[0] - max(4, round(mask.shape[0] * 0.15)))
    hypotheses = []
    start_mask = np.zeros_like(mask)
    for component_id in range(1, count):
        component = labels == component_id
        bottom_pixels = int(component[bottom_start:].sum())
        if bottom_pixels < max(10, round(stats[component_id, cv2.CC_STAT_AREA] * 0.002)):
            continue
        start_mask[bottom_start:] |= component[bottom_start:]
        ys = np.flatnonzero(component.any(axis=1))
        widths = [longest_run(component[y]) / mask.shape[1] for y in ys]
        hypotheses.append({
            "component_id": component_id,
            "pixels": int(stats[component_id, cv2.CC_STAT_AREA]),
            "rows": ys.tolist(),
            "relative_widths": widths,
            "median_relative_width": float(np.median(widths)) if widths else None,
            "p10_relative_width": float(np.percentile(widths, 10)) if widths else None,
        })
    endpoint_count = int(endpoints.sum())
    junction_count = int(junctions.sum())
    if not hypotheses:
        state, reason = "undetermined", "no_bottom_connected_surface_hypothesis"
    elif len(hypotheses) > 1:
        state, reason = "undetermined", "multiple_bottom_surface_hypotheses"
    elif endpoint_count > 2 or junction_count > 0:
        state, reason = "undetermined", "branching_or_ambiguous_direction"
    else:
        state, reason = "observed_candidate", "single_image_space_surface_hypothesis"
    return {
        "mask": mask, "skeleton": skeleton, "start_mask": start_mask,
        "endpoints": endpoints, "junctions": junctions, "hypotheses": hypotheses,
        "endpoint_count": endpoint_count, "junction_pixel_count": junction_count,
        "state": state, "reason": reason,
    }


def policy_map(class_map: np.ndarray, additions: np.ndarray) -> np.ndarray:
    result = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    result[:] = (30, 35, 42)
    result[np.isin(class_map, STRUCTURE_CONTEXT + CONTEXT_ONLY)] = (70, 74, 82)
    result[np.isin(class_map, NATURAL_NONWALK + EXCLUDED_GROUND)] = (115, 115, 105)
    result[np.isin(class_map, BOUNDARY)] = (55, 120, 255)
    result[np.isin(class_map, CONNECTION)] = (35, 220, 230)
    result[np.isin(class_map, OCCUPANCY)] = (235, 70, 65)
    result[np.isin(class_map, OCCLUSION)] = (250, 210, 50)
    result[np.isin(class_map, HAZARD)] = (220, 55, 220)
    result[class_map == SIDEWALK] = (35, 215, 90)
    result[additions] = (255, 155, 35)
    return result


def acceptance(sample_id: str, class_map: np.ndarray, additions: np.ndarray, repaired: np.ndarray) -> dict[str, object]:
    counts = Counter(map(int, class_map.ravel()))
    base_preserved = not np.any((class_map == SIDEWALK) & ~repaired)
    checks: dict[str, object] = {"base_sidewalk_preserved": base_preserved}
    if sample_id == "IMG_4697":
        checks.update({"manhole_pixels": counts[41], "manhole_repaired_pixels": int((additions & (class_map == 41)).sum())})
    elif sample_id == "IMG_4364":
        checks.update({"repair_pixels": int(additions.sum()), "repair_classes_only": bool(np.all(np.isin(class_map[additions], list(REPAIR))))})
    elif sample_id == "MP_SEL_SUR_006472":
        checks.update({"lane_marking_pixels": counts[24], "lane_marking_repaired_pixels": int((additions & (class_map == 24)).sum())})
    elif sample_id == "kr_bicycle_01":
        checks.update({"bicycle_pixels": counts[52], "bicycle_observed": counts[52] > 0})
    elif sample_id == "commons_01":
        checks.update({"car_pixels": counts[55], "car_observed": counts[55] > 0})
    elif sample_id == "IMG_4282":
        checks.update({"person_pixels": counts[19], "person_observed": counts[19] > 0})
    elif sample_id == "kr_case_candidate_04":
        static = sum(counts[key] for key in FIXED_OCCUPANCY)
        checks.update({"person_pixels": counts[19], "fixed_candidate_pixels": static})
    elif sample_id == "kr_bollard_01":
        checks.update({"traffic_cone_or_bollard_taxonomy_class": False, "taxonomy_gap_recorded": True})
    return checks


def make_summary(
    output: Path, sample_id: str, image: np.ndarray, class_map: np.ndarray, policy: np.ndarray,
    repaired: np.ndarray, additions: np.ndarray, suspect: np.ndarray, occupancy: np.ndarray,
    occlusion: np.ndarray, hazard: np.ndarray, diagnostic: dict[str, object], free: np.ndarray,
    states: dict[str, dict[str, str]],
) -> None:
    size = fit(image.shape[1], image.shape[0])
    photo = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    labels = cv2.resize(class_map.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    base = labels == SIDEWALK
    repaired_s = cv2.resize(repaired.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    additions_s = cv2.resize(additions.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    suspect_s = cv2.resize(suspect.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    occupancy_s = cv2.resize(occupancy.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    occlusion_s = cv2.resize(occlusion.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    hazard_s = cv2.resize(hazard.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    free_s = cv2.resize(free.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    palette = np.asarray([class_color(i) for i in range(256)], dtype=np.uint8)
    semantic = palette[labels]
    policy_s = cv2.resize(policy, size, interpolation=cv2.INTER_NEAREST)
    repair_view = overlay(overlay(photo, base), additions_s, (255, 155, 35), 0.80)
    curb_view = overlay(photo, labels == 2, (55, 120, 255), 0.75)
    curb_view = overlay(curb_view, labels == 9, (30, 225, 235), 0.65)
    curb_view = overlay(curb_view, suspect_s, (245, 55, 65), 0.82)
    object_view = overlay(photo, np.isin(labels, BOUNDARY), (55, 120, 255), 0.55)
    object_view = overlay(object_view, occupancy_s, (235, 70, 65), 0.68)
    object_view = overlay(object_view, hazard_s, (220, 55, 220), 0.75)
    occlusion_view = overlay(photo, occlusion_s, (250, 210, 50), 0.75)
    branch_view = overlay(photo, diagnostic["start_mask"], (255, 170, 25), 0.45)
    branch_view = overlay(branch_view, diagnostic["skeleton"], (245, 65, 65), 0.90)
    branch_view = overlay(branch_view, diagnostic["junctions"], (220, 55, 220), 0.95)
    hypothesis_view = overlay(photo, repaired_s, (35, 200, 85), 0.35)
    hypothesis_view = overlay(hypothesis_view, diagnostic["skeleton"], (255, 75, 70), 0.90)
    free_view = overlay(photo, free_s, (35, 220, 95), 0.58)

    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    titles = [
        "1 Original", "2 Full 65-class semantic", "3 Policy layers", "4 Initial Sidewalk",
        "5 Surface repair", "6 Curb + outside suspect", "7 Boundary/occupancy/hazard", "8 Person/rider occlusion",
        "9 Start + skeleton + all branches", "10 Non-destructive hypotheses", "11 Full-surface free candidate",
        "12 Relative width profiles", "13 A-F states", "", "", "",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    panels = [photo, semantic, policy_s, overlay(photo, base), repair_view, curb_view,
              object_view, occlusion_view, branch_view, hypothesis_view, free_view]
    for ax, panel in zip(axes.flat[:11], panels):
        ax.imshow(panel)
    ax = axes.flat[11]
    ax.axis("on")
    for index, hypothesis in enumerate(diagnostic["hypotheses"], start=1):
        values = hypothesis["relative_widths"]
        ax.plot(np.linspace(0, 1, len(values)), values, label=f"surface {index}")
    ax.set_xlabel("normalized image row")
    ax.set_ylabel("relative image width")
    ax.grid(alpha=0.25)
    if diagnostic["hypotheses"]:
        ax.legend(fontsize=7)
    axes.flat[12].text(
        0.01, 0.99,
        "\n".join(f"{key}: {value['state']}\n  {value['reason']}" for key, value in states.items()),
        va="top", family="monospace", fontsize=8,
    )
    fig.suptitle(f"{sample_id} | V2 A-F pilot | no new inference, no YOLO", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def build_class_policy(id2label: dict[int, str]) -> dict[str, object]:
    categories = {
        "basic_walkway": [15], "conditional_surface_inherit": list(REPAIR),
        "connection_only": list(CONNECTION), "boundary": list(BOUNDARY),
        "structure_context": list(STRUCTURE_CONTEXT), "fixed_occupancy_candidate": list(FIXED_OCCUPANCY),
        "moving_occupancy_candidate": list(MOVING_OCCUPANCY), "dynamic_occlusion": list(OCCLUSION),
        "hazard_or_unknown": list(HAZARD), "natural_nonwalk": list(NATURAL_NONWALK),
        "excluded_ground_observation": list(EXCLUDED_GROUND), "context_only": list(CONTEXT_ONLY),
    }
    flattened = [value for values in categories.values() for value in values]
    if sorted(flattened) != list(range(65)):
        raise RuntimeError("Class policy does not cover IDs 0..64 exactly once")
    return {
        "class_count": 65,
        "coverage_check": {"missing": [], "duplicates": []},
        "categories": {name: [{"id": key, "name": id2label[key]} for key in values] for name, values in categories.items()},
        "rules": {
            "pedestrian_area_auto_union": False, "boundary_suspect_auto_delete": False,
            "object_silhouette_is_ground_footprint": False, "traffic_cone_class_available": False,
        },
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing V2 pilot: {OUT}")
    artifacts = OUT / "artifacts"
    artifacts.mkdir(parents=True)
    items = {row["sample_id"]: row for row in json.loads(INPUTS.read_text(encoding="utf-8"))}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in config["id2label"].items()}
    class_policy = build_class_policy(id2label)
    (OUT / "class_policy.json").write_text(json.dumps(class_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan = """# V2 8-image pilot plan

- Reuse cached native-size Mapillary 65-class argmax maps; no model inference.
- Use Sidewalk only as the basic surface; do not union Pedestrian Area.
- Conditionally inherit Lane Marking - General, Catch Basin, and Manhole from surrounding Sidewalk.
- Generate Curb-based boundary-outside suspects without deleting them.
- Preserve the full repaired surface while showing starts, skeleton, endpoints, and junctions as diagnostics.
- Show Mapillary-only occupancy, occlusion, hazard, free-surface, and relative image-space width evidence.
- Do not use YOLO, VLM labeling, metric-width claims, or a destructive Main corridor.
"""
    (OUT / "PLAN.md").write_text(plan, encoding="utf-8")

    manifest_rows = []
    for sample_id in IDS:
        item = items[sample_id]
        image_path = ROOT / item["relative_path"]
        class_path = RAW_ROOT / f"{sample_id}.png"
        if not image_path.exists() or not class_path.exists():
            raise FileNotFoundError(sample_id)
        if sha256(image_path) != item["sha256"]:
            raise RuntimeError(f"Image hash mismatch: {sample_id}")
        manifest_rows.append({
            "sample_id": sample_id, "group": "core" if sample_id in CORE else "supplement",
            "image_path": item["relative_path"], "image_sha256": sha256(image_path),
            "class_map_path": str(class_path.relative_to(ROOT)).replace("\\", "/"),
            "class_map_sha256": sha256(class_path), "width": item["width"], "height": item["height"],
        })
    manifest = {"sample_count": 8, "new_model_inference": False, "yolo_used": False, "vlm_used": False,
                "samples": manifest_rows}
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_rows = []
    repair_rows = []
    review_cards = []
    for order, sample_id in enumerate(IDS, start=1):
        item = items[sample_id]
        image_path = ROOT / item["relative_path"]
        class_path = RAW_ROOT / f"{sample_id}.png"
        original = Image.open(image_path).convert("RGB")
        image = np.asarray(original)
        class_map = cv2.imread(str(class_path), cv2.IMREAD_UNCHANGED)
        if class_map is None or class_map.shape != image.shape[:2]:
            raise ValueError(f"Class-map shape mismatch: {sample_id}")
        art = artifacts / sample_id
        art.mkdir()
        shutil.copy2(image_path, art / "01_original.jpg")

        base = class_map == SIDEWALK
        repaired, additions, per_component_repair = repair_surface(class_map)
        suspect, per_component_boundary = boundary_suspect(class_map)
        connection = np.isin(class_map, CONNECTION)
        boundary = np.isin(class_map, BOUNDARY)
        occupancy = np.isin(class_map, OCCUPANCY)
        occlusion = np.isin(class_map, OCCLUSION)
        hazard = np.isin(class_map, HAZARD)
        free = repaired & ~occupancy & ~occlusion & ~hazard
        diagnostic = direction_diagnostics(repaired)
        policy = policy_map(class_map, additions)

        save_mask(art / "04_initial_sidewalk.png", base)
        save_mask(art / "05_repair_additions.png", additions)
        save_mask(art / "05_repaired_surface.png", repaired)
        save_mask(art / "06_boundary_outside_suspect.png", suspect)
        save_mask(art / "06_connection_info.png", connection)
        save_mask(art / "07_boundary.png", boundary)
        save_mask(art / "07_occupancy_candidate.png", occupancy)
        save_mask(art / "07_hazard.png", hazard)
        save_mask(art / "08_dynamic_occlusion.png", occlusion)
        save_mask(art / "11_free_surface_candidate.png", free)
        cv2.imwrite(str(art / "03_policy_layers.png"), cv2.cvtColor(policy, cv2.COLOR_RGB2BGR))
        analysis_size = (diagnostic["mask"].shape[1], diagnostic["mask"].shape[0])
        save_mask(art / "09_start_regions_analysis_size.png", diagnostic["start_mask"])
        save_mask(art / "09_skeleton_analysis_size.png", diagnostic["skeleton"])
        save_mask(art / "09_endpoints_analysis_size.png", diagnostic["endpoints"])
        save_mask(art / "09_junctions_analysis_size.png", diagnostic["junctions"])

        for row in per_component_repair:
            repair_rows.append({"sample_id": sample_id, **row})
        width_data = {key: value for key, value in diagnostic.items() if key not in {"mask", "skeleton", "start_mask", "endpoints", "junctions"}}
        width_data["analysis_width"] = analysis_size[0]
        width_data["analysis_height"] = analysis_size[1]
        (art / "10_branch_hypotheses.json").write_text(json.dumps(width_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (art / "06_boundary_components.json").write_text(json.dumps(per_component_boundary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        checks = acceptance(sample_id, class_map, additions, repaired)
        states = {
            "A": {"state": "observed", "reason": "cached_native_65_class_argmax"},
            "B": {"state": "observed", "reason": "complete_65_class_policy_mapping"},
            "C": {"state": "observed", "reason": f"repair={int(additions.sum())}; boundary_suspect={int(suspect.sum())}; auto_delete=0"},
            "D": {"state": str(diagnostic["state"]), "reason": str(diagnostic["reason"])},
            "E": {"state": "observed_candidate", "reason": f"occupancy={int(occupancy.sum())}; occlusion={int(occlusion.sum())}; hazard={int(hazard.sum())}"},
            "F": {"state": str(diagnostic["state"]), "reason": "relative_image_space_only; " + str(diagnostic["reason"])},
        }
        state_payload = {"states": states, "acceptance_evidence": checks,
                         "limitations": ["no_pixel_ground_truth", "no_metric_camera_calibration", "object_silhouette_is_not_ground_footprint"]}
        (art / "13_states.json").write_text(json.dumps(state_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        make_summary(art / "summary.png", sample_id, image, class_map, policy, repaired, additions, suspect,
                     occupancy, occlusion, hazard, diagnostic, free, states)

        counts = Counter(map(int, class_map.ravel()))
        row = {
            "sample_id": sample_id, "order": order, "group": "core" if sample_id in CORE else "supplement",
            "sidewalk_pixels": counts[15], "pedestrian_area_pixels_excluded": counts[11],
            "repair_added_pixels": int(additions.sum()), "lane_marking_repaired_pixels": int((additions & (class_map == 24)).sum()),
            "catch_basin_repaired_pixels": int((additions & (class_map == 36)).sum()),
            "manhole_repaired_pixels": int((additions & (class_map == 41)).sum()),
            "boundary_suspect_pixels_not_deleted": int(suspect.sum()), "occupancy_candidate_pixels": int(occupancy.sum()),
            "dynamic_occlusion_pixels": int(occlusion.sum()), "hazard_pixels": int(hazard.sum()),
            "free_surface_candidate_pixels": int(free.sum()), "surface_hypothesis_count": len(diagnostic["hypotheses"]),
            "direction_state": diagnostic["state"], "direction_reason": diagnostic["reason"],
            "base_sidewalk_preserved": checks["base_sidewalk_preserved"], "visual_acceptance": "pending_review",
        }
        result_rows.append(row)
        review_cards.append(
            f'<section><h2>{order}. {html.escape(sample_id)} <small>{row["group"]}</small></h2>'
            f'<p>Sidewalk {row["sidewalk_pixels"]:,} · repair +{row["repair_added_pixels"]:,} · '
            f'boundary suspect {row["boundary_suspect_pixels_not_deleted"]:,} (not deleted) · '
            f'occupancy {row["occupancy_candidate_pixels"]:,} · occlusion {row["dynamic_occlusion_pixels"]:,} · '
            f'direction {html.escape(str(row["direction_state"]))}</p>'
            f'<a href="artifacts/{html.escape(sample_id)}/summary.png"><img src="artifacts/{html.escape(sample_id)}/summary.png"></a>'
            f'<details><summary>Acceptance evidence</summary><pre>{html.escape(json.dumps(checks, ensure_ascii=False, indent=2))}</pre></details></section>'
        )

    with (OUT / "per_image_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)
    with (OUT / "repair_components.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(repair_rows[0]))
        writer.writeheader(); writer.writerows(repair_rows)

    page = (
        '<!doctype html><meta charset="utf-8"><title>Walkway V2 A-F pilot — 8 images</title>'
        '<style>body{font:15px system-ui;background:#10151d;color:#eef;max-width:1600px;margin:auto}'
        'section{background:#19212c;margin:22px;padding:18px;border-radius:12px}img{width:100%;height:auto}'
        'small{color:#8fd}p{line-height:1.5}pre{white-space:pre-wrap}.warning{background:#422b23;padding:14px;border-radius:10px}</style>'
        '<h1>Walkway Detection V2 A-F pilot — core 4 + supplement 4</h1>'
        '<p>No new inference, no YOLO, no VLM labeling, no destructive Main corridor.</p>'
        '<p class="warning">Red boundary-outside regions are review candidates and were not deleted. Width plots are relative image-space diagnostics, not metres.</p>'
        + "".join(review_cards)
    )
    (OUT / "review.html").write_text(page, encoding="utf-8")
    report = """# Walkway Detection V2 A-F Pilot Report

- Inputs: 8 cached native-size Mapillary argmax maps
- New model inference: no
- YOLO: no
- VLM labeling: no
- Basic surface: Sidewalk only
- Pedestrian Area automatic union: no
- Boundary suspect automatic deletion: no
- Width: image-space diagnostic only

The run is complete and ready for visual review. It is not a pixel-accuracy or product-performance result because the 8 images do not have complete 65-class or walkable-space ground truth.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
