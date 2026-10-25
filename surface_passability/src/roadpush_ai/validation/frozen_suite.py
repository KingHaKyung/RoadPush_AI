from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from roadpush_ai.occupancy.blockers import BANNER_ID, BANNER_SUPPORT_IDS, build_v3_blockers
from roadpush_ai.boundary.sidewalk import BoundaryMode, construct_boundary
from roadpush_ai.validation.surface_evaluator import AUDIT, HUMAN_LABELS, PRIORITY, ROOT, _bbox, contour, instance_panel, paint
from roadpush_ai.surface.preprocessing import build_shared_preprocessing, load_cached_inputs
from roadpush_ai.boundary.row_evaluator import evaluate_rows
from roadpush_ai.surface.structural_routes import StructuralRouteResult, build_structural_route_stage


OUTPUT_NAMES = {
    BoundaryMode.TEST1_OBSERVED_ONLY: "test1_observed_only_40_v5",
    BoundaryMode.TEST2_LOCAL_BRIDGE: "test2_route_connected_local_bridge_40_v1",
    BoundaryMode.TEST3_A2_ROW_SPAN: "test3_all_candidate_outer_row_span_40_v1",
}
TITLES = {
    BoundaryMode.TEST1_OBSERVED_ONLY: "V3 Test1 Observed Only",
    BoundaryMode.TEST2_LOCAL_BRIDGE: "V3 Test2 Route-Connected Local Bridge",
    BoundaryMode.TEST3_A2_ROW_SPAN: "V3 Test3 All-Candidate Outer Row Span",
}

GREEN = (45, 210, 95)
MAGENTA = (255, 45, 220)
DARK_BLUE = (20, 55, 180)
YELLOW = (255, 215, 30)
ORANGE = (255, 145, 35)
GRAY = (145, 145, 145)
LIGHT_PURPLE = (175, 90, 225)
CYAN = (40, 200, 255)
AMBER = (255, 180, 30)
RED = (255, 75, 65)


def _draw_path(image: np.ndarray, path, color, thickness: int = 4) -> None:
    if len(path) < 2:
        return
    points = np.asarray([(x, y) for y, x in path], np.int32)
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)


def _blank_panel(image: np.ndarray, message: str) -> np.ndarray:
    panel = np.full_like(image, 255)
    scale = max(0.55, min(1.0, image.shape[1] / 850.0))
    size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    point = (max(12, (image.shape[1] - size[0]) // 2), max(35, image.shape[0] // 2))
    cv2.putText(panel, message, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (55, 55, 55), 2, cv2.LINE_AA)
    return panel


def _row_panel(image: np.ndarray, evaluated: dict) -> np.ndarray:
    panel = paint(image, evaluated["free"], GREEN, 0.27)
    panel = contour(panel, evaluated["free"], MAGENTA, 2)
    for row in evaluated["rows"]:
        # Panel 9 is intentionally route-agnostic.  Show every passable
        # segment after blocker subtraction; route selection and reliability
        # remain JSON diagnostics and must not hide or recolor free space.
        for left, right in row.get("free_segments", []):
            cv2.line(panel, (left, row["y"]), (right, row["y"]), DARK_BLUE, 5, cv2.LINE_AA)
    return panel


def _obstacle_panel(image: np.ndarray, blockers: dict) -> np.ndarray:
    panel = paint(image, blockers["recognized_mask"], LIGHT_PURPLE, 0.50)
    return instance_panel(panel, blockers)


def _record_source_mask(record: dict, labels: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(labels, dtype=bool)
    box = _bbox(record)
    if not box:
        return mask
    x, y, width, height = map(int, box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(labels.shape[1], x + width), min(labels.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return mask
    class_id = record.get("class_id")
    if class_id is not None:
        mask[y0:y1, x0:x1] |= labels[y0:y1, x0:x1] == int(class_id)
    if record.get("kind") == "grounded_banner_support_cluster":
        ids = {BANNER_ID, *BANNER_SUPPORT_IDS}
        mask[y0:y1, x0:x1] |= np.isin(labels[y0:y1, x0:x1], tuple(sorted(ids)))
    if record.get("source") == "class_agnostic_corridor_hole":
        ids = [int(key) for key in record.get("class_mix", {})]
        if ids:
            mask[y0:y1, x0:x1] |= np.isin(labels[y0:y1, x0:x1], ids)
    return mask


def _transient_affects_rows(source: np.ndarray, rows: list[dict]) -> bool:
    for row in rows:
        y = int(row["y"])
        left, right = map(int, row["structural_sidewalk_span"])
        if source[y, left:right + 1].any():
            return True
    return False


def review_relevant_blockers(labels: np.ndarray, boundary: np.ndarray, blockers: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    """Limit panel 6 to objects relevant to this test's structural corridor.

    The actual blocker masks are preserved verbatim so panel 7 and all numeric
    evaluation remain independent of this visualization-only filter.
    """
    nearby = cv2.dilate(boundary.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    applied = blockers["final_mask"] & boundary
    recognized = np.zeros_like(boundary)
    filtered = []
    audit = []
    for record in blockers["objects"]:
        source = _record_source_mask(record, labels)
        box = _bbox(record)
        applied_in_box = False
        if box:
            x, y, width, height = map(int, box)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(labels.shape[1], x + width), min(labels.shape[0], y + height)
            applied_in_box = bool(applied[y0:y1, x0:x1].any())
        reasons = []
        rejected_hole = record.get("source") == "class_agnostic_corridor_hole" and not record.get("accepted")
        if bool((source & nearby).any()) and not rejected_hole:
            reasons.append("touches_structural_boundary")
        if record.get("accepted") and applied_in_box:
            reasons.append("approved_blocker_applied_in_boundary")
        if record.get("kind") == "transient_occluder" and _transient_affects_rows(source, rows):
            reasons.append("transient_affects_evaluated_row")
        if record.get("kind") == "grounded_banner_support_cluster" and (reasons or applied_in_box):
            reasons.append("banner_context_grouping")
        if reasons:
            visible = dict(record)
            visible["visual_relevance_reasons"] = sorted(set(reasons))
            filtered.append(visible)
            recognized |= source
        audit.append({
            "class_name": record.get("class_name"),
            "kind": record.get("kind"),
            "strategy": record.get("strategy"),
            "bbox": box,
            "shown": bool(reasons),
            "reasons": sorted(set(reasons)),
        })
    visual = dict(blockers)
    visual["recognized_mask"] = recognized
    visual["objects"] = filtered
    return visual, audit


def _applied_blocker_panel(image: np.ndarray, boundary: np.ndarray, blockers: dict) -> np.ndarray:
    panel = paint(image, boundary, GREEN, 0.18)
    panel = contour(panel, boundary, MAGENTA, 2)
    applied = blockers["final_mask"] & boundary
    selected_wheeled = blockers["wheeled_silhouette_mask"] & applied
    lower_silhouette = blockers["silhouette_mask"] & ~selected_wheeled & applied
    footprint = blockers["footprint_mask"] & applied
    hazard = blockers["hazard_mask"] & applied
    panel = paint(panel, hazard, RED, 0.78)
    panel = paint(panel, footprint, CYAN, 0.86)
    panel = paint(panel, lower_silhouette, AMBER, 0.86)
    panel = paint(panel, selected_wheeled, RED, 0.88)
    return panel


def render_nine_panel(
    artifact: Path,
    sample_id: str,
    shared,
    stage: StructuralRouteResult,
    blockers: dict,
    evaluated: dict,
    mode: BoundaryMode,
) -> None:
    p1 = contour(paint(shared.image, shared.structure_copy, GREEN, 0.38), shared.structure_copy, MAGENTA, 2)
    p2 = contour(paint(shared.image, stage.raw_boundary, GREEN, 0.38), stage.raw_boundary, MAGENTA, 2)
    p3 = contour(paint(shared.image, stage.raw_boundary, GREEN, 0.20), stage.raw_boundary, MAGENTA, 2)
    _draw_path(p3, shared.route_path, DARK_BLUE, 5)
    if mode != BoundaryMode.TEST1_OBSERVED_ONLY:
        _draw_path(p3, stage.route_path, YELLOW, 4)

    if mode == BoundaryMode.TEST1_OBSERVED_ONLY:
        p4 = _blank_panel(shared.image, "not applied to Test1")
        p5 = _blank_panel(shared.image, "not applied to Test1")
    else:
        p4 = shared.image.copy()
        p4 = paint(p4, stage.vetoed_candidate, GRAY, 0.78)
        p4 = paint(p4, stage.raw_candidate & ~stage.vetoed_candidate, ORANGE, 0.78)
        p5 = contour(paint(shared.image, stage.pruned_boundary, GREEN, 0.38), stage.pruned_boundary, MAGENTA, 2)
        p5 = paint(p5, stage.approved_removal, ORANGE, 0.82)

    review_blockers, _ = review_relevant_blockers(
        shared.labels, stage.pruned_boundary, blockers, evaluated["rows"]
    )
    p6 = _obstacle_panel(shared.image, review_blockers)
    p7 = _applied_blocker_panel(shared.image, stage.pruned_boundary, blockers)
    # Panel 8 is explicitly the final free mask.  Its contour must therefore
    # include blocker-created internal cuts/holes, not the pre-blocker
    # structural component.
    p8 = contour(paint(shared.image, evaluated["free"], GREEN, 0.44), evaluated["free"], MAGENTA, 3)
    p9 = _row_panel(shared.image, evaluated)

    panels = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
    titles = [
        "1 Shared repaired + structure_copy input",
        "2 Raw structural boundary before side pruning",
        "3 Routes: dark blue=Test1 observed, yellow=test-specific",
        "4 Raw side candidates: orange=accepted, gray=veto",
        "5 Boundary after pruning: orange=actually removed",
        "6 Relevant segmentation + instances + blocker strategy",
        "7 Applied blocker inside pruned boundary only",
        "8 Final free sidewalk + true final boundary",
        "9 Passable width: every post-blocker free segment",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), constrained_layout=True, facecolor="white")
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=8.8, color="black")
        axis.axis("off")
    axes.flat[5].legend(
        handles=[
            Patch(color="#af5ae1", label="recognized segmentation"),
            Patch(color="#28c8ff", label="applied footprint"),
            Patch(color="#ffb41e", label="applied lower-19% silhouette"),
            Patch(color="#ff4b41", label="applied selected wheeled silhouette/hazard"),
        ],
        loc="lower left", fontsize=6.8, framealpha=0.82,
    )
    displacement = stage.route_displacement
    dx = displacement.get("mean_abs_dx_px")
    dx_text = "n/a" if dx is None else f"{dx:.1f}px"
    fig.suptitle(
        f"{sample_id} | {TITLES[mode]} | route mean |dx| {dx_text} | "
        f"candidates {int(stage.raw_candidate.sum()):,}px | removed {int(stage.approved_removal.sum()):,}px",
        fontsize=12.5, color="black",
    )
    fig.savefig(artifact / "summary.png", dpi=112, facecolor="white")
    plt.close(fig)


def _clean_route(route: dict) -> dict:
    return {
        "start": list(route["start"]) if route.get("start") else None,
        "target_y": int(route.get("target_y", 0)),
        "main_index": route.get("main_index"),
        "angle_deg": route.get("angle"),
        "vertical_progress": route.get("vertical_progress"),
        "high_confidence": bool(route.get("high_confidence")),
        "alternative_route_count": max(0, len(route.get("paths", [])) - 1),
        "observed_start_reused": route.get("observed_start_reused", True),
        "observed_direction_angle_deg": route.get("observed_direction_angle_deg"),
        "direction_constraint": route.get("direction_constraint", "shared_observed_route"),
    }


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def _row_changes(reference: list[dict], current: list[dict], key: str) -> int:
    def normalized(row):
        value = row.get(key)
        return tuple(tuple(x) if isinstance(x, list) else x for x in value) if isinstance(value, list) else value
    a = {row["y"]: normalized(row) for row in reference}
    b = {row["y"]: normalized(row) for row in current}
    return sum(a.get(y) != b.get(y) for y in set(a) | set(b))


def _save_masks(artifact: Path, shared, stage, blockers, evaluated) -> None:
    masks = {
        "01_structure_copy_input.png": shared.structure_copy,
        "02_raw_structural_boundary.png": stage.raw_boundary,
        "03_side_raw_candidate.png": stage.raw_candidate,
        "04_side_vetoed_candidate.png": stage.vetoed_candidate,
        "05_side_approved_removal.png": stage.approved_removal,
        "06_pruned_structural_boundary.png": stage.pruned_boundary,
        "07_applied_final_blocker.png": blockers["final_mask"] & stage.pruned_boundary,
        "08_final_free_sidewalk.png": evaluated["free"],
    }
    for name, mask in masks.items():
        cv2.imwrite(str(artifact / name), mask.astype(np.uint8) * 255)


def _make_result(item, display_order, mode, boundary, stage, blockers, evaluated, blocker_digest) -> dict:
    strategies = Counter(row.get("strategy", "unknown") for row in blockers["objects"])
    accepted = [row for row in blockers["objects"] if row.get("accepted")]
    veto_reasons = Counter(row.get("veto_reason") for row in stage.candidate_rows if row.get("veto_reason"))
    applied = blockers["final_mask"] & stage.pruned_boundary
    return {
        "sample_id": item["sample_id"],
        "display_order": display_order,
        "original_order": item.get("order"),
        "split": item.get("split"),
        "test": mode.value,
        "boundary_calls": stage.calls,
        "raw_structural_boundary_pixels": int(stage.raw_boundary.sum()),
        "test_boundary_addition_pixels": int(boundary.additions.sum()),
        "side_raw_candidate_pixels": int(stage.raw_candidate.sum()),
        "side_vetoed_candidate_pixels": int(stage.vetoed_candidate.sum()),
        "side_removed_pixels": int(stage.approved_removal.sum()),
        "side_accepted_groups": stage.accepted_groups,
        "side_veto_reason_counts": dict(veto_reasons),
        "side_candidate_rows": stage.candidate_rows,
        "pruned_structural_boundary_pixels": int(stage.pruned_boundary.sum()),
        "common_observed_route": None,
        "test_structural_route": _clean_route(stage.route),
        "route_displacement": stage.route_displacement,
        "recognized_relevant_object_count": len(blockers["objects"]),
        "accepted_blocker_object_count": len(accepted),
        "selected_strategy_counts": dict(strategies),
        "objects": blockers["objects"],
        "yolo": blockers["yolo"],
        "shared_blocker_mask_sha256": blocker_digest,
        "applied_final_blocker_pixels": int(applied.sum()),
        "strategy_overlap_pixels": blockers["strategy_overlap_pixels"],
        "transient_occluded_row_count": evaluated["transient_occluded_rows"],
        "boundary_uncertain_row_count": evaluated["boundary_uncertain_rows"],
        "branch_uncertain_row_count": evaluated["branch_uncertain_rows"],
        "rule_violations": blockers["violations"],
        "rows": evaluated["rows"],
        "blocker_policy": blockers["policy"],
        "new_mapillary_inference": False,
        "new_yolo_inference": bool(blockers["yolo"]["inference_run"]),
        "post_blocker_navigation_route": None,
        "scalar_width_summary": None,
    }


def _card_html(row: dict, output_name: str) -> str:
    sid = html.escape(row["sample_id"])
    labels = "".join(
        f'<label><input type="radio" name="review-{sid}" value="{html.escape(label)}"> {html.escape(label)}</label>'
        for label in HUMAN_LABELS
    )
    dx = row["route_displacement"].get("mean_abs_dx_px")
    dx_text = "n/a" if dx is None else f"{dx:.1f}px"
    veto = ", ".join(f"{k}:{v}" for k, v in row["side_veto_reason_counts"].items()) or "none"
    return (
        f'<section id="{sid}"><h2>{row["display_order"]:02d}. {sid} '
        f'<small>{html.escape(str(row.get("split") or ""))}</small></h2>'
        f'<p>route mean |dx| {dx_text} · raw candidates {row["side_raw_candidate_pixels"]:,}px · '
        f'removed {row["side_removed_pixels"]:,}px · veto {html.escape(veto)} · '
        f'applied blocker {row["applied_final_blocker_pixels"]:,}px · rows {len(row["rows"])} · '
        f'YOLO {html.escape(row["yolo"]["result"])}</p>'
        f'<details><summary>Human review label</summary><div class="labels">{labels}</div></details>'
        f'<a href="artifacts/{sid}/result.json">result.json</a>'
        f'<img loading="lazy" src="artifacts/{sid}/summary.png"></section>'
    )


def _review_page(results: list[dict], validation: dict, mode: BoundaryMode) -> str:
    nav = " ".join(f'<a href="#{html.escape(sid)}">{html.escape(sid)}</a>' for sid in PRIORITY)
    cards = "".join(_card_html(row, OUTPUT_NAMES[mode]) for row in results)
    storage = f"v3-{mode.value}-9panel"
    return f'''<!doctype html><html><meta charset="utf-8"><title>{html.escape(TITLES[mode])}</title>
<style>body{{font:15px system-ui;background:#101722;color:#eef;max-width:1800px;margin:auto;padding:18px}}header{{position:sticky;top:0;background:#101722ee;padding:8px;z-index:2}}section{{background:#192331;margin:24px 0;padding:18px;border-radius:12px}}img{{display:block;width:100%;height:auto;margin-top:12px;background:white;border-radius:7px}}a{{color:#75ddff}}small{{color:#8ff}}.labels{{display:flex;gap:14px;flex-wrap:wrap;padding:12px 0}}.warn{{color:#ffd477}}</style>
<header><h1>{html.escape(TITLES[mode])} — 40 images — fixed 3×3 review</h1><p>Automatic status: {html.escape(validation['status'])}. Cached Mask2Former maps; CPU-only conditional YOLO; no scalar-width summary.</p><nav>{nav}</nav></header>
<p class="warn">Human labels: {', '.join(HUMAN_LABELS)}. Reviews do not tune thresholds or modify results.</p>{cards}
<script>document.querySelectorAll('input[type=radio]').forEach(x=>{{const k='{storage}-'+x.name;if(localStorage.getItem(k)===x.value)x.checked=true;x.addEventListener('change',()=>localStorage.setItem(k,x.value));}});</script></html>'''


def _comparison_page(
    all_results: dict[BoundaryMode, list[dict]],
    validation: dict,
    output_names: dict[BoundaryMode, str] | None = None,
) -> str:
    output_names = output_names or OUTPUT_NAMES
    by_mode = {mode: {row["sample_id"]: row for row in rows} for mode, rows in all_results.items()}
    order = [row["sample_id"] for row in all_results[BoundaryMode.TEST1_OBSERVED_ONLY]]
    cards = []
    for index, sid in enumerate(order, 1):
        rows = [by_mode[mode][sid] for mode in BoundaryMode]
        cells = []
        for mode, row in zip(BoundaryMode, rows):
            dx = row["route_displacement"].get("mean_abs_dx_px")
            dx_text = "n/a" if dx is None else f"{dx:.1f}px"
            cells.append(
                f'<div><h3>{html.escape(TITLES[mode])}</h3><p>route |dx| {dx_text} · '
                f'removed {row["side_removed_pixels"]:,}px · blocker {row["applied_final_blocker_pixels"]:,}px</p>'
                f'<a href="../{output_names[mode]}/artifacts/{html.escape(sid)}/result.json">result.json</a>'
                f'<img loading="lazy" src="../{output_names[mode]}/artifacts/{html.escape(sid)}/summary.png"></div>'
            )
        cards.append(f'<section id="{html.escape(sid)}"><h2>{index:02d}. {html.escape(sid)}</h2><div class="grid">{"".join(cells)}</div></section>')
    nav = " ".join(f'<a href="#{html.escape(sid)}">{html.escape(sid)}</a>' for sid in PRIORITY)
    return f'''<!doctype html><html><meta charset="utf-8"><title>RoadPush V3 Test1/Test2/Test3 comparison</title>
<style>body{{font:15px system-ui;background:#101722;color:#eef;max-width:2200px;margin:auto;padding:18px}}header{{position:sticky;top:0;background:#101722ee;padding:8px;z-index:2}}section{{background:#192331;margin:24px 0;padding:18px;border-radius:12px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}img{{width:100%;height:auto;background:white;border-radius:6px}}a{{color:#75ddff}}h3{{margin-bottom:4px}}p{{line-height:1.35}}@media(max-width:1200px){{.grid{{grid-template-columns:1fr}}}}</style>
<header><h1>RoadPush V3 — Test1 / Test2 / Test3</h1><p>Automatic status: {html.escape(validation['status'])}. Fixed 3×3 sheets shown side by side; priority samples first.</p><nav>{nav}</nav></header>{''.join(cards)}</html>'''


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "display_order", "original_order", "sample_id", "split", "test",
        "test_boundary_addition_pixels", "side_raw_candidate_pixels", "side_vetoed_candidate_pixels",
        "side_removed_pixels", "applied_final_blocker_pixels", "boundary_rows_changed_relative_to_test1",
        "free_space_rows_changed_relative_to_test1", "route_changed_relative_to_test1",
        "transient_occluded_row_count", "boundary_uncertain_row_count", "branch_uncertain_row_count",
        "new_yolo_inference",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def run_suite() -> dict:
    output_root = ROOT / "outputs" / "walkway_detection_v3"
    final_paths = {mode: output_root / name for mode, name in OUTPUT_NAMES.items()}
    comparison_path = output_root / "comparison_route_connected_vs_all_candidate_40_v1"
    for path in [*final_paths.values(), comparison_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V3 output: {path}")
    staging_root = ROOT / "tmp" / "v3_test1_test2_test3_40_v1_staging"
    if staging_root.exists():
        raise FileExistsError(f"Refusing to overwrite staging: {staging_root}")
    stage_paths = {mode: staging_root / OUTPUT_NAMES[mode] for mode in BoundaryMode}
    stage_comparison = staging_root / comparison_path.name
    for path in [*stage_paths.values(), stage_comparison]:
        (path / "artifacts").mkdir(parents=True)

    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    priority_index = {sid: index for index, sid in enumerate(PRIORITY)}
    items = sorted(items, key=lambda row: (priority_index.get(row["sample_id"], len(PRIORITY)), row.get("order", 999)))
    model_holder: dict = {}
    all_results: dict[BoundaryMode, list[dict]] = {mode: [] for mode in BoundaryMode}

    for display_order, item in enumerate(items, 1):
        print(f"[{display_order:02d}/{len(items):02d}] {item['sample_id']}", flush=True)
        image, labels = load_cached_inputs(item)
        shared = build_shared_preprocessing(image, labels)
        blockers = build_v3_blockers(shared, model_holder, wheeled_silhouette_angle_max_deg=85.0)
        blocker_digest = _mask_digest(blockers["final_mask"])
        sample_results = {}
        sample_evaluated = {}
        sample_stages = {}
        for mode in BoundaryMode:
            boundary = construct_boundary(shared, mode)
            stage = build_structural_route_stage(boundary, shared)
            evaluated = evaluate_rows(stage.pruned_boundary, blockers, shared, stage.route, stage.route_path)
            artifact = stage_paths[mode] / "artifacts" / item["sample_id"]
            artifact.mkdir()
            _save_masks(artifact, shared, stage, blockers, evaluated)
            render_nine_panel(artifact, item["sample_id"], shared, stage, blockers, evaluated, mode)
            result = _make_result(item, display_order, mode, boundary, stage, blockers, evaluated, blocker_digest)
            review_blockers, visibility_audit = review_relevant_blockers(
                shared.labels, stage.pruned_boundary, blockers, evaluated["rows"]
            )
            result["panel6_total_object_records"] = len(blockers["objects"])
            result["panel6_displayed_object_records"] = len(review_blockers["objects"])
            result["panel6_object_visibility_audit"] = visibility_audit
            result["common_observed_route"] = _clean_route(shared.route)
            sample_results[mode] = result
            sample_evaluated[mode] = evaluated
            sample_stages[mode] = stage

        reference = sample_results[BoundaryMode.TEST1_OBSERVED_ONLY]
        reference_eval = sample_evaluated[BoundaryMode.TEST1_OBSERVED_ONLY]
        reference_stage = sample_stages[BoundaryMode.TEST1_OBSERVED_ONLY]
        for mode in BoundaryMode:
            result = sample_results[mode]
            evaluated = sample_evaluated[mode]
            stage = sample_stages[mode]
            result["raw_boundary_changed_relative_to_test1"] = bool(np.any(stage.raw_boundary != reference_stage.raw_boundary))
            result["pruned_boundary_changed_relative_to_test1"] = bool(np.any(stage.pruned_boundary != reference_stage.pruned_boundary))
            result["free_mask_changed_relative_to_test1"] = bool(np.any(evaluated["free"] != reference_eval["free"]))
            result["boundary_rows_changed_relative_to_test1"] = _row_changes(reference_eval["rows"], evaluated["rows"], "structural_sidewalk_span")
            result["free_space_rows_changed_relative_to_test1"] = _row_changes(reference_eval["rows"], evaluated["rows"], "free_segments")
            result["route_changed_relative_to_test1"] = stage.route_path != reference_stage.route_path
            artifact = stage_paths[mode] / "artifacts" / item["sample_id"]
            (artifact / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            all_results[mode].append(result)

    blocker_source = (Path(__file__).parent / "blockers.py").read_bytes()
    route_source = (Path(__file__).parent / "structural_routes.py").read_bytes()
    failures = []
    if len(items) != 40:
        failures.append("sample_count")
    expected_calls = {
        BoundaryMode.TEST1_OBSERVED_ONLY: (False, False, False, False, False),
        BoundaryMode.TEST2_LOCAL_BRIDGE: (True, False, True, True, False),
        BoundaryMode.TEST3_A2_ROW_SPAN: (False, True, True, False, True),
    }
    for mode, rows in all_results.items():
        bridge, a2, pruning, component_before, all_candidate_span = expected_calls[mode]
        for row in rows:
            calls = row["boundary_calls"]
            if (
                calls["v2_horizontal_obstacle_bridge"] != bridge
                or calls["a2_row_span_completion"] != a2
                or calls["side_branch_pruning"] != pruning
                or calls["route_component_selection_before_completion"] != component_before
                or calls["all_candidate_outer_row_span"] != all_candidate_span
            ):
                failures.append(f"call_graph_mismatch:{mode.value}:{row['sample_id']}")
            if calls["v2_1_natural_restoration"] or calls["a1_corridor_hole_restoration"]:
                failures.append(f"forbidden_restoration:{mode.value}:{row['sample_id']}")
            if calls["blocker_used_for_side_pruning"]:
                failures.append(f"blocker_used_for_side_pruning:{mode.value}:{row['sample_id']}")
            if row["blocker_policy"]["wheeled_silhouette_angle_max_deg"] != 85.0:
                failures.append(f"selector_threshold:{mode.value}:{row['sample_id']}")
            if row["blocker_policy"]["global_cycle_closure"] or row["blocker_policy"]["natural_restoration"]:
                failures.append(f"forbidden_blocker_policy:{mode.value}:{row['sample_id']}")
            if row["rule_violations"]:
                failures.append(f"rule_violation:{mode.value}:{row['sample_id']}")
    for index in range(len(items)):
        digests = {all_results[mode][index]["shared_blocker_mask_sha256"] for mode in BoundaryMode}
        if len(digests) != 1:
            failures.append(f"blocker_mismatch:{items[index]['sample_id']}")

    change_counts = {}
    for mode, rows in all_results.items():
        change_counts[mode.value] = {
            "raw_boundary_changed_images": sum(row["raw_boundary_changed_relative_to_test1"] for row in rows),
            "pruned_boundary_changed_images": sum(row["pruned_boundary_changed_relative_to_test1"] for row in rows),
            "route_changed_images": sum(row["route_changed_relative_to_test1"] for row in rows),
            "side_pruning_changed_images": sum(row["side_removed_pixels"] > 0 for row in rows),
            "free_mask_changed_images": sum(row["free_mask_changed_relative_to_test1"] for row in rows),
            "boundary_row_changed_images": sum(row["boundary_rows_changed_relative_to_test1"] > 0 for row in rows),
            "free_space_row_changed_images": sum(row["free_space_rows_changed_relative_to_test1"] > 0 for row in rows),
            "total_side_removed_pixels": sum(row["side_removed_pixels"] for row in rows),
        }
    flat_results = [row for rows in all_results.values() for row in rows]
    cached_maps_reused = all(not row["new_mapillary_inference"] for row in flat_results)
    cpu_only = os.environ.get("ROADPUSH_YOLO_DEVICE", "").lower() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    global_cycle_closure_used = any(
        row["blocker_policy"].get("global_cycle_closure")
        or any(obj.get("kind") == "cycle_gap_closure" for obj in row["objects"])
        for row in flat_results
    )
    natural_restoration_used = any(
        row["boundary_calls"].get("v2_1_natural_restoration")
        or row["blocker_policy"].get("natural_restoration")
        or any(obj.get("natural_restoration_applied") for obj in row["objects"])
        for row in flat_results
    )
    post_blocker_navigation_route_present = any(row.get("post_blocker_navigation_route") is not None for row in flat_results)
    scalar_width_summary_present = any(row.get("scalar_width_summary") is not None for row in flat_results)
    side_pruning_uses_blocker = any(row["boundary_calls"].get("blocker_used_for_side_pruning") for row in flat_results)
    if not cached_maps_reused:
        failures.append("new_mask2former_inference_detected")
    if not cpu_only:
        failures.append("cpu_runtime_contract_not_enforced")
    if global_cycle_closure_used:
        failures.append("global_cycle_closure_detected")
    if natural_restoration_used:
        failures.append("natural_restoration_detected")
    if post_blocker_navigation_route_present:
        failures.append("post_blocker_navigation_route_detected")
    if scalar_width_summary_present:
        failures.append("scalar_width_summary_detected")
    if side_pruning_uses_blocker:
        failures.append("side_pruning_blocker_dependency_detected")
    validation = {
        "status": "automatic_checks_passed_visual_review_pending" if not failures else "automatic_checks_failed",
        "processed_samples_per_test": {mode.value: len(rows) for mode, rows in all_results.items()},
        "priority_order": PRIORITY,
        "changed_image_counts_relative_to_test1": change_counts,
        "cached_mask2former_maps_reused": cached_maps_reused,
        "cpu_only": cpu_only,
        "runtime_device_evidence": {
            "ROADPUSH_YOLO_DEVICE": os.environ.get("ROADPUSH_YOLO_DEVICE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "conditional_yolo_samples": [row["sample_id"] for row in all_results[BoundaryMode.TEST1_OBSERVED_ONLY] if row["new_yolo_inference"]],
        "same_blocker_builder_and_mask_all_tests": not any(x.startswith("blocker_mismatch:") for x in failures),
        "shared_blocker_builder_sha256": hashlib.sha256(blocker_source).hexdigest(),
        "structural_route_stage_sha256": hashlib.sha256(route_source).hexdigest(),
        "wheeled_selector_angle_from_vertical_deg": 85.0,
        "global_cycle_closure_used": global_cycle_closure_used,
        "natural_restoration_used": natural_restoration_used,
        "post_blocker_navigation_route_present": post_blocker_navigation_route_present,
        "scalar_width_summary_present": scalar_width_summary_present,
        "side_pruning_uses_blocker_or_footprint": side_pruning_uses_blocker,
        "failures": sorted(set(failures)),
    }

    audit_lines = [
        "# V3 frozen call-graph audit",
        "",
        "| Stage / function | Test1 | Test2 | Test3 |",
        "|---|---:|---:|---:|",
        "| `build_shared_preprocessing`: semantic repair + `structure_copy` | on | on | on |",
        "| common Test1 observed route | reference | seed/reference | seed/reference |",
        "| V2 short horizontal obstacle bridge | off | on | off |",
        "| route-component selection before completion | off | on | off |",
        "| all-candidate outer row-span completion | off | off | on |",
        "| V2.1 natural restoration | off | off | off |",
        "| A1 restoration | off | off | off |",
        "| test-specific structural-route recompute | off | on | on |",
        "| side-branch pruning | off | on | on |",
        "| shared blocker builder (85° selector) | on | on | on |",
        "| row evaluator | on | on | on |",
        "| post-blocker navigation route | off | off | off |",
        "| scalar width summary | off | off | off |",
        "",
        "Test2 is Route-Connected Local Bridge. Test3 extracts outer row endpoints from the complete `structure_copy & ~curb_suspect` mask before any route-component selection.",
        "",
        "Side pruning receives a zero blocker mask; `frame_truncated` is a hard veto, only the upper 70% is eligible, and at least two consecutive sampled rows are required.",
    ]
    (stage_comparison / "CALL_GRAPH_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    (stage_comparison / "completion_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage_comparison / "comparison.html").write_text(_comparison_page(all_results, validation), encoding="utf-8")
    for mode, rows in all_results.items():
        test_validation = {**validation, "test": mode.value, "test_change_counts": change_counts[mode.value]}
        path = stage_paths[mode]
        (path / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (path / "completion_validation.json").write_text(json.dumps(test_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (path / "review.html").write_text(_review_page(rows, test_validation, mode), encoding="utf-8")
        _write_csv(path / "summary.csv", rows)

    output_root.mkdir(parents=True, exist_ok=True)
    for mode, final_path in final_paths.items():
        shutil.move(str(stage_paths[mode]), str(final_path))
    shutil.move(str(stage_comparison), str(comparison_path))
    shutil.rmtree(staging_root)
    return {"comparison": str(comparison_path / "comparison.html"), **validation}
