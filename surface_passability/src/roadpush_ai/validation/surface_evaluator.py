from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.occupancy.blockers import build_v3_blockers
from roadpush_ai.boundary.sidewalk import BoundaryMode, construct_boundary
from roadpush_ai.surface.preprocessing import build_shared_preprocessing, load_cached_inputs
from roadpush_ai.boundary.row_evaluator import evaluate_rows


ROOT = get_settings().project_root
AUDIT = ROOT / "outputs" / "walkway_detection_ablation_v2" / "model_audit_40" / "audit_manifest.json"
PRIORITY = [
    "commons_01", "kr_trash_01", "commons_04", "IMG_4282",
    "kr_electric_scooter_02", "kr_bollard_01", "IMG_4508", "IMG_4586",
    "kr_electric_scooter_01", "kr_bicycle_01", "kr_bicycle_02",
    "kr_clothing_bin_02", "kr_signboard_01",
]
HUMAN_LABELS = [
    "pass", "over-blocked", "under-blocked", "instance-split failure",
    "sidewalk-boundary failure", "uncertain",
]


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1.0 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def contour(image: np.ndarray, mask: np.ndarray, color=(255, 45, 220), thickness=3) -> np.ndarray:
    result = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result, contours, -1, color, thickness, cv2.LINE_AA)
    return result


def _bbox(record: dict):
    return record.get("bbox") or record.get("source_bbox") or record.get("component_bbox") or record.get("intrusion_bbox")


def instance_panel(image: np.ndarray, blockers: dict) -> np.ndarray:
    panel = image.copy()
    colors = {
        "footprint": (40, 200, 255),
        "selected_silhouette": (255, 75, 65),
        "lower_19_percent_silhouette": (255, 145, 35),
        "transient_rows_only": (180, 90, 235),
        "context_only": (145, 145, 145),
        "full_semantic_hazard": (235, 65, 80),
    }
    for index, record in enumerate(blockers["objects"], 1):
        box = _bbox(record)
        if not box or len(box) != 4:
            continue
        x, y, width, height = map(int, box)
        strategy = record.get("strategy", "context_only")
        color = colors.get(strategy, (40, 200, 255))
        cv2.rectangle(panel, (x, y), (x + width - 1, y + height - 1), color, 2, cv2.LINE_AA)
        cv2.putText(panel, str(index), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        if record.get("axis_center") and record.get("axis_vector"):
            cx, cy = record["axis_center"]
            vx, vy = record["axis_vector"]
            length = max(18, round(max(width, height) * 0.28))
            p0 = (round(cx - vx * length), round(cy - vy * length))
            p1 = (round(cx + vx * length), round(cy + vy * length))
            cv2.arrowedLine(panel, p0, p1, color, 3, cv2.LINE_AA, tipLength=0.20)
    return panel


def row_panel(image: np.ndarray, free: np.ndarray, rows: list[dict]) -> np.ndarray:
    panel = paint(image, free, (75, 190, 235), 0.28)
    for row in rows:
        y = row["y"]
        for left, right in row["free_segments"]:
            cv2.line(panel, (left, y), (right, y), (85, 210, 255), 2, cv2.LINE_AA)
        selected = row["selected_route_segment"]
        if selected:
            color = (180, 90, 235) if row["transient_occlusion"] else (35, 75, 255)
            cv2.line(panel, (selected[0], y), (selected[1], y), color, 5, cv2.LINE_AA)
    return panel


def render_six_panel(
    artifact: Path,
    sample_id: str,
    shared,
    boundary,
    blockers: dict,
    evaluated: dict,
    experiment_title: str,
) -> None:
    p1 = paint(shared.image, shared.structure_copy, (45, 210, 95), 0.38)
    p1 = contour(p1, shared.observed_component)
    p2 = paint(shared.image, blockers["recognized_mask"], (175, 90, 225), 0.58)
    p3 = instance_panel(shared.image, blockers)
    applied = boundary.mask
    wheeled_silhouette = blockers.get(
        "wheeled_silhouette_mask", np.zeros_like(applied, dtype=bool)
    ) & applied
    other_silhouette = blockers["silhouette_mask"] & ~wheeled_silhouette & applied
    p4 = paint(shared.image, blockers["hazard_mask"] & applied, (235, 65, 80), 0.70)
    p4 = paint(p4, blockers["footprint_mask"] & applied, (40, 200, 255), 0.84)
    p4 = paint(p4, other_silhouette, (255, 145, 35), 0.84)
    p4 = paint(p4, wheeled_silhouette, (255, 75, 65), 0.84)
    p5 = contour(paint(shared.image, evaluated["free"], (45, 210, 95), 0.44), evaluated["component"])
    p6 = row_panel(shared.image, evaluated["free"], evaluated["rows"])

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    titles = [
        "1 Repaired + structure_copy sidewalk input",
        "2 Recognized relevant obstacle segmentation",
        "3 Object instances + selected strategy",
        "4 Applied blocker (cyan footprint / orange lower-19% / red selected wheeled silhouette)",
        "5 Resulting free sidewalk + true structural boundary",
        "6 Row-wise free space (blue selected / purple transient-occluded)",
    ]
    for axis, title, panel in zip(axes.flat, titles, [p1, p2, p3, p4, p5, p6]):
        axis.imshow(panel)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    axes.flat[2].legend(
        handles=[
            Patch(color="#28c8ff", label="footprint"),
            Patch(color="#ff9123", label="lower-19% silhouette"),
            Patch(color="#ff4b41", label="selected wheeled silhouette"),
            Patch(color="#b45aeb", label="transient/context"),
            Patch(color="#eb4150", label="hazard"),
        ],
        loc="lower left", fontsize=7, framealpha=0.75,
    )
    fig.suptitle(
        f"{sample_id} | {experiment_title} | blockers {int(blockers['final_mask'].sum()):,}px | "
        f"rows {len(evaluated['rows'])}",
        fontsize=13,
    )
    fig.savefig(artifact / "summary.png", dpi=115)
    plt.close(fig)


def process(item: dict, output: Path, model_holder: dict, display_order: int,
            blocker_kwargs: dict | None = None,
            experiment_title: str = "V3 Test1 Observed Only") -> dict:
    image, labels = load_cached_inputs(item)
    shared = build_shared_preprocessing(image, labels)
    boundary = construct_boundary(shared, BoundaryMode.TEST1_OBSERVED_ONLY)
    blockers = build_v3_blockers(shared, model_holder, **(blocker_kwargs or {}))
    evaluated = evaluate_rows(boundary.mask, blockers, shared)

    artifact = output / "artifacts" / item["sample_id"]
    artifact.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(artifact / "01_structure_copy_input.png"), shared.structure_copy.astype(np.uint8) * 255)
    cv2.imwrite(str(artifact / "02_observed_boundary.png"), boundary.mask.astype(np.uint8) * 255)
    cv2.imwrite(str(artifact / "03_recognized_obstacles.png"), blockers["recognized_mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(artifact / "04_final_blocker.png"), blockers["final_mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(artifact / "05_free_sidewalk.png"), evaluated["free"].astype(np.uint8) * 255)
    render_six_panel(
        artifact, item["sample_id"], shared, boundary, blockers, evaluated,
        experiment_title,
    )

    strategies = Counter(row.get("strategy", "unknown") for row in blockers["objects"])
    accepted = [row for row in blockers["objects"] if row.get("accepted")]
    result = {
        "sample_id": item["sample_id"],
        "display_order": display_order,
        "original_order": item.get("order"),
        "split": item.get("split"),
        "boundary_mode": boundary.mode.value,
        "boundary_calls": boundary.calls,
        "recognized_relevant_object_count": len(blockers["objects"]),
        "accepted_blocker_object_count": len(accepted),
        "selected_strategy_counts": dict(strategies),
        "objects": blockers["objects"],
        "yolo": blockers["yolo"],
        "final_blocker_pixels": int(blockers["final_mask"].sum()),
        "strategy_overlap_pixels": blockers["strategy_overlap_pixels"],
        "boundary_rows_changed_relative_to_test1": 0,
        "free_space_rows_changed_relative_to_test1": 0,
        "transient_occluded_row_count": evaluated["transient_occluded_rows"],
        "boundary_uncertain_row_count": evaluated["boundary_uncertain_rows"],
        "branch_uncertain_row_count": evaluated["branch_uncertain_rows"],
        "rule_violations": blockers["violations"],
        "rows": evaluated["rows"],
        "blocker_policy": blockers["policy"],
        "new_mapillary_inference": False,
        "new_yolo_inference": bool(blockers["yolo"]["inference_run"]),
        "scalar_width_summary": None,
    }
    (artifact / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _review_page(results: list[dict], validation: dict, experiment_title: str) -> str:
    cards = []
    choices = "".join(
        f'<label><input type="radio" name="review-{{sid}}" value="{html.escape(label)}"> {html.escape(label)}</label>'
        for label in HUMAN_LABELS
    )
    for row in results:
        sid = html.escape(row["sample_id"])
        radio = choices.replace("{sid}", sid)
        violations = ", ".join(row["rule_violations"]) or "none"
        cards.append(
            f'<section id="{sid}"><h2>{row["display_order"]:02d}. {sid} '
            f'<small>{html.escape(str(row.get("split") or ""))}</small></h2>'
            f'<p>objects {row["recognized_relevant_object_count"]} · accepted {row["accepted_blocker_object_count"]} · '
            f'blocker {row["final_blocker_pixels"]:,} px · transient rows {row["transient_occluded_row_count"]} · '
            f'boundary-uncertain {row["boundary_uncertain_row_count"]} · branch-uncertain {row["branch_uncertain_row_count"]} · '
            f'YOLO {html.escape(row["yolo"]["result"])} · violations {html.escape(violations)}</p>'
            f'<details><summary>Human review label</summary><div class="labels">{radio}</div></details>'
            f'<a href="artifacts/{sid}/result.json">row/object diagnostics</a>'
            f'<img loading="lazy" src="artifacts/{sid}/summary.png"></section>'
        )
    nav = " ".join(f'<a href="#{html.escape(sid)}">{html.escape(sid)}</a>' for sid in PRIORITY)
    return f'''<!doctype html><html><meta charset="utf-8"><title>{html.escape(experiment_title)}</title>
<style>body{{font:15px system-ui;background:#101722;color:#eef;max-width:1600px;margin:auto;padding:18px}}header{{position:sticky;top:0;background:#101722ee;padding:8px;z-index:2}}section{{background:#192331;margin:24px 0;padding:18px;border-radius:12px}}img{{width:100%;height:auto;margin-top:10px}}a{{color:#75ddff}}small{{color:#8ff}}.labels{{display:flex;gap:14px;flex-wrap:wrap;padding:12px 0}}.warn{{color:#ffd477}}</style>
<header><h1>{html.escape(experiment_title)} — 40 images</h1><p>Validation: {html.escape(validation['status'])}. Cached Mask2Former maps; CPU-only conditional YOLO. This is a blocker-policy probe on the Test1 boundary, not a Test1/2/3 integration result.</p><nav>{nav}</nav></header>
<p class="warn">Human labels: {', '.join(HUMAN_LABELS)}. Choices are stored in this browser only and do not tune thresholds or modify results.</p>
{''.join(cards)}
<script>document.querySelectorAll('input[type=radio]').forEach(x=>{{const k='v3-test1-'+x.name; if(localStorage.getItem(k)===x.value)x.checked=true; x.addEventListener('change',()=>localStorage.setItem(k,x.value));}});</script></html>'''


def run_experiment(
    output: Path,
    *,
    blocker_kwargs: dict | None = None,
    experiment_title: str = "RoadPush V3 Test1 Observed Only",
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing V3 output: {output}")
    staging = ROOT / "tmp" / f"{output.name}_staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    priority_index = {sid: index for index, sid in enumerate(PRIORITY)}
    items = sorted(items, key=lambda row: (
        priority_index.get(row["sample_id"], len(PRIORITY)), row.get("order", 999)
    ))
    model_holder: dict = {}
    results = [
        process(
            item, staging, model_holder, index,
            blocker_kwargs=blocker_kwargs,
            experiment_title=experiment_title,
        )
        for index, item in enumerate(items, 1)
    ]

    blocker_source = (Path(__file__).parent / "blockers.py").read_bytes()
    failures = []
    if len(results) != 40:
        failures.append("sample_count")
    if any(row["boundary_calls"]["v2_horizontal_obstacle_bridge"] for row in results):
        failures.append("v2_bridge_leak")
    if any(row["boundary_calls"]["v2_1_natural_restoration"] for row in results):
        failures.append("natural_restoration_leak")
    if any(row["boundary_calls"]["a2_row_span_completion"] for row in results):
        failures.append("a2_leak")
    if any(row["rule_violations"] for row in results):
        failures.append("rule_violations_present")
    validation = {
        "status": "automatic_checks_passed_visual_review_pending" if not failures else "automatic_checks_failed",
        "expected_samples": 40,
        "processed_samples": len(results),
        "priority_order": PRIORITY,
        "cached_mapillary_maps_reused": all(not row["new_mapillary_inference"] for row in results),
        "cpu_only": True,
        "conditional_yolo_samples": [row["sample_id"] for row in results if row["new_yolo_inference"]],
        "yolo_split_confirmed_samples": [row["sample_id"] for row in results if row["yolo"]["result"] == "split_confirmed"],
        "total_final_blocker_pixels": sum(row["final_blocker_pixels"] for row in results),
        "total_transient_occluded_rows": sum(row["transient_occluded_row_count"] for row in results),
        "total_boundary_uncertain_rows": sum(row["boundary_uncertain_row_count"] for row in results),
        "total_branch_uncertain_rows": sum(row["branch_uncertain_row_count"] for row in results),
        "samples_with_rule_violations": [row["sample_id"] for row in results if row["rule_violations"]],
        "global_cycle_closure_used": False,
        "natural_restoration_used": False,
        "scalar_width_summary_present": False,
        "shared_blocker_builder_sha256": hashlib.sha256(blocker_source).hexdigest(),
        "failures": failures,
    }
    (staging / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (staging / "completion_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "display_order", "original_order", "sample_id", "split",
        "recognized_relevant_object_count", "accepted_blocker_object_count", "final_blocker_pixels",
        "boundary_rows_changed_relative_to_test1", "free_space_rows_changed_relative_to_test1",
        "transient_occluded_row_count", "boundary_uncertain_row_count", "branch_uncertain_row_count",
        "new_yolo_inference",
    ]
    with (staging / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in results)
    (staging / "review.html").write_text(
        _review_page(results, validation, experiment_title), encoding="utf-8"
    )
    (staging / "REPORT.md").write_text(
        f"# {experiment_title} — 40 images\n\n"
        f"- Automatic status: {validation['status']}\n"
        "- Boundary: V2 semantic repair + structure_copy + observed selected component\n"
        "- Disabled: V2 bridge, V2.1 natural restoration, A1 restoration, A2 completion\n"
        "- Shared V3 blocker builder: enabled; global cycle closure disabled\n"
        "- Width output: row geometry and reliability only; no scalar summary\n"
        "- Human visual review: pending\n",
        encoding="utf-8",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(output))
    return validation

