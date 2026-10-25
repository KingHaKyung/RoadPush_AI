from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from roadpush_ai.validation.surface_evaluator import AUDIT, PRIORITY
from roadpush_ai.occupancy.obstacle_geometry import (
    extract_obstacle_geometry,
    prepare_obstacle_geometry,
)
from roadpush_ai.road.enhanced_context import (
    ENHANCED_ROAD_POLICY_ID,
    build_enhanced_road_context,
    write_enhanced_road_diagnostics,
)
from roadpush_ai.road.pipeline import process_road_from_prepared
from roadpush_ai.inference.semantic_inputs import load_or_infer_inputs
from roadpush_ai.surface.context import build_surface_pipeline_context
from roadpush_ai.surface.sidewalk_pipeline import process_sidewalk_from_prepared
from roadpush_ai.phase6.selector import (
    SurfaceSelectorPolicy,
    select_surface_for_input_profile,
)


ROOT = get_settings().project_root
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "surface_generalization" / "phase6_surface_pipeline_v1"
)
ROAD_REVIEW_LABELS = (
    "road-good",
    "road-overreach",
    "road-underreach",
    "forbidden-leak",
    "blocker-wrong",
    "out-of-scope-scene",
)


def _observed_segments(mask: np.ndarray) -> list[dict]:
    rows = []
    for y in range(mask.shape[0]):
        xs = np.flatnonzero(mask[y])
        if not len(xs):
            continue
        starts = [int(xs[0])]
        ends = []
        for left, right in zip(xs[:-1], xs[1:]):
            if right > left + 1:
                ends.append(int(left))
                starts.append(int(right))
        ends.append(int(xs[-1]))
        rows.append({"y": y, "segments": [list(pair) for pair in zip(starts, ends)]})
    return rows


def _sidewalk_summary(results: list[dict], expected_count: int | None = None) -> dict:
    if expected_count is None:
        expected_count = len(results)
    summary = {
        "sample_count": len(results),
        "samples_with_accepted_satellites": sum(
            row["accepted_satellite_count"] > 0 for row in results
        ),
        "total_accepted_satellites": sum(
            row["accepted_satellite_count"] for row in results
        ),
        "total_accepted_satellite_pixels": sum(
            row["accepted_satellite_pixels"] for row in results
        ),
        "samples_changed_vs_current_test3": sum(
            row["boundary_removed_vs_current_test3_pixels"] > 0
            or row["boundary_added_vs_current_test3_pixels"] > 0
            for row in results
        ),
        "total_boundary_removed_vs_current_test3_pixels": sum(
            row["boundary_removed_vs_current_test3_pixels"] for row in results
        ),
        "total_boundary_added_vs_current_test3_pixels": sum(
            row["boundary_added_vs_current_test3_pixels"] for row in results
        ),
        "samples_with_side_pruning": sum(
            row["side_removed_pixels"] > 0 for row in results
        ),
        "total_side_removed_pixels": sum(row["side_removed_pixels"] for row in results),
        "yolo_inference_samples": [
            row["sample_id"] for row in results if row["yolo"]["inference_run"]
        ],
        "automatic_failures": [
            f"rule_violation:{row['sample_id']}"
            for row in results
            if row["rule_violations"]
        ],
        "cached_mask2former_reused": all(
            not row["new_mask2former_inference"] for row in results
        ),
        "cpu_only_yolo": os.environ.get("ROADPUSH_YOLO_DEVICE", "").lower()
        == "cpu"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    summary["status"] = (
        "automatic_checks_passed_visual_review_pending"
        if len(results) == expected_count and not summary["automatic_failures"]
        else "automatic_checks_failed"
    )
    return summary


def _review_page(results: list[dict], status: str) -> str:
    nav = " ".join(
        f'<a href="#{html.escape(sid)}">{html.escape(sid)}</a>' for sid in PRIORITY
    )
    cards = []
    for row in results:
        sidewalk = row["sidewalk"]
        road = row["road"]
        selection = row["surface_selection"]
        sid = html.escape(row["sample_id"])
        labels = "".join(
            f'<label><input type="radio" name="review-{sid}" '
            f'value="{html.escape(label)}"> {html.escape(label)}</label>'
            for label in ROAD_REVIEW_LABELS
        )
        cards.append(
            f'<section id="{sid}"><h2>{row["display_order"]:02d}. {sid} '
            f'<small>{html.escape(str(row.get("split") or ""))}</small></h2>'
            f'<p>Road / Service Lane / Parking = '
            f'{road["source_pixels"]["road"]:,} / '
            f'{road["source_pixels"]["service_lane"]:,} / '
            f'{road["source_pixels"]["parking"]:,}px · repairs '
            f'{road["accepted_repair_pixels"]:,}px · structural '
            f'{road["structural_road_pixels"]:,}px · free '
            f'{road["free_road_pixels"]:,}px · blocker '
            f'{road["road_blocker_pixels"]:,}px</p>'
            f'<p>Enhanced V2: linear drain '
            f'{road["enhanced_policy"]["linear_drain"]["approved_pixels"]:,}px · '
            f'Raw Road ownership removal '
            f'{road["enhanced_policy"]["ownership"]["approved_removal_pixels"]:,}px · '
            f'ownership {html.escape(road["enhanced_policy"]["ownership"]["status"])}</p>'
            f'<p>Sidewalk baseline: satellites '
            f'{sidewalk["accepted_satellite_count"]} · side removed '
            f'{sidewalk["side_removed_pixels"]:,}px · YOLO '
            f'{html.escape(sidewalk["yolo"]["result"])}</p>'
            f'<p>Input profile <b>{html.escape(selection["input_profile"])}</b> · '
            f'selected surface <b>{html.escape(str(selection["selected_surface"]))}</b> · '
            f'state <b>{html.escape(selection["state"])}</b></p>'
            f'<div class="labels">{labels}</div>'
            f'<p><a href="artifacts/{sid}/sidewalk/result.json">sidewalk JSON</a> · '
            f'<a href="artifacts/{sid}/road/road_result.json">road JSON</a></p>'
            f'<div class="pair"><figure><figcaption>Frozen-equivalent Sidewalk</figcaption>'
            f'<img loading="lazy" src="artifacts/{sid}/sidewalk/summary.png"></figure>'
            f'<figure><figcaption>Enhanced Phase 3 Road V2</figcaption>'
            f'<img loading="lazy" src="artifacts/{sid}/road/road_summary.png"></figure></div>'
            f'</section>'
        )
    return f'''<!doctype html><html><meta charset="utf-8"><title>Surface Generalization Phase 6</title>
<style>body{{font:15px system-ui;background:#101722;color:#eef;max-width:1900px;margin:auto;padding:18px}}header{{position:sticky;top:0;background:#101722ee;padding:8px;z-index:2}}section{{background:#192331;margin:24px 0;padding:18px;border-radius:12px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{display:block;width:100%;height:auto;background:white;border-radius:7px}}figure{{margin:0}}figcaption{{font-weight:700;margin:8px}}a{{color:#75ddff}}small{{color:#8ff}}.labels{{display:flex;gap:14px;flex-wrap:wrap;padding:12px 0}}.warn{{color:#ffd477}}@media(max-width:1000px){{.pair{{grid-template-columns:1fr}}}}</style>
<header><h1>Surface Generalization Phase 6 — production selector integration</h1><p>Automatic status: {html.escape(status)}</p><nav>{nav}</nav></header>
<p class="warn">Each image builds the accepted Sidewalk/Road contexts once. The selector consumes those same runtime contexts and does not alter either branch.</p>{''.join(cards)}
<script>document.querySelectorAll('input[type=radio]').forEach(x=>{{const k='surface-phase3-'+x.name;if(localStorage.getItem(k)===x.value)x.checked=true;x.addEventListener('change',()=>localStorage.setItem(k,x.value));}});</script></html>'''


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production Sidewalk/Road pipeline with Phase-6 selection."
    )
    parser.add_argument("--manifest", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--input-profile",
        choices=("image_upload", "cctv"),
        default="image_upload",
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        help="Keep only rows whose selection-manifest tier matches this value.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Image directory for selection manifests that use output_filename.",
    )
    parser.add_argument(
        "--semantic-cache-root",
        type=Path,
        help="Cache directory used when class_map_path is absent.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip Matplotlib review images while retaining masks and JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite surface output: {output}")
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    if args.tier:
        tiers = set(args.tier)
        items = [item for item in items if item.get("tier") in tiers]
    cache_root = args.semantic_cache_root or output / "semantic_cache"
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    normalized_items = []
    for source_item in items:
        item = dict(source_item)
        if "relative_path" not in item:
            if args.image_root is None or "output_filename" not in item:
                raise ValueError(
                    "Manifest rows without relative_path require --image-root and "
                    "output_filename."
                )
            image_root = args.image_root
            if not image_root.is_absolute():
                image_root = ROOT / image_root
            item["relative_path"] = str(
                (image_root / item["output_filename"]).resolve()
            )
        if "class_map_path" not in item:
            item["class_map_path"] = str(
                (cache_root / f"{item['sample_id']}.png").resolve()
            )
        item.setdefault("split", item.get("tier"))
        item.setdefault("order", item.get("selection_order", 999))
        normalized_items.append(item)
    items = normalized_items
    if args.sample_id:
        requested = set(args.sample_id)
        items = [item for item in items if item["sample_id"] in requested]
        missing = requested - {item["sample_id"] for item in items}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
    priority_index = {sid: index for index, sid in enumerate(PRIORITY)}
    items = sorted(
        items,
        key=lambda row: (
            priority_index.get(row["sample_id"], len(PRIORITY)),
            row.get("order", 999),
        ),
    )
    expected_count = len(items)

    model_holder: dict = {}
    selector_policy = SurfaceSelectorPolicy()
    combined_results = []
    diagnostics = []
    for display_order, item in enumerate(items, 1):
        sid = item["sample_id"]
        print(f"[{display_order:02d}/{len(items):02d}] {sid}", flush=True)
        inputs = load_or_infer_inputs(item, model_holder)
        sidewalk_pipeline = build_surface_pipeline_context(inputs.image, inputs.labels)
        enhanced = build_enhanced_road_context(
            inputs.labels, sidewalk_pipeline.surface
        )
        road = enhanced.road
        geometry = extract_obstacle_geometry(inputs.image, inputs.labels)
        geometry = prepare_obstacle_geometry(
            geometry, [sidewalk_pipeline.surface, road.surface], model_holder
        )
        sample_artifact = artifacts / sid
        sample_artifact.mkdir()
        sidewalk_runtime: dict = {}
        road_runtime: dict = {}
        sidewalk_result, sidewalk_diagnostic = process_sidewalk_from_prepared(
            item,
            display_order,
            sample_artifact / "sidewalk",
            inputs,
            sidewalk_pipeline,
            geometry,
            render_review=not args.no_render,
            runtime_capture=sidewalk_runtime,
        )
        road_result, road_diagnostic = process_road_from_prepared(
            item,
            display_order,
            sample_artifact / "road",
            sidewalk_pipeline,
            road,
            geometry,
            render_review=not args.no_render,
            runtime_capture=road_runtime,
        )
        selection = select_surface_for_input_profile(
            args.input_profile,
            sidewalk_pipeline.surface,
            road,
            sidewalk_runtime["structural_mask"],
            road_runtime["structural_mask"],
            inputs.labels,
            selector_policy,
        ).to_dict()
        enhanced_diagnostic = write_enhanced_road_diagnostics(
            sample_artifact / "road", enhanced
        )
        road_result["enhanced_policy"] = enhanced_diagnostic
        (sample_artifact / "road" / "road_result.json").write_text(
            json.dumps(road_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        combined_results.append(
            {
                "sample_id": sid,
                "display_order": display_order,
                "split": item.get("split"),
                "sidewalk": sidewalk_result,
                "road": road_result,
                "surface_selection": selection,
                "bike_lane": {
                    "observed_pixels": int(road.bike_lane.sum()),
                    "observed_segments": _observed_segments(road.bike_lane),
                    "included_in_road_width": False,
                },
            }
        )
        diagnostics.append(
            {
                "sample_id": sid,
                "semantic_inference_run": inputs.inference_run,
                "geometry_extractions": 1,
                "geometry_preparations": 1,
                "ownership_evaluations": sidewalk_diagnostic["ownership_evaluations"]
                + road_diagnostic["ownership_evaluations"],
                "shared_yolo_inference_run": geometry.yolo_inference is not None,
                "sidewalk_yolo_triggered": sidewalk_result["yolo"]["triggered"],
                "road_yolo_triggered": road_result["yolo"]["triggered"],
                "linear_drain_approved_pixels": enhanced_diagnostic["linear_drain"][
                    "approved_pixels"
                ],
                "ownership_removed_pixels": enhanced_diagnostic["ownership"][
                    "approved_removal_pixels"
                ],
                "ownership_status": enhanced_diagnostic["ownership"]["status"],
                "semantic_loads": 1,
                "sidewalk_context_builds": 1,
                "road_context_builds": 1,
                "selector_calls": 1,
            }
        )

    sidewalk_results = [row["sidewalk"] for row in combined_results]
    road_results = [row["road"] for row in combined_results]
    sidewalk_summary = _sidewalk_summary(sidewalk_results, expected_count)
    automatic_failures = []
    for row in road_results:
        sid = row["sample_id"]
        if row["rule_violations"]:
            automatic_failures.append(f"road_rule_violation:{sid}")
        for field in (
            "structural_forbidden_overlap_pixels",
            "free_forbidden_overlap_pixels",
            "structural_bike_lane_overlap_pixels",
            "free_bike_lane_overlap_pixels",
        ):
            if row[field]:
                automatic_failures.append(f"{field}:{sid}")
    automatic_failures.extend(
        f"sidewalk:{failure}" for failure in sidewalk_summary["automatic_failures"]
    )
    summary = {
        "sample_count": len(combined_results),
        "sidewalk_regression_status": sidewalk_summary["status"],
        "road_samples_with_source": sum(
            sum(row["source_pixels"].values()) > 0 for row in road_results
        ),
        "road_samples_with_structural_surface": sum(
            row["structural_road_pixels"] > 0 for row in road_results
        ),
        "road_source_pixels": {
            key: sum(row["source_pixels"][key] for row in road_results)
            for key in ("road", "service_lane", "parking")
        },
        "total_accepted_repair_pixels": sum(
            row["accepted_repair_pixels"] for row in road_results
        ),
        "total_structural_road_pixels": sum(
            row["structural_road_pixels"] for row in road_results
        ),
        "total_free_road_pixels": sum(row["free_road_pixels"] for row in road_results),
        "total_road_blocker_pixels": sum(
            row["road_blocker_pixels"] for row in road_results
        ),
        "automatic_failures": automatic_failures,
        "cpu_only_yolo": sidewalk_summary["cpu_only_yolo"],
        "cached_mask2former_reused": sidewalk_summary["cached_mask2former_reused"],
        "enhanced_road_policy_id": ENHANCED_ROAD_POLICY_ID,
        "total_linear_drain_approved_pixels": sum(
            row["linear_drain_approved_pixels"] for row in diagnostics
        ),
        "total_ownership_removed_pixels": sum(
            row["ownership_removed_pixels"] for row in diagnostics
        ),
        "input_profile": args.input_profile,
        "selected_surface_counts": {
            key: sum(
                row["surface_selection"]["selected_surface"] == key
                for row in combined_results
            )
            for key in ("sidewalk", "road")
        },
        "uncertain_selection_count": sum(
            row["surface_selection"]["selected_surface"] is None
            for row in combined_results
        ),
    }
    summary["status"] = (
        "automatic_checks_passed_user_road_review_required"
        if len(combined_results) == expected_count and not automatic_failures
        else "automatic_checks_failed"
    )
    pipeline_manifest = {
        "runner": "surface_generalization_phase6_production_selector_v1",
        "expected_sample_count": expected_count,
        "frozen_runner_main_called": False,
        "frozen_module_globals_replaced": False,
        "surface_profiles": ["sidewalk", "road"],
        "road_policy_id": ENHANCED_ROAD_POLICY_ID,
        "selector_policy": selector_policy.__dict__,
        "input_profile": args.input_profile,
        "selector_uses_existing_contexts": True,
        "context_rebuild_for_selector": False,
        "original_phase3_runner_unchanged": True,
        "probe_runner_imported": False,
        "semantic_input_policy": "exif_normalized_cached_or_cpu_mask2former_then_cache",
        "semantic_loads": sum(row["semantic_loads"] for row in diagnostics),
        "sidewalk_context_builds": sum(
            row["sidewalk_context_builds"] for row in diagnostics
        ),
        "road_context_builds": sum(
            row["road_context_builds"] for row in diagnostics
        ),
        "selector_calls": sum(row["selector_calls"] for row in diagnostics),
        "geometry_extractions": sum(row["geometry_extractions"] for row in diagnostics),
        "geometry_preparations": sum(row["geometry_preparations"] for row in diagnostics),
        "ownership_evaluations": sum(row["ownership_evaluations"] for row in diagnostics),
        "semantic_inference_samples": [
            row["sample_id"] for row in diagnostics if row["semantic_inference_run"]
        ],
        "shared_yolo_inference_samples": [
            row["sample_id"] for row in diagnostics if row["shared_yolo_inference_run"]
        ],
        "sidewalk_yolo_trigger_samples": [
            row["sample_id"] for row in diagnostics if row["sidewalk_yolo_triggered"]
        ],
        "road_yolo_trigger_samples": [
            row["sample_id"] for row in diagnostics if row["road_yolo_triggered"]
        ],
        "samples": diagnostics,
    }
    for name, payload in (
        ("results.json", combined_results),
        ("summary.json", summary),
        ("sidewalk_results.json", sidewalk_results),
        ("sidewalk_summary.json", sidewalk_summary),
        ("pipeline_manifest.json", pipeline_manifest),
    ):
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not args.no_render:
        (output / "review.html").write_text(
            _review_page(combined_results, summary["status"]), encoding="utf-8"
        )
    print(json.dumps({"output": str(output), **summary, **pipeline_manifest}, indent=2))


if __name__ == "__main__":
    main()


