from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from roadpush_ai.presentation.phase7b_panel import (
    _render_panel,
    _source_image,
)
from roadpush_ai.phase7b.handoff import (
    DEFAULT_PHASE7A_RESULTS,
    Phase7AAdapter,
)
from roadpush_ai.phase7b.occupancy_builder import ExistingOccupancyEvidenceIndex
from roadpush_ai.phase7b.pipeline import (
    Phase7BPipelineBundle,
    run_phase7_b_sample,
    validate_phase7_b_record,
)


REPOSITORY = get_settings().project_root
STEP0_BASELINE = REPOSITORY / "scripts" / "walkway_v3" / "phase7_b_step0_baseline.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_bytes(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalized_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_normalized_bytes(records)).hexdigest()


def _compute_all(
    adapter: Phase7AAdapter,
    evidence: ExistingOccupancyEvidenceIndex,
    *,
    keep_bundles: bool,
) -> tuple[list[dict[str, Any]], dict[str, Phase7BPipelineBundle]]:
    records: list[dict[str, Any]] = []
    bundles: dict[str, Phase7BPipelineBundle] = {}
    for sample_id in adapter.sample_ids:
        bundle = run_phase7_b_sample(
            sample_id,
            adapter=adapter,
            evidence_index=evidence,
        )
        validate_phase7_b_record(bundle.record)
        records.append(bundle.record)
        if keep_bundles:
            bundles[sample_id] = bundle
    return records, bundles


def _review_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if record["passability_state"] == "undetermined":
        reasons.append("undetermined")
    if record["measurement_scope"] == "partial":
        reasons.append("partial_scope")
    if record["passability_state"] == "blocked":
        reasons.append("blocked")
    if record["human_review_required"]:
        reasons.append("human_review_required")
    if record["measurement_confidence"] != "reliable":
        reasons.append("non_reliable_confidence")
    rows = record["diagnostics"]["rows"]
    if rows["excluded_in_scope_row_count"]:
        reasons.append("row_exclusion")
    coverage = rows["measurement_run_coverage"]
    if coverage is not None and coverage < 1.0:
        reasons.append("measurement_run_coverage_below_one")
    meaningful_flags = {
        "reliable_geometric_closure",
        "metric_path_differs_from_leftmost_topology_witness",
        "no_full_run_connected_corridor",
        "measurement_low_confidence",
        "measurement_insufficient_confidence",
    }
    for flag in record["flags"]:
        if flag in meaningful_flags:
            reasons.append(flag)
    return list(dict.fromkeys(reasons))


def _write_flagged_review(
    output: Path,
    records: list[dict[str, Any]],
    bundles: dict[str, Phase7BPipelineBundle],
) -> tuple[Path, list[dict[str, Any]], float]:
    start = perf_counter()
    panel_dir = output / "flagged_panels"
    panel_dir.mkdir()
    manifest: list[dict[str, Any]] = []
    cards: list[str] = []
    for index, record in enumerate(records, 1):
        reasons = _review_reasons(record)
        if not reasons:
            continue
        sample_id = record["sample_id"]
        bundle = bundles[sample_id]
        panel_path = panel_dir / f"{sample_id}.png"
        if not cv2.imwrite(
            str(panel_path), _render_panel(bundle, _source_image(bundle))
        ):
            raise OSError(panel_path)
        panel_hash = _file_hash(panel_path)
        manifest.append(
            {
                "sample_id": sample_id,
                "reasons": reasons,
                "panel": str(panel_path.relative_to(output)).replace("\\", "/"),
                "panel_sha256": panel_hash,
            }
        )
        cards.append(
            f'''<section class="card" id="c{len(manifest)}"><h2>{html.escape(sample_id)}</h2><p class="reason">{html.escape(', '.join(reasons))}</p><p>Phase7-A={html.escape(record['phase7a_status'])} · scope={record['measurement_scope']} · confidence={record['measurement_confidence']} · state={record['passability_state']} · robust={record['minimum_passable_ratio']} · raw={record['raw_minimum_passable_ratio']}</p><img src="flagged_panels/{html.escape(sample_id)}.png" alt="{html.escape(sample_id)} flagged Phase 7-B review"></section>'''
        )
    manifest_path = output / "flagged_review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reason_counts = Counter(reason for item in manifest for reason in item["reasons"])
    review = output / "flagged_review.html"
    review.write_text(
        f'''<!doctype html><html><head><meta charset="utf-8"><title>Phase 7-B full-160 flagged-only review</title><style>body{{margin:0;background:#0e2034;color:#edf4ff;font-family:Arial,sans-serif}}header{{padding:24px;background:#13283f;position:sticky;top:0;z-index:2}}h1{{margin:0 0 8px}}.warn{{color:#ffd479}}.card{{margin:18px;border:1px solid #31506f;border-radius:12px;overflow:hidden;background:#142a42}}.card h2,.card p{{padding:0 16px}}.reason{{color:#ffcf70}}img{{display:block;width:100%;height:auto}}</style></head><body><header><h1>Phase 7-B full-160 — flagged-only review</h1><p class="warn">There is no accepted prior Phase 7-B baseline, so this is not a changed-only review. Only review-relevant records are rendered.</p><p>full numeric records={len(records)} · rendered flagged records={len(manifest)} · reasons={html.escape(str(dict(reason_counts)))}</p></header>{''.join(cards)}</body></html>''',
        encoding="utf-8",
    )
    return review, manifest, perf_counter() - start


def run_full160(output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 7-B full run: {output}")
    output.mkdir(parents=True)
    phase7a_hash_before = _file_hash(DEFAULT_PHASE7A_RESULTS)
    adapter = Phase7AAdapter()
    evidence = ExistingOccupancyEvidenceIndex()
    if len(adapter.sample_ids) != 160:
        raise AssertionError(f"Expected 160 Phase 7-A inputs, got {len(adapter.sample_ids)}")

    primary_start = perf_counter()
    primary, bundles = _compute_all(adapter, evidence, keep_bundles=True)
    primary_seconds = perf_counter() - primary_start
    rerun_start = perf_counter()
    rerun, _ = _compute_all(adapter, evidence, keep_bundles=False)
    rerun_seconds = perf_counter() - rerun_start

    primary_hash = _normalized_hash(primary)
    rerun_hash = _normalized_hash(rerun)
    primary_by_id = {record["sample_id"]: record for record in primary}
    rerun_by_id = {record["sample_id"]: record for record in rerun}
    mismatches = [
        sample_id
        for sample_id in adapter.sample_ids
        if _normalized_bytes([primary_by_id[sample_id]])
        != _normalized_bytes([rerun_by_id[sample_id]])
    ]
    if primary_hash != rerun_hash or mismatches:
        raise AssertionError(f"Phase 7-B deterministic rerun mismatch: {mismatches}")

    results_path = output / "results.json"
    rerun_path = output / "results_rerun.json"
    pretty_primary = json.dumps(primary, ensure_ascii=False, indent=2) + "\n"
    pretty_rerun = json.dumps(rerun, ensure_ascii=False, indent=2) + "\n"
    results_path.write_text(pretty_primary, encoding="utf-8")
    rerun_path.write_text(pretty_rerun, encoding="utf-8")
    if results_path.read_bytes() != rerun_path.read_bytes():
        raise AssertionError("Pretty serialized rerun differs despite normalized equality")

    review_path, review_manifest, render_seconds = _write_flagged_review(
        output, primary, bundles
    )
    phase7a_hash_after = _file_hash(DEFAULT_PHASE7A_RESULTS)
    if phase7a_hash_before != phase7a_hash_after:
        raise AssertionError("Authoritative Phase 7-A results changed")

    status_counts = Counter(record["phase7a_status"] for record in primary)
    scope_counts = Counter(record["measurement_scope"] for record in primary)
    confidence_counts = Counter(record["measurement_confidence"] for record in primary)
    state_counts = Counter(record["passability_state"] for record in primary)
    abstention_counts = Counter(
        reason for record in primary for reason in record["abstention_reasons"]
    )
    all_default_null = all(
        record["surface_width_m"] is None
        and record["absolute_passable_width_m"] is None
        and record["effective_blocked_width_m"] is None
        and record["agent_width_m"] is None
        and record["safety_factor"] is None
        and record["required_safe_width_m"] is None
        and record["required_passable_ratio"] is None
        and record["required_safe_ratio"] is None
        for record in primary
    )
    validation = {
        "schema_version": "phase7_b_full160_validation_v1",
        "record_count": len(primary),
        "all_records_valid": True,
        "all_default_physical_inputs_and_absolute_outputs_null": all_default_null,
        "phase7a_results_sha256_before": phase7a_hash_before,
        "phase7a_results_sha256_after": phase7a_hash_after,
        "phase7a_unchanged": phase7a_hash_before == phase7a_hash_after,
        "normalized_results_sha256": primary_hash,
        "normalized_rerun_sha256": rerun_hash,
        "deterministic": primary_hash == rerun_hash and not mismatches,
        "mismatch_sample_ids": mismatches,
        "results_file_sha256": _file_hash(results_path),
        "rerun_file_sha256": _file_hash(rerun_path),
        "results_files_byte_identical": results_path.read_bytes()
        == rerun_path.read_bytes(),
    }
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "phase7_b_full160_summary_v1",
        "record_count": len(primary),
        "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts),
        "confidence_counts": dict(confidence_counts),
        "state_counts": dict(state_counts),
        "abstention_counts": dict(abstention_counts),
        "flagged_review_count": len(review_manifest),
        "flagged_review_label": "flagged-only (no accepted prior Phase 7-B baseline)",
        "primary_compute_seconds": primary_seconds,
        "deterministic_rerun_compute_seconds": rerun_seconds,
        "flagged_render_seconds": render_seconds,
        "results": str(results_path.resolve()),
        "flagged_review": str(review_path.resolve()),
        "step0_baseline": str(STEP0_BASELINE.resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return review_path


def main() -> None:
    """Run the accepted 160-case Phase 7-B batch from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_full160(arguments.output).resolve())


if __name__ == "__main__":
    main()
