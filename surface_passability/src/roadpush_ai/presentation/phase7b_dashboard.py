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
from roadpush_ai.phase7b.handoff import Phase7AAdapter
from roadpush_ai.phase7b.occupancy_builder import ExistingOccupancyEvidenceIndex
from roadpush_ai.phase7b.pipeline import (
    run_phase7_b_sample,
    validate_phase7_b_record,
)


REPOSITORY = get_settings().project_root
FROZEN_V2 = (
    REPOSITORY
    / "outputs"
    / "surface_generalization"
    / "phase7_b_full160_v2_provisional_final"
)
FROZEN_RESULTS = FROZEN_V2 / "results.json"
FROZEN_AMBIGUOUS_MANIFEST = FROZEN_V2 / "ambiguous_review_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _display_ratio(value: float | None) -> str:
    return "계산 불가" if value is None else f"{value * 100:.1f}%"


def render_full160_dashboard(output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite full dashboard: {output}")
    panel_dir = output / "panels"
    panel_dir.mkdir(parents=True)

    frozen_hash_before = _sha256(FROZEN_RESULTS)
    frozen_records = json.loads(FROZEN_RESULTS.read_text(encoding="utf-8"))
    if len(frozen_records) != 160:
        raise AssertionError(f"expected 160 frozen records, got {len(frozen_records)}")
    frozen_by_id = {record["sample_id"]: record for record in frozen_records}
    if len(frozen_by_id) != 160:
        raise AssertionError("frozen results contain duplicate sample IDs")

    ambiguous_reference = json.loads(
        FROZEN_AMBIGUOUS_MANIFEST.read_text(encoding="utf-8")
    )
    ambiguous_hashes = {
        item["sample_id"]: item["panel_sha256"] for item in ambiguous_reference
    }
    if len(ambiguous_hashes) != 16:
        raise AssertionError("expected 16 ambiguous panel references")

    adapter = Phase7AAdapter()
    evidence = ExistingOccupancyEvidenceIndex()
    sample_ids = adapter.sample_ids
    if list(sample_ids) != [record["sample_id"] for record in frozen_records]:
        raise AssertionError("adapter order differs from frozen canonical order")

    compute_start = perf_counter()
    bundles = []
    for sample_id in sample_ids:
        bundle = run_phase7_b_sample(
            sample_id, adapter=adapter, evidence_index=evidence
        )
        validate_phase7_b_record(bundle.record)
        expected = frozen_by_id[sample_id]
        if bundle.record != expected:
            raise AssertionError(f"{sample_id}: recomputed record differs from frozen v2")
        if _normalized_bytes(bundle.record) != _normalized_bytes(expected):
            raise AssertionError(f"{sample_id}: normalized record bytes differ")
        bundles.append(bundle)
    compute_seconds = perf_counter() - compute_start

    recomputed_records = [bundle.record for bundle in bundles]
    results_path = output / "results.json"
    results_path.write_text(
        json.dumps(recomputed_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if _normalized_bytes(recomputed_records) != _normalized_bytes(frozen_records):
        raise AssertionError("full recomputed results differ from frozen v2")

    render_start = perf_counter()
    manifest: list[dict[str, Any]] = []
    cards: list[str] = []
    ambiguous_pixel_matches = 0
    for index, bundle in enumerate(bundles, 1):
        record = bundle.record
        sample_id = record["sample_id"]
        panel_path = panel_dir / f"{sample_id}.png"
        if not cv2.imwrite(
            str(panel_path), _render_panel(bundle, _source_image(bundle))
        ):
            raise OSError(panel_path)
        panel_hash = _sha256(panel_path)
        reference_hash = ambiguous_hashes.get(sample_id)
        if reference_hash is not None:
            if panel_hash != reference_hash:
                raise AssertionError(
                    f"{sample_id}: dashboard renderer changed ambiguous reference"
                )
            ambiguous_pixel_matches += 1

        authority = record["result_authority"]
        manifest.append(
            {
                "index": index,
                "sample_id": sample_id,
                "panel": f"panels/{sample_id}.png",
                "panel_sha256": panel_hash,
                "phase7a_status": record["phase7a_status"],
                "measurement_scope": record["measurement_scope"],
                "result_authority": authority,
                "human_review_required": record["human_review_required"],
                "minimum_passable_ratio": record["minimum_passable_ratio"],
                "raw_minimum_passable_ratio": record[
                    "raw_minimum_passable_ratio"
                ],
                "maximum_obstructed_width_ratio": record[
                    "maximum_obstructed_width_ratio"
                ],
                "obstacle_area_ratio": record["obstacle_area_ratio"],
                "frozen_record_match": True,
                "ambiguous_reference_pixel_match": (
                    True if reference_hash is not None else None
                ),
            }
        )
        cards.append(
            f'<section class="card" data-authority="{html.escape(authority)}" '
            f'data-scope="{html.escape(record["measurement_scope"])}">'
            f'<h2>{index}. {html.escape(sample_id)}</h2>'
            f'<p>{html.escape(authority)} · 범위 {html.escape(record["measurement_scope"])}'
            f' · 상대 통행 가능 폭 {_display_ratio(record["minimum_passable_ratio"])}</p>'
            f'<img loading="lazy" src="panels/{html.escape(sample_id)}.png" '
            f'alt="{html.escape(sample_id)} Phase 7-B final dashboard"></section>'
        )
    render_seconds = perf_counter() - render_start

    manifest_path = output / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    authority_counts = Counter(
        record["result_authority"] for record in recomputed_records
    )
    scope_counts = Counter(record["measurement_scope"] for record in recomputed_records)
    review_path = output / "review.html"
    review_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Phase 7-B final full-160 dashboard</title><style>"
        "body{margin:0;background:#102238;color:#edf4ff;font-family:'Malgun Gothic',sans-serif}"
        "header{padding:18px 22px;background:#172d46;position:sticky;top:0;z-index:2}"
        ".controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}"
        "button{padding:7px 12px;border:1px solid #58799a;border-radius:6px;background:#203b58;color:#fff;cursor:pointer}"
        ".card{margin:18px;background:#182e46;border:1px solid #3c5d7d;border-radius:10px;overflow:hidden}"
        "h1,h2,p{margin:10px 16px}.note{color:#ffcf70}img{display:block;width:100%;height:auto}"
        "</style></head><body><header><h1>Phase 7-B 최종 160장 대시보드</h1>"
        "<p>동결된 V2 결과와 160건 전체 재계산이 일치한 최종 시각화입니다. "
        "provisional_review 16건은 검토용이며 자동 권위 집계에서 제외합니다.</p>"
        f"<p class='note'>전체 160 · 권한 {html.escape(str(dict(authority_counts)))} "
        f"· 범위 {html.escape(str(dict(scope_counts)))}</p>"
        "<div class='controls'><span>표시:</span>"
        "<button onclick=\"filterCards('all')\">전체</button>"
        "<button onclick=\"filterCards('authoritative')\">권위 결과</button>"
        "<button onclick=\"filterCards('provisional_review')\">검토용 16건</button>"
        "<button onclick=\"filterCards('none')\">계산 불가</button></div></header>"
        + "".join(cards)
        + "<script>function filterCards(v){document.querySelectorAll('.card').forEach(" 
        "e=>e.style.display=(v==='all'||e.dataset.authority===v)?'block':'none');}</script>"
        "</body></html>",
        encoding="utf-8",
    )

    frozen_hash_after = _sha256(FROZEN_RESULTS)
    validation = {
        "schema_version": "phase7_b_full160_dashboard_validation_v1",
        "record_count": len(recomputed_records),
        "panel_count": len(manifest),
        "all_recomputed_records_equal_frozen_v2": True,
        "all_recomputed_normalized_bytes_equal_frozen_v2": True,
        "frozen_results_sha256_before": frozen_hash_before,
        "frozen_results_sha256_after": frozen_hash_after,
        "frozen_results_immutable": frozen_hash_before == frozen_hash_after,
        "recomputed_results_sha256": _sha256(results_path),
        "review_manifest_sha256": _sha256(manifest_path),
        "review_html_sha256": _sha256(review_path),
        "ambiguous_reference_count": len(ambiguous_hashes),
        "ambiguous_reference_pixel_matches": ambiguous_pixel_matches,
        "authority_counts": dict(authority_counts),
        "scope_counts": dict(scope_counts),
        "compute_seconds": compute_seconds,
        "render_seconds": render_seconds,
    }
    if not validation["frozen_results_immutable"]:
        raise AssertionError("frozen v2 results changed during dashboard build")
    if ambiguous_pixel_matches != 16:
        raise AssertionError("not all ambiguous reference panels matched")
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return review_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render_full160_dashboard(args.output).resolve())
