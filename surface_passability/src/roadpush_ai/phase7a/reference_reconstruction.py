from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import cv2
import numpy as np

from roadpush_ai.phase7a.extension_s1 import (
    S1A_POLICY_ID,
    freeze_v6_observation,
    generate_s1a_proposal,
)
from roadpush_ai.phase7a.reference_catalog import OUTPUT_ROOT
from roadpush_ai.phase7a.retry_v6 import (
    POLICY_ID as V6_POLICY_ID,
    fit_trapezoid_boundary_retry_v6,
)


REFERENCE = OUTPUT_ROOT / "phase7_v5_probe_v_grounded_vegetation_160_v1"
DEFAULT_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1a_targeted_v1"
TARGETS = (
    "kr_independent_01",
    "alley_044",
    "alley_014",
    "alley_045",
    "alley_081",
    "IMG_4282",
    "alley_005",
    "alley_100",
    "IMG_4431",  # visually complete, 94% component-span accepted control
    "alley_002",
    "alley_018",
    "kr_independent_04",
    "IMG_4364",
    "alley_066",
    "alley_008",
    "alley_043",
    "MP_SEL_PN000644",
    "kr_motorcycle_01",
    "kr_cone_02",
)


def _read_mask(path: str | Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value > 0


def _read_semantic(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if value.ndim != 2:
        raise ValueError(f"Expected a single-channel semantic map: {path}")
    if value.shape != shape:
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return value.astype(np.uint8)


def _disposition(result: dict[str, Any]) -> str:
    if result.get("selected_surface") is None:
        return "uncertain"
    if result["envelope_fit"]["status"] != "accepted":
        return "rejected"
    if result["blocker_reuse"]["warnings"]:
        return "boundary_valid_but_blocker_suspect"
    return "accepted"


def _v6_targeted(reference: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    saved = reference["saved_phase6_artifacts"]
    selected = reference.get("selected_surface")
    if selected is None:
        result, masks = fit_trapezoid_boundary_retry_v6(
            None, reference["selector_state"], None, None
        )
        shape = cv2.imread(reference["source_image"], cv2.IMREAD_COLOR).shape[:2]
        structural = np.zeros(shape, dtype=bool)
        blocker = np.zeros(shape, dtype=bool)
    else:
        structural = _read_mask(saved["phase6_structural_path"])
        blocker = _read_mask(saved["phase6_blocker_path"])
        semantic = _read_semantic(saved["cached_semantic_map_path"], structural.shape)
        obstacle_path = Path(saved["retry_artifact_dir"]) / "obstacle_silhouette.png"
        obstacle = _read_mask(obstacle_path)
        raw_path = saved.get("phase6_raw_structural_path")
        raw = _read_mask(raw_path) if raw_path else None
        result, masks = fit_trapezoid_boundary_retry_v6(
            selected,
            reference["selector_state"],
            structural,
            blocker,
            semantic_labels=semantic,
            obstacle_silhouette_mask=obstacle,
            raw_structural_mask=raw,
        )
    result["final_disposition"] = _disposition(result)
    return result, masks, structural, blocker


def _v6_equivalence_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    ignored = {"policy_id", "v6_contract"}
    mismatches = []
    for key, value in actual.items():
        if key in ignored:
            continue
        normalized = json.loads(json.dumps(value, ensure_ascii=False))
        if expected.get(key) != normalized:
            mismatches.append(key)
    return mismatches


def _load_image(path: str, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return image


def _blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.58) -> np.ndarray:
    out = image.copy()
    out[mask] = (out[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def _title(panel: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((44, panel.shape[1], 3), (22, 31, 45), dtype=np.uint8)
    cv2.putText(bar, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 245, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _render_composite(
    reference: dict[str, Any],
    v6: dict[str, Any],
    structural: np.ndarray,
    blocker: np.ndarray,
    diagnostic: dict[str, Any],
    masks: dict[str, np.ndarray],
    destination: Path,
) -> None:
    image = _load_image(reference["source_image"], structural.shape)
    p1 = _blend(image, structural, (55, 210, 95), 0.48)
    p1 = _blend(p1, blocker, (35, 45, 240), 0.72)

    p2 = image.copy()
    support = v6["envelope_fit"].get("support_y_range")
    for evidence in v6.get("row_evidence", []):
        y = int(evidence["y"])
        inside = bool(support and support[0] <= y <= support[1])
        color = (65, 235, 105) if inside and evidence.get("clean_bilateral") else (0, 175, 255)
        if evidence.get("left") is not None:
            cv2.circle(p2, (int(evidence["left"]), y), 1, color, -1)
        if evidence.get("right") is not None:
            cv2.circle(p2, (int(evidence["right"]), y), 1, color, -1)
    if support:
        for y in range(int(support[0]), int(support[1]) + 1):
            left = int(round(v6["envelope_fit"]["left_fit"]["slope"] * y + v6["envelope_fit"]["left_fit"]["intercept"]))
            right = int(round(v6["envelope_fit"]["right_fit"]["slope"] * y + v6["envelope_fit"]["right_fit"]["intercept"]))
            if 0 <= left < p2.shape[1]:
                p2[y, left] = (255, 80, 40)
            if 0 <= right < p2.shape[1]:
                p2[y, right] = (255, 80, 40)

    p3 = _blend(image, masks["proposal"], (215, 90, 230), 0.58)
    p4 = _blend(image, masks["observed_seed"], (65, 220, 90), 0.62)
    p4 = _blend(p4, masks["extension_only"], (30, 180, 255), 0.72)
    p5 = np.zeros_like(image)
    p5[structural] = (70, 95, 70)
    p5[masks["proposal"]] = (120, 45, 150)
    p5[masks["structural_agreement"]] = (60, 235, 120)

    p6 = image.copy()
    for key, color in (("far_upper", (255, 205, 30)), ("near_lower", (255, 110, 40))):
        direction = diagnostic.get(key)
        if not direction:
            continue
        terminal = int(direction["terminal_y"])
        h2 = int(direction["h2"]["y"])
        cv2.line(p6, (0, terminal), (p6.shape[1] - 1, terminal), color, 2)
        cv2.line(p6, (0, h2), (p6.shape[1] - 1, h2), color, 3)
        if direction.get("h1"):
            h1 = int(direction["h1"]["y"])
            cv2.line(p6, (0, h1), (p6.shape[1] - 1, h1), (210, 210, 210), 1)

    titled = [
        _title(p1, "1 Original + Phase 6 structural/blocker"),
        _title(p2, "2 V6 bands/support + frozen pre-gate fit"),
        _title(p3, "3 S1-A full bounded straight proposal"),
        _title(p4, "4 Observed seed (green) / extension-only (orange)"),
        _title(p5, "5 Structural agreement (green) / disagreement (purple)"),
        _title(p6, "6 H1/H2 and independent far/near stops"),
    ]
    composite = np.vstack([np.hstack(titled[:3]), np.hstack(titled[3:])])
    if not cv2.imwrite(str(destination), composite):
        raise OSError(destination)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def _direction_sentence(name: str, value: dict[str, Any] | None) -> str:
    if value is None:
        return f"{name}: constructible하지 않음."
    horizon = "H2가 H1보다 먼저 도달(short_horizon_only)" if value["short_horizon_only"] else "H1과 H2 모두 기록"
    h2 = value["h2"]
    return (
        f"{name}: {value['extension_row_count']}행 생성, {value['stop_reason']}에서 정지; {horizon}. "
        f"H2 Δleft={h2['delta_x_left']:.1f}, Δright={h2['delta_x_right']:.1f}, "
        f"Δcenter={h2['centerline_delta_x']:.1f}, width ratio={_fmt(h2['width_ratio'])}."
    )


def _judgment_sentence(item: dict[str, Any]) -> str:
    diag = item["s1a"]
    if diag["construction_status"] == "not_constructible":
        return (
            f"V6 기준은 {diag['v6_straight_disposition']}이며 유효한 pre-gate support/fit이 없어 S1-A는 선을 만들지 않았다. "
            "새 anchor나 synthetic line은 생성하지 않았고, S1 선택 및 S2 eligibility는 아직 판단하지 않았다."
        )
    structural = diag["structural_agreement"]
    outside = diag["outside_seed_observation_consistency"]
    return (
        f"V6 기준은 {diag['v6_straight_disposition']}이다. 동결 fit으로 {diag['construction_status']}했으며 "
        f"proposal structural fraction={_fmt(structural['proposal_structural_fraction'])}, "
        f"row p10={_fmt(structural['row_fill_p10'])}, outside-seed clean rows={outside['clean_row_count']}, "
        f"diagnostic contradictory rows={outside.get('contradictory_row_count', 0)}이다. "
        "이 값은 S1-B/C 정책을 검토하기 위한 신호일 뿐 pass/fail이나 모델 선택을 확정하지 않는다."
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, targets: tuple[str, ...] = TARGETS, *, full_160: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite S1-A output: {output}")
    composites = output / "composites"
    mask_root = output / "masks"
    composites.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    references = json.loads((REFERENCE / "results.json").read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in references}
    missing = [sample_id for sample_id in targets if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing targeted samples: {missing}")

    measurement_started = time.perf_counter()
    items: list[dict[str, Any]] = []
    render_payloads: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]] = {}
    for sample_id in targets:
        reference = by_id[sample_id]
        v6, _, structural, blocker = _v6_targeted(reference)
        mismatches = _v6_equivalence_mismatches(v6, reference)
        package = freeze_v6_observation(sample_id, v6, structural.shape)
        diagnostic, masks = generate_s1a_proposal(package, structural, blocker)
        render_payloads[sample_id] = (v6, structural, blocker, diagnostic, masks)
        sample_mask_dir = mask_root / sample_id
        sample_mask_dir.mkdir()
        mask_files: dict[str, dict[str, str]] = {}
        for name, mask in masks.items():
            path = sample_mask_dir / f"{name}.png"
            if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                raise OSError(path)
            mask_files[name] = {
                "path": str(path.resolve()),
                "file_sha256": _file_sha256(path),
                "pixel_sha256": diagnostic.get("mask_sha256", {}).get(name, ""),
            }
        items.append(
            {
                "sample_id": sample_id,
                "source_image": reference["source_image"],
                "selected_surface": reference.get("selected_surface"),
                "v6_policy_id": V6_POLICY_ID,
                "v6_reference_equivalent": not mismatches,
                "v6_reference_mismatch_keys": mismatches,
                "v6_blocker_file_sha256": (reference.get("saved_phase6_artifacts") or {}).get(
                    "phase6_blocker_file_sha256"
                ),
                "s1a": diagnostic,
                "mask_files": mask_files,
            }
        )
    measurement_seconds = time.perf_counter() - measurement_started

    render_started = time.perf_counter()
    cards: list[str] = []
    for item in items:
        sample_id = item["sample_id"]
        reference = by_id[sample_id]
        v6, structural, blocker, diagnostic, masks = render_payloads[sample_id]
        _render_composite(
            reference,
            v6,
            structural,
            blocker,
            diagnostic,
            masks,
            composites / f"{sample_id}.jpg",
        )
        structural_metrics = diagnostic.get("structural_agreement", {})
        observation = diagnostic.get("outside_seed_observation_consistency", {})
        cards.append(
            f'''<article id="{html.escape(sample_id)}"><h2>{html.escape(sample_id)} <small>{html.escape(diagnostic["construction_status"])}</small></h2>
<img loading="lazy" src="composites/{html.escape(sample_id)}.jpg" alt="{html.escape(sample_id)} six diagnostic panels">
<div class="facts"><b>V6 baseline</b> {html.escape(diagnostic["v6_straight_disposition"])} · <b>surface</b> {_fmt(item["selected_surface"])} · <b>support</b> {_fmt(diagnostic.get("support_y_range"))} · <b>bands</b> {len(diagnostic.get("selected_band_centers", []))} · <b>V6 exact reference</b> {str(item["v6_reference_equivalent"]).lower()}</div>
<div class="grid"><section><h3>Far / upper</h3><p>{html.escape(_direction_sentence("far", diagnostic.get("far_upper")))}</p></section><section><h3>Near / lower</h3><p>{html.escape(_direction_sentence("near", diagnostic.get("near_lower")))}</p></section><section><h3>Structural persistence</h3><p>fraction {_fmt(structural_metrics.get("proposal_structural_fraction"))} · row median {_fmt(structural_metrics.get("row_fill_median"))} · p10 {_fmt(structural_metrics.get("row_fill_p10"))} · zero-run {_fmt(structural_metrics.get("longest_zero_overlap_run"))}</p></section><section><h3>Outside-seed observations</h3><p>clean rows {_fmt(observation.get("clean_row_count"))} · median residual {_fmt(observation.get("median_maximum_residual_px"))} · p90 {_fmt(observation.get("p90_maximum_residual_px"))} · diagnostic contradictions {_fmt(observation.get("contradictory_row_count"))}</p></section></div>
<section class="judgment"><h3>7 사진별 판단 및 변경 요약</h3><p>{html.escape(_judgment_sentence(item))}</p><p>V6 결과·마스크 변경 없음. 새 band/refit/inference/context/geometry/selector 호출: 0/0/0/0/0/0.</p></section></article>'''
        )
    render_seconds = time.perf_counter() - render_started

    counts = Counter(item["s1a"]["construction_status"] for item in items)
    mismatch_count = sum(not item["v6_reference_equivalent"] for item in items)
    summary = {
        "status": (
            "FULL-160 DIAGNOSTIC ONLY — NOT A PROMOTION RUN"
            if full_160
            else "TARGETED DIAGNOSTIC ONLY — USER REVIEW REQUIRED"
        ),
        "baseline": V6_POLICY_ID,
        "track": "Extension Track S1",
        "package": "S1-A proposal-generation ablation",
        "policy_id": S1A_POLICY_ID,
        "sample_count": len(items),
        "construction_status_counts": dict(counts),
        "threshold_status": "provisional_not_frozen",
        "selection_or_promotion_performed": False,
        "full_160_executed": full_160,
        "measurement_runtime_seconds": measurement_seconds,
        "render_runtime_seconds": render_seconds,
        "total_runtime_seconds": measurement_seconds + render_seconds,
        "semantic_inference_count": 0,
        "context_build_count": 0,
        "geometry_extraction_count": 0,
        "selector_call_count": 0,
        "v6_reference_mismatch_count": mismatch_count,
    }
    nav = " ".join(f'<a href="#{html.escape(value)}">{html.escape(value)}</a>' for value in targets)
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Extension Track S1-A {'full-160 diagnostic' if full_160 else 'targeted review'}</title><style>
body{{font:15px/1.55 system-ui;background:#0d1724;color:#eef;max-width:1900px;margin:auto;padding:18px}}header{{background:#17283c;padding:18px;position:sticky;top:0;z-index:2;max-height:42vh;overflow:auto}}a{{color:#7bddff;margin-right:9px}}article{{background:#18283d;margin:20px 0;padding:16px;border-radius:12px}}h2{{margin:0 0 8px}}small{{color:#7ff0bd}}img{{width:100%;height:auto;border-radius:7px}}.facts,.judgment{{padding:11px;background:#102033;margin-top:9px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}}section{{background:#111f30;padding:10px}}.judgment{{border-left:5px solid #7bddff}}.tools{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}input,select,button{{font:inherit;padding:7px 10px;background:#0d1724;color:#eef;border:1px solid #567;border-radius:5px}}nav{{font-size:13px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Phase 7 Extension Track S1-A — {'full-160 diagnostic' if full_160 else 'targeted straight proposal generation'}</h1><p>baseline={V6_POLICY_ID} · package=S1-A · samples={len(items)} · counts={html.escape(str(dict(counts)))} · provisional=true · full-160={str(full_160).lower()}</p><p>measurement={measurement_seconds:.3f}s · render={render_seconds:.3f}s · total={measurement_seconds + render_seconds:.3f}s · inference/context/geometry/selector=0/0/0/0 · V6 mismatches={mismatch_count}</p><p>진단 전용입니다. 생성 설정, pass/fail, S2 eligibility는 동결되지 않았고 V6 결과는 변경되지 않습니다.</p><div class="tools"><input id="search" placeholder="sample ID 검색"><select id="status"><option value="">모든 construction</option><option>proposal_generated</option><option>no_op</option><option>not_constructible</option></select><select id="v6"><option value="">모든 V6 disposition</option><option>accepted</option><option>rejected</option><option>uncertain</option><option>boundary_valid_but_blocker_suspect</option></select><button id="clear">필터 초기화</button><b id="visible"></b></div><nav>{nav}</nav></header>{''.join(cards)}<script>
const articles=[...document.querySelectorAll('article')],q=document.querySelector('#search'),s=document.querySelector('#status'),v=document.querySelector('#v6'),visible=document.querySelector('#visible');function apply(){{let n=0;for(const a of articles){{const text=a.innerText.toLowerCase(),ok=(!q.value||a.id.toLowerCase().includes(q.value.toLowerCase()))&&(!s.value||text.includes(s.value))&&(!v.value||text.includes('v6 baseline '+v.value));a.style.display=ok?'':'none';if(ok)n++;}}visible.textContent=`표시 ${{n}}/${{articles.length}}`;}}q.oninput=s.onchange=v.onchange=apply;document.querySelector('#clear').onclick=()=>{{q.value='';s.value='';v.value='';apply();}};apply();
</script></body></html>'''
    (output / "results.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review = output / "review.html"
    review.write_text(page, encoding="utf-8")
    return review


def bundle(output: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite S1-A bundle: {zip_path}")
    included = {"review.html", "results.json", "summary.json", "composites"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.txt",
            "Phase 7 Extension Track S1-A full-160 diagnostic bundle\n\n"
            "1. Extract the ZIP while preserving folders.\n"
            "2. Open review.html in a browser.\n"
            "3. Use the search and disposition/construction filters at the top.\n\n"
            "This is a diagnostic-only S1-A artifact. It is not an S1-D promotion run, "
            "does not freeze thresholds, and does not modify Retry V6.\n",
        )
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            if relative.parts[0] not in included:
                continue
            archive.write(path, relative.as_posix())
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-160", action="store_true")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.all_160:
        references = json.loads((REFERENCE / "results.json").read_text(encoding="utf-8"))
        targets = tuple(row["sample_id"] for row in references)
        if len(targets) != 160:
            raise RuntimeError(f"Expected 160 reference rows, found {len(targets)}")
    else:
        targets = TARGETS
    review = run(args.output, targets, full_160=args.all_160)
    print(review.resolve())
    if args.bundle:
        print(bundle(args.output, args.bundle).resolve())


if __name__ == "__main__":
    main()
