from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import html
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from roadpush_ai.phase7a.extension_s1 import (
    S1AGenerationPolicy,
    S1_TARGETED_POLICY_ID,
    S1_TARGETED_V2_POLICY_ID,
    S1_TARGETED_V3_POLICY_ID,
    S1_TARGETED_V4A_POLICY_ID,
    S1_TARGETED_V4B_POLICY_ID,
    S1_TARGETED_V4C_POLICY_ID,
    S1_V4_PROBE_POLICY_ID,
    evaluate_s1_targeted,
    evaluate_s1_targeted_v2,
    evaluate_s1_targeted_v3,
    evaluate_s1_targeted_v4a,
    evaluate_s1_targeted_v4b,
    evaluate_s1_targeted_v4c,
    freeze_v6_observation,
    grade_s1_v4_probe_result,
    generate_s1a_proposal,
)
from roadpush_ai.phase7a.reference_reconstruction import (
    REFERENCE,
    TARGETS,
    V6_POLICY_ID,
    _load_image,
    _v6_equivalence_mismatches,
    _v6_targeted,
)
from roadpush_ai.phase7a.reference_catalog import OUTPUT_ROOT


DEFAULT_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_targeted_provisional_v1"
V2_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_targeted_provisional_v2"
V3_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_targeted_provisional_v3"
V3_FULL160_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_full160_regression_v1"
V4A_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_targeted_provisional_v4a"
V4A_FULL160_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_full160_regression_v4a"
V4B_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_targeted_provisional_v4b"
V4B_FULL160_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_full160_regression_v4b"
V4C_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_targeted_provisional_v4c"
V4C_FULL160_OUTPUT = OUTPUT_ROOT / "phase7_extension_track_s1_full160_regression_v4c"
BANNER_CLASS_ID = 32

# Explicit occluder allowlist for V4b. Hazards, boundaries, vegetation, and
# ground/surface classes are intentionally absent.
V4B_OCCLUDER_CLASS_IDS = (
    1, 19, 20, 21, 22,
    33, 34, 35, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 54, 55, 56, 57, 58, 59, 60, 61, 62,
)


def _v4b_occluder_mask(
    reference: dict[str, Any],
    shape: tuple[int, int],
    *,
    include_banner: bool = False,
) -> np.ndarray:
    saved = reference.get("saved_phase6_artifacts") or {}
    semantic_path = saved.get("cached_semantic_map_path")
    if not semantic_path:
        return np.zeros(shape, dtype=bool)
    labels = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(semantic_path)
    if labels.ndim != 2:
        raise ValueError(f"Expected single-channel semantic map: {semantic_path}")
    if labels.shape != shape:
        labels = cv2.resize(
            labels.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    class_ids = (
        (*V4B_OCCLUDER_CLASS_IDS, BANNER_CLASS_ID)
        if include_banner
        else V4B_OCCLUDER_CLASS_IDS
    )
    return np.isin(labels, class_ids)


def _blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.copy()
    output[mask] = (
        output[mask] * (1.0 - alpha) + np.asarray(color) * alpha
    ).astype(np.uint8)
    return output


def _title(panel: np.ndarray, text: str, color: tuple[int, int, int] = (240, 245, 255)) -> np.ndarray:
    bar = np.full((44, panel.shape[1], 3), (22, 31, 45), dtype=np.uint8)
    cv2.putText(bar, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _reason_text(reason: str) -> str:
    return {
        "missing_frozen_v6_support_or_fit": "기존 V6 직선 근거가 부족함",
        "preexisting_hard_veto": "기존 V6 hard veto가 있음",
        "straight_direction_evidence": "기존 관측이 직선 방향으로 안정적이지 않음",
        "proposal_structural_fraction": "후보 전체가 구조 영역과 충분히 겹치지 않음",
        "row_fill_p10": "일부 구간에서 구조 영역 지지가 약함",
        "extension_structural_persistence": "연장 구간의 구조 영역이 지속되지 않음",
        "observed_extension_connection": "관측 구간과 연장 구간이 연결되지 않음",
        "contradictory_observation_fraction": "기존 관측 중 직선과 모순되는 비율이 큼",
        "persistent_contradictory_observations": "직선과 모순되는 관측이 연속해서 나타남",
        "far_upper_geometry": "먼쪽 경계의 좌우 순서가 무너짐",
        "near_lower_geometry": "가까운쪽 경계의 좌우 순서가 무너짐",
        "basic_geometry_validity": "좌우 경계의 기본 형상이 유효하지 않음",
        "extension_structural_continuity": "실제 연장 범위의 구조 영역 지속성이 부족함",
        "persistent_in_range_observation_contradiction": "실제 연장 범위에서 지속적인 관측 모순이 확인됨",
        "persistent_in_range_containment_violation": "후보가 관측 통로 밖으로 지속적으로 돌출함",
        "minimum_safe_extension_prefix": "seed와 연결된 안전 연장 구간이 너무 짧음",
        "occlusion_adjusted_extension_structural_continuity": "장애물 가림을 제외해도 연장 구조가 부족함",
        "occlusion_adjusted_persistent_containment_violation": "장애물 가림을 제외해도 후보가 관측 통로 밖으로 지속 돌출함",
    }.get(reason, reason)


def _status_text(decision: dict[str, Any]) -> str:
    if decision["s1_status"] == "selected":
        return "S1 채택 — 연장 직선을 최종 S1 마스크로 사용"
    if decision["s1_status"] == "selected_partial":
        return "S1 부분 채택 — 직선 prefix를 사용하고 이후는 S2로 전달"
    if decision["s1_status"] == "ambiguous_hold":
        return "영역 생성 완료 — 자동 채택하지 않고 사람 검토 대기"
    if decision["s1_status"] == "unchanged":
        return "신규 연장 없음 — 기존 결과를 그대로 유지"
    if decision["s1_status"] == "no_extension_needed":
        return "연장 불필요 — 관측된 직선만 유지"
    return "S1 기각 — S1에서는 아무 선도 선택하지 않음"


def _v6_subclassification(
    disposition: str,
    v6: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[str, str]:
    """Return a stable display key and the exact source evidence."""

    direction = v6.get("probe_b_direction_consistency", {}).get("classification")
    reasons = [str(value) for value in v6.get("envelope_fit", {}).get("rejection_reasons", [])]
    warnings = [str(value) for value in v6.get("blocker_reuse", {}).get("warnings", [])]
    selector_reasons = [str(value) for value in reference.get("selector_reasons", [])]
    if disposition == "accepted":
        return str(direction or "unclassified"), str(direction or "unclassified")
    if disposition == "rejected":
        joined = " | ".join(reasons)
        if "probe_b_coherent_turn_deferred" in reasons:
            return "coherent_turn", joined
        if "probe_b_unstable_bilateral" in reasons:
            return "unstable_bilateral", joined
        if "probe_v_grounded_vegetation_cap" in reasons:
            return "grounded_vegetation_hard_veto", joined
        if "insufficient_anchor_rows" in reasons or "no_bilateral_clean_caps" in reasons:
            return "insufficient_bilateral_anchor", joined
        return "other_rejection", joined or "rejected_without_reason"
    if disposition == "uncertain":
        exact = " | ".join(selector_reasons) or "selector_uncertain"
        return "no_viable_camera_anchor_surface", exact
    if disposition == "boundary_valid_but_blocker_suspect":
        exact = " | ".join(warnings) or "blocker_suspect"
        return "blocker_coverage_suspect", exact
    return "unclassified", disposition


def _render(
    reference: dict[str, Any],
    v6: dict[str, Any],
    v6_masks: dict[str, np.ndarray],
    structural: np.ndarray,
    blocker: np.ndarray,
    diagnostic: dict[str, Any],
    masks: dict[str, np.ndarray],
    decision: dict[str, Any],
    effective: np.ndarray,
    destination: Path,
) -> None:
    image = _load_image(reference["source_image"], structural.shape)
    p1 = _blend(image, structural, (55, 210, 95), 0.42)
    p1 = _blend(p1, blocker, (35, 45, 240), 0.72)

    support = diagnostic.get("support_y_range")
    fit = v6.get("envelope_fit", {})
    left_fit = fit.get("left_fit", {})
    right_fit = fit.get("right_fit", {})
    extension = np.asarray(masks["extension_only"], dtype=bool)
    observed = np.asarray(masks["observed_seed"], dtype=bool)
    h, w = structural.shape

    def label(panel: np.ndarray, value: str, xy: tuple[int, int], color=(245, 245, 245), scale=0.45) -> None:
        cv2.putText(panel, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def fitted_x(side_fit: dict[str, Any], y: int) -> int | None:
        if not side_fit.get("valid"):
            return None
        return int(round(float(side_fit["slope"]) * y + float(side_fit["intercept"])))

    # 2: immutable V6 seed and the exact fitted lines used by S1.
    p2 = _blend(image, observed, (230, 205, 45), 0.62)
    if support:
        lo, hi = map(int, support)
        for side_fit in (left_fit, right_fit):
            x0, x1 = fitted_x(side_fit, lo), fitted_x(side_fit, hi)
            if x0 is not None and x1 is not None:
                cv2.line(p2, (x0, lo), (x1, hi), (20, 230, 255), 3)
        cv2.rectangle(p2, (0, lo), (w - 1, hi), (255, 190, 20), 2)
        label(p2, f"V6 support y={lo}-{hi}", (12, min(h - 12, hi + 22)), (255, 220, 80), 0.52)

    # 3: seed versus new extension, with the literal stop reason at each end.
    p3 = _blend(image, observed, (230, 205, 45), 0.48)
    p3 = _blend(p3, extension, (215, 90, 230), 0.58)
    for key, color in (("far_upper", (255, 205, 30)), ("near_lower", (255, 110, 40))):
        value = diagnostic.get(key)
        if not value:
            continue
        terminal_y, stop_y = int(value["terminal_y"]), int(value["stop_y"])
        cv2.line(p3, (0, terminal_y), (w - 1, terminal_y), color, 2)
        cv2.line(p3, (0, stop_y), (w - 1, stop_y), color, 3)
        text_y = max(16, min(h - 8, stop_y - 5 if key == "near_lower" else stop_y + 17))
        label(p3, f"{key}: {value['stop_reason']} @ y={stop_y}", (10, text_y), color, 0.43)

    # 4: extension-only structure support. Blue is the exact neutralized subset.
    p4 = (image.astype(np.float32) * 0.32).astype(np.uint8)
    supported_extension = extension & structural
    unsupported_extension = extension & ~structural
    structural_neutral = np.asarray(
        masks.get("occlusion_neutral_structural", np.zeros_like(extension)),
        dtype=bool,
    )
    p4[supported_extension] = (55, 220, 90)
    p4[unsupported_extension & ~structural_neutral] = (45, 55, 235)
    p4[structural_neutral] = (235, 135, 35)
    extension_rows = np.flatnonzero(np.any(extension, axis=1))
    v4b_fill_audit = decision.get("v4b_row_fill_audit", {})
    fills = {}
    for y_value in extension_rows:
        y = int(y_value)
        audit = v4b_fill_audit.get(str(y))
        value = audit.get("adjusted_fill") if audit else None
        fills[y] = (
            float(value)
            if value is not None
            else float(supported_extension[y].sum() / extension[y].sum())
        )
    graph_left = max(8, w - 112)
    cv2.line(p4, (graph_left + 50, 0), (graph_left + 50, h - 1), (60, 190, 255), 1)
    for y, fill_value in fills.items():
        cv2.line(p4, (graph_left, y), (graph_left + int(round(fill_value * 100)), y), (245, 245, 245), 1)
    structural_check = next(
        (
            c
            for c in decision["checks"]
            if c["name"]
            in {
                "extension_structural_continuity",
                "occlusion_adjusted_extension_structural_continuity",
            }
        ),
        None,
    )
    p10 = None if structural_check is None else structural_check.get("value")
    p10_text = "n/a" if p10 is None else f"{float(p10):.3f}"
    label(p4, "row fill bar: 0% | 50% | 100%", (max(8, w - 245), 18), (240, 240, 240), 0.38)
    label(p4, f"p10={p10_text}  threshold=0.500", (10, h - 13), (80, 240, 130) if structural_check and structural_check["passed"] else (80, 100, 255), 0.5)

    # 5: signed containment. Blue is only the same-side neutralized overhang.
    p5 = (image.astype(np.float32) * 0.46).astype(np.uint8)
    p5 = _blend(p5, extension, (180, 75, 205), 0.26)
    row_by_y = {int(row["y"]): row for row in v6.get("row_evidence", [])}
    residual_by_y = {
        int(row["y"]): row
        for row in diagnostic.get("outside_seed_observation_consistency", {}).get("residuals", [])
        if extension[int(row["y"])].any()
    }
    containment_neutral = np.asarray(
        masks.get("occlusion_neutral_containment", np.zeros_like(extension)),
        dtype=bool,
    )
    v4b_containment_audit = decision.get("v4b_containment_audit", {})
    contradiction_key = "containment_contradiction" if decision.get("observation_comparison") == "signed_containment" else "diagnostic_contradiction"
    for y_value in extension_rows:
        y = int(y_value)
        evidence, residual = row_by_y.get(y), residual_by_y.get(y)
        if evidence is None or residual is None:
            cv2.line(p5, (w - 12, y), (w - 2, y), (145, 145, 145), 1)
            continue
        ol, or_ = int(evidence["left"]), int(evidence["right"])
        cl, cr = fitted_x(left_fit, y), fitted_x(right_fit, y)
        if cl is None or cr is None:
            continue
        cv2.line(p5, (max(0, ol), y), (min(w - 1, or_), y), (245, 220, 40), 1)
        if cl < ol:
            cv2.line(p5, (max(0, cl), y), (min(w - 1, ol), y), (35, 45, 245), 2)
        if cr > or_:
            cv2.line(p5, (max(0, or_), y), (min(w - 1, cr), y), (35, 45, 245), 2)
        adjusted = v4b_containment_audit.get(str(y), {}).get(
            "adjusted_contradiction"
        )
        contradictory = (
            bool(adjusted)
            if adjusted is not None
            else bool(residual.get(contradiction_key))
        )
        tick_color = (35, 45, 245) if contradictory else (55, 220, 90)
        cv2.line(p5, (w - 12, y), (w - 2, y), tick_color, 2)
    p5[containment_neutral] = (235, 135, 35)
    blocker_outline, _ = cv2.findContours(blocker.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p5, blocker_outline, -1, (20, 150, 255), 2)
    containment_check = next(
        (
            c
            for c in decision["checks"]
            if c["name"]
            in {
                "persistent_in_range_containment_violation",
                "occlusion_adjusted_persistent_containment_violation",
            }
        ),
        None,
    )
    containment_value = (containment_check or {}).get("value", {})
    label(p5, f"clean={containment_value.get('clean_rows', 0)} outside={containment_value.get('contradictory_rows', 0)} run={containment_value.get('longest_consecutive_run', 0)}", (10, h - 13), (245, 245, 245), 0.5)

    # 6: row ledger: structure amount and containment outcome are separate columns.
    p6 = np.full_like(image, (12, 20, 31))
    label(p6, "Y", (12, 20), (180, 205, 225), 0.45)
    label(p6, "STRUCTURAL ROW FILL", (70, 20), (180, 205, 225), 0.45)
    label(p6, "CONTAINMENT", (430, 20), (180, 205, 225), 0.45)
    cv2.line(p6, (200, 26), (200, h - 1), (60, 190, 255), 1)
    for y_value in extension_rows:
        y = int(y_value)
        draw_y = max(28, y)
        fill_value = fills.get(y, 0.0)
        fill_color = (55, 220, 90) if fill_value >= 0.5 else (45, 55, 235)
        cv2.line(p6, (70, draw_y), (70 + int(round(fill_value * 260)), draw_y), fill_color, 1)
        residual = residual_by_y.get(y)
        color = (115, 115, 115) if residual is None else (45, 55, 235) if residual.get(contradiction_key) else (55, 220, 90)
        cv2.line(p6, (430, draw_y), (610, draw_y), color, 1)
    if support:
        cv2.rectangle(p6, (5, int(support[0])), (30, int(support[1])), (255, 190, 20), -1)
    label(p6, "gray = no clean bilateral observation", (390, h - 12), (160, 160, 160), 0.38)

    # 7: actual checks instead of a duplicate proposal image.
    p7 = np.full_like(image, (15, 28, 43))
    label(p7, f"FINAL: {decision['s1_status'].upper()}", (22, 38), (80, 230, 120) if decision["s1_status"] != "failed" else (80, 100, 255), 0.82)
    y_text = 82
    for check in decision["checks"]:
        state = "PASS" if check["passed"] else "FAIL"
        color = (80, 230, 120) if check["passed"] else (80, 100, 255)
        value = check.get("value")
        if isinstance(value, float):
            value_text = f"{value:.3f}"
        elif isinstance(value, dict) and "fraction" in value:
            fraction = value.get("fraction")
            fraction_text = "n/a" if fraction is None else f"{float(fraction):.3f}"
            value_text = f"rows={value.get('clean_rows')} outside={value.get('contradictory_rows')} frac={fraction_text} run={value.get('longest_consecutive_run')}"
        elif isinstance(value, dict):
            value_text = ", ".join(f"{k}={v}" for k, v in value.items())
        else:
            value_text = str(value)
        label(p7, f"[{state}] {check['name']}", (24, y_text), color, 0.48)
        label(p7, value_text[:88], (42, y_text + 20), (190, 205, 220), 0.38)
        y_text += 62

    p8 = _blend(image, effective, (70, 230, 120), 0.62)
    if not effective.any():
        cv2.putText(p8, "S1 SELECTED MASK: NONE", (20, max(42, p8.shape[0] // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 240), 3, cv2.LINE_AA)

    status_color = (
        (80, 210, 255)
        if decision["s1_status"] == "ambiguous_hold"
        else (80, 240, 150)
        if decision["s1_status"] != "failed"
        else (100, 130, 255)
    )
    panels = [
        _title(p1, "1 Original + Phase 6 structural/blocker"),
        _title(p2, f"2 Frozen V6 seed + fitted lines ({v6['final_disposition']})"),
        _title(p3, "3 Extension geometry + literal stop reasons"),
        _title(p4, "4 Extension support (green=supported, red=missing, blue=occluder-neutral)"),
        _title(p5, "5 Signed containment (cyan=observed, red=outside, blue=occluder-neutral)"),
        _title(p6, "6 Row ledger (structure / containment / unavailable)"),
        _title(p7, "7 Validation checklist (no duplicate proposal image)"),
        _title(p8, f"8 Effective S1 result: {decision['s1_status']}", status_color),
    ]
    composite = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:])])
    if not cv2.imwrite(str(destination), composite):
        raise OSError(destination)


def run(
    output: Path = DEFAULT_OUTPUT,
    *,
    version: str = "v1",
    targets: tuple[str, ...] = TARGETS,
    full_160: bool = False,
    changed_only_against: Path | None = None,
    include_banner_occluder: bool = False,
    grade_v4_probe: bool = False,
    comparison_baseline_variant: str | None = None,
    render_images: bool = True,
) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite targeted output: {output}")
    composites = output / "composites"
    baseline_composites = output / "baseline_composites"
    selected_masks = output / "effective_selected_masks"
    output.mkdir(parents=True)
    if render_images:
        composites.mkdir()
    if render_images and version in {"v4b", "v4c"} and (
        changed_only_against is not None or comparison_baseline_variant is not None
    ):
        baseline_composites.mkdir()
    if render_images:
        selected_masks.mkdir()

    references = json.loads((REFERENCE / "results.json").read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in references}
    baseline_by_id: dict[str, dict[str, Any]] = {}
    if changed_only_against is not None:
        baseline_path = (
            changed_only_against / "results.json"
            if changed_only_against.is_dir()
            else changed_only_against
        )
        baseline_items = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_by_id = {row["sample_id"]: row for row in baseline_items}
    items: list[dict[str, Any]] = []
    cards: list[str] = []
    rendered_sample_ids: list[str] = []
    started = time.perf_counter()

    for sample_id in targets:
        reference = by_id[sample_id]
        v6, v6_masks, structural, blocker = _v6_targeted(reference)
        package = freeze_v6_observation(sample_id, v6, structural.shape)
        generation_policy = S1AGenerationPolicy(
            stop_before_side_frame_contact=(version in {"v3", "v4a", "v4b", "v4c"}),
            minimum_resolvable_width_px=(12 if version in {"v4a", "v4b", "v4c"} else None),
            minimum_width_uncertainty_multiplier=2.0,
        )
        diagnostic, masks = generate_s1a_proposal(
            package, structural, blocker, generation_policy
        )
        if version in {"v4a", "v4b", "v4c"}:
            _, v3_generation_masks = generate_s1a_proposal(
                package,
                structural,
                blocker,
                S1AGenerationPolicy(stop_before_side_frame_contact=True),
            )
            newly_added = masks["extension_only"] & ~v3_generation_masks["extension_only"]
            if np.any(newly_added):
                raise AssertionError(
                    f"{sample_id}: {version} must be a subset of the v3 candidate "
                    f"but added {int(np.count_nonzero(newly_added))} pixels"
                )
            diagnostic["v4a_subset_of_v3_candidate"] = True
        if version == "v4b":
            obstacle = _v4b_occluder_mask(
                reference,
                structural.shape,
                include_banner=include_banner_occluder,
            )
            decision, selected = evaluate_s1_targeted_v4b(
                package, diagnostic, masks, obstacle
            )
        elif version == "v4c":
            obstacle = _v4b_occluder_mask(
                reference,
                structural.shape,
                include_banner=include_banner_occluder,
            )
            decision, selected = evaluate_s1_targeted_v4c(
                package, diagnostic, masks, obstacle
            )
        else:
            evaluator = {
                "v1": evaluate_s1_targeted,
                "v2": evaluate_s1_targeted_v2,
                "v3": evaluate_s1_targeted_v3,
                "v4a": evaluate_s1_targeted_v4a,
            }[version]
            decision, selected = evaluator(package, diagnostic, masks)
        if include_banner_occluder and version in {"v4b", "v4c"}:
            decision["policy_id"] = f'{decision["policy_id"]}_banner32'
            decision["banner_occluder_enabled"] = True
            decision["occluder_class_ids"] = [*V4B_OCCLUDER_CLASS_IDS, BANNER_CLASS_ID]
        if grade_v4_probe:
            if version != "v4c":
                raise ValueError("V4 probe grading requires the V4c safe-prefix result")
            decision, selected = grade_s1_v4_probe_result(
                package,
                decision,
                selected,
                masks,
                top_bottom_width_ratio=v6.get("envelope_fit", {}).get(
                    "top_bottom_width_ratio"
                ),
            )
        effective = selected["effective_selected"]
        selected_path = selected_masks / f"{sample_id}.png"
        selected_file_sha256 = None
        if render_images:
            if not cv2.imwrite(str(selected_path), effective.astype(np.uint8) * 255):
                raise OSError(selected_path)
            selected_file_sha256 = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        baseline = baseline_by_id.get(sample_id)
        comparison_flags = None
        if baseline is not None:
            baseline_decision = baseline["s1_decision"]
            comparison_flags = {
                "status": baseline_decision["s1_status"] != decision["s1_status"],
                "effective_mask": baseline_decision["effective_selected_mask_sha256"]
                != decision["effective_selected_mask_sha256"],
                "candidate": baseline["s1_generation"].get("mask_sha256", {}).get("extension_only")
                != diagnostic.get("mask_sha256", {}).get("extension_only"),
                "reasons": baseline_decision.get("s1_failure_reasons", [])
                != decision.get("s1_failure_reasons", []),
                "occluder_neutrality": (
                    baseline_decision.get("v4b_neutral_structural_pixels")
                    != decision.get("v4b_neutral_structural_pixels")
                    or baseline_decision.get("v4b_neutral_containment_pixels")
                    != decision.get("v4b_neutral_containment_pixels")
                ),
            }
            if version == "v4c":
                comparison_flags["candidate"] = bool(
                    np.any(selected["validated_extension"] != masks["extension_only"])
                )
        render_this = comparison_flags is None or any(comparison_flags.values())
        render_masks = dict(masks)
        if version == "v4b":
            render_masks["occlusion_neutral_structural"] = selected[
                "occlusion_neutral_structural"
            ]
            render_masks["occlusion_neutral_containment"] = selected[
                "occlusion_neutral_containment"
            ]
        if version == "v4c":
            render_masks["occlusion_neutral_structural"] = selected[
                "occlusion_neutral_structural"
            ]
            render_masks["occlusion_neutral_containment"] = selected[
                "occlusion_neutral_containment"
            ]
        if version == "v4c" and "validated_extension" in selected:
            render_masks["extension_only"] = selected["validated_extension"]
            render_masks["proposal"] = selected["validated_proposal"]
            render_masks["structural_agreement"] = selected["validated_proposal"] & structural
        if render_this and render_images:
            if version in {"v4b", "v4c"}:
                baseline_variant = comparison_baseline_variant or (
                    "v4a" if version == "v4b" else "v4b_banner" if include_banner_occluder else "v4b"
                )
                if baseline_variant == "v4a":
                    baseline_decision, baseline_selected = evaluate_s1_targeted_v4a(
                        package, diagnostic, masks
                    )
                    baseline_render_masks = masks
                elif baseline_variant in {"v4b", "v4b_banner", "v4c", "v4c_banner"}:
                    baseline_include_banner = baseline_variant.endswith("_banner")
                    baseline_obstacle = _v4b_occluder_mask(
                        reference,
                        structural.shape,
                        include_banner=baseline_include_banner,
                    )
                    baseline_evaluator = (
                        evaluate_s1_targeted_v4c
                        if baseline_variant.startswith("v4c")
                        else evaluate_s1_targeted_v4b
                    )
                    baseline_decision, baseline_selected = baseline_evaluator(
                        package, diagnostic, masks, baseline_obstacle
                    )
                    if baseline_include_banner:
                        baseline_decision["policy_id"] = (
                            f'{baseline_decision["policy_id"]}_banner32'
                        )
                    baseline_render_masks = dict(masks)
                    baseline_render_masks["occlusion_neutral_structural"] = baseline_selected[
                        "occlusion_neutral_structural"
                    ]
                    baseline_render_masks["occlusion_neutral_containment"] = baseline_selected[
                        "occlusion_neutral_containment"
                    ]
                    if baseline_variant.startswith("v4c") and "validated_extension" in baseline_selected:
                        baseline_render_masks["extension_only"] = baseline_selected[
                            "validated_extension"
                        ]
                        baseline_render_masks["proposal"] = baseline_selected[
                            "validated_proposal"
                        ]
                        baseline_render_masks["structural_agreement"] = (
                            baseline_selected["validated_proposal"] & structural
                        )
                else:
                    raise ValueError(f"Unknown comparison baseline variant: {baseline_variant}")
                if baseline is not None and (
                    baseline_decision["s1_status"]
                    != baseline["s1_decision"]["s1_status"]
                    or baseline_decision["effective_selected_mask_sha256"]
                    != baseline["s1_decision"]["effective_selected_mask_sha256"]
                ):
                    raise AssertionError(
                        f"{sample_id}: regenerated baseline does not match saved baseline"
                    )
                _render(
                    reference,
                    v6,
                    v6_masks,
                    structural,
                    blocker,
                    diagnostic,
                    baseline_render_masks,
                    baseline_decision,
                    baseline_selected["effective_selected"],
                    baseline_composites / f"{sample_id}.jpg",
                )
            _render(
                reference,
                v6,
                v6_masks,
                structural,
                blocker,
                diagnostic,
                render_masks,
                decision,
                effective,
                composites / f"{sample_id}.jpg",
            )
            rendered_sample_ids.append(sample_id)
        failures_ko = [_reason_text(value) for value in decision["s1_failure_reasons"]]
        answer = _status_text(decision)
        evidence = " / ".join(failures_ko) if failures_ko else "현재의 모든 공통 검증 조건을 통과함"
        handoff = (
            "S2 검토 대상으로 전달 가능"
            if decision["s2_eligible"]
            else "S2 전달 안 함"
        )
        checks_html = "".join(
            f'<li class="{"pass" if c["passed"] else "fail"}">{html.escape(c["name"])}: '
            f'{"통과" if c["passed"] else "실패"} · 값={html.escape(str(c["value"]))} · 기준={html.escape(c["rule"])}</li>'
            for c in decision["checks"]
        )
        if render_this and render_images:
            cards.append(
                f'''<article id="{html.escape(sample_id)}" data-status="{decision['s1_status']}"><h2>{html.escape(sample_id)} <small>{html.escape(decision['s1_status'])}</small></h2>
<img loading="lazy" src="composites/{html.escape(sample_id)}.jpg" alt="{html.escape(sample_id)} S1 targeted decision panels">
<section><b>V6 분류</b>: {html.escape(v6['final_disposition'])} → {html.escape(_v6_subclassification(v6['final_disposition'], v6, reference)[0])}<br><b>정확한 근거</b>: {html.escape(_v6_subclassification(v6['final_disposition'], v6, reference)[1])}</section>
<section class="qa"><h3>질문</h3><p>현재 S1 공통 로직만으로 이 직선 후보를 실제로 그릴 것인가?</p><h3>답변</h3><p><b>{html.escape(answer)}</b></p><p>핵심 근거: {html.escape(evidence)}. {html.escape(handoff)}.</p><h3>보완 방법</h3><p>{"최소 해상도 폭 정지만 추가했고 검증은 v3 그대로 유지했습니다." if version == "v4a" else "v4a 후보를 그대로 유지하면서 허용된 장애물과 실제로 겹친 픽셀만 p10 분모 및 같은 측 containment 간격에서 중립 처리했습니다. prefix 복구는 적용하지 않았습니다." if version == "v4b" else "v4b 보정을 먼저 적용한 뒤 한 방향에서 7행 연속 문제가 이어질 때 그 지점 바깥만 제거하고 seed-connected 안전 prefix를 유지했습니다." if version == "v4c" else "측면 프레임 접촉 직전 정지와 signed containment를 적용했습니다. 대표성 검사는 적용하지 않았습니다." if version == "v3" else "이번 실행에는 사용자 피드백 기반 보완 규칙을 넣지 않았습니다. 이 결과를 먼저 검토한 뒤 공통 로직의 수정 여부를 결정합니다."}</p></section>
<details><summary>적용된 검증값 전체 보기</summary><ul>{checks_html}</ul></details></article>'''
            )
        items.append(
            {
                "sample_id": sample_id,
                "source_image": reference["source_image"],
                "v6_policy_id": V6_POLICY_ID,
                "v6_final_disposition": v6["final_disposition"],
                "v6_subclassification": _v6_subclassification(
                    v6["final_disposition"], v6, reference
                )[0],
                "v6_subclassification_evidence": _v6_subclassification(
                    v6["final_disposition"], v6, reference
                )[1],
                "v6_reference_equivalent": not _v6_equivalence_mismatches(v6, reference),
                "s1_generation": diagnostic,
                "s1_decision": decision,
                "effective_selected_mask": str(selected_path.resolve()) if render_images else None,
                "effective_selected_mask_file_sha256": selected_file_sha256,
                "comparison_to_baseline": comparison_flags,
            }
        )

    counts = Counter(item["s1_decision"]["s1_status"] for item in items)
    handoff_count = sum(bool(item["s1_decision"]["s2_eligible"]) for item in items)
    elapsed = time.perf_counter() - started
    disposition_breakdown: dict[str, dict[str, Any]] = {}
    for disposition in sorted({item["v6_final_disposition"] for item in items}):
        group = [item for item in items if item["v6_final_disposition"] == disposition]
        subclasses: dict[str, dict[str, Any]] = {}
        for subclass in sorted({item["v6_subclassification"] for item in group}):
            subgroup = [item for item in group if item["v6_subclassification"] == subclass]
            subclasses[subclass] = {
                "input": len(subgroup),
                "constructible": sum(
                    item["s1_generation"]["construction_status"] != "not_constructible"
                    for item in subgroup
                ),
                "not_constructible": sum(
                    item["s1_generation"]["construction_status"] == "not_constructible"
                    for item in subgroup
                ),
                "s1_status_counts": dict(
                    Counter(item["s1_decision"]["s1_status"] for item in subgroup)
                ),
                "evidence": sorted(
                    {item["v6_subclassification_evidence"] for item in subgroup}
                ),
            }
        disposition_breakdown[disposition] = {
            "input": len(group),
            "constructible": sum(
                item["s1_generation"]["construction_status"] != "not_constructible"
                for item in group
            ),
            "not_constructible": sum(
                item["s1_generation"]["construction_status"] == "not_constructible"
                for item in group
            ),
            "s1_status_counts": dict(
                Counter(item["s1_decision"]["s1_status"] for item in group)
            ),
            "subclassifications": subclasses,
        }
    summary = {
        "status": (
            "S1 FULL-160 REGRESSION — USER REVIEW REQUIRED"
            if full_160
            else "PROVISIONAL S1 TARGETED REVIEW — USER APPROVAL REQUIRED"
        ),
        "policy_id": S1_V4_PROBE_POLICY_ID if grade_v4_probe else {
            "v1": S1_TARGETED_POLICY_ID,
            "v2": S1_TARGETED_V2_POLICY_ID,
            "v3": S1_TARGETED_V3_POLICY_ID,
            "v4a": S1_TARGETED_V4A_POLICY_ID,
            "v4b": S1_TARGETED_V4B_POLICY_ID,
            "v4c": S1_TARGETED_V4C_POLICY_ID,
        }[version],
        "baseline": V6_POLICY_ID,
        "sample_count": len(items),
        "s1_status_counts": dict(counts),
        "v6_disposition_breakdown": disposition_breakdown,
        "s2_eligible_count": handoff_count,
        "threshold_status": (
            "provisional_full160_ablation" if version in {"v4a", "v4b", "v4c"} else "provisional_targeted_review"
        ),
        "user_feedback_specific_rules_added": version in {"v4a", "v4b", "v4c"},
        "banner_occluder_enabled": include_banner_occluder,
        "human_review_grading_enabled": grade_v4_probe,
        "comparison_baseline_variant": comparison_baseline_variant,
        "changed_only_against": str(changed_only_against.resolve()) if changed_only_against else None,
        "rendered_changed_sample_count": len(rendered_sample_ids),
        "rendered_sample_ids": rendered_sample_ids,
        "image_rendering_enabled": render_images,
        "official_full_160_executed": full_160,
        "runtime_seconds": elapsed,
        "v6_reference_mismatch_count": sum(not item["v6_reference_equivalent"] for item in items),
    }
    nav = " ".join(f'<a href="#{html.escape(value)}">{html.escape(value)}</a>' for value in rendered_sample_ids)
    version_note = (
        "V4b 장애물 중립화에 Banner(class 32)를 추가하고 V4c 안전 prefix를 적용한 뒤, 생성 마스크를 바꾸지 않는 사람 검토용 등급을 부여했습니다."
        if grade_v4_probe
        else
        "V6 coherent_turn/unstable_bilateral 분류는 참고값으로만 표시하며 채택·기각에 사용하지 않습니다. "
        "검증은 기본 형상, 실제 연장 범위의 구조 지속성, 실제 연장 범위의 지속적 관측 모순으로 단순화했습니다."
        if version == "v2"
        else (
            "V6 분류는 집계용으로만 사용합니다. 측면 프레임 접촉 직전 정지와 signed containment를 적용하고, v2 임계값은 유지하며 대표성 검사는 제외했습니다."
            if version == "v3"
            else (
                "V3 생성·검증을 그대로 유지하고, 좌우 폭이 경계 불확실성보다 작아지기 직전에 정지하는 최소 해상도 폭 규칙만 추가했습니다."
                if version == "v4a"
                else (
                "V4a 후보는 변경하지 않고 허용된 occluder와 실제로 겹친 픽셀만 p10과 같은 측 containment에서 중립 처리했습니다. prefix salvage는 적용하지 않았습니다."
                if version == "v4b"
                else (
                "V4b 검증 후 한 방향에서 7행 연속으로 구조 또는 containment 문제가 이어질 때 그 지점부터 바깥만 제거하고 안전 prefix를 다시 검증합니다."
                if version == "v4c"
                else "현재 설계의 공통 근거만으로 후보 생성 후 채택/기각을 결정합니다."
                )
                )
            )
        )
    )
    breakdown_rows = "".join(
        f"<tr><td>{html.escape(disposition)}</td><td>{html.escape(subclass)}</td><td>{value['input']}</td><td>{value['constructible']}</td><td>{value['not_constructible']}</td><td>{html.escape(str(value['s1_status_counts']))}</td><td>{html.escape(' / '.join(value['evidence']))}</td></tr>"
        for disposition, parent in disposition_breakdown.items()
        for subclass, value in parent["subclassifications"].items()
    )
    grading_rows = "".join(
        f"<tr><td>{html.escape(item['sample_id'])}</td><td>{html.escape(item['s1_decision']['s1_status'])}</td>"
        f"<td>{html.escape(', '.join(item['s1_decision'].get('quality_grading_reasons', [])) or '-')}</td>"
        f"<td>{html.escape(str(item['s1_decision'].get('quality_metrics', {})))}</td></tr>"
        for item in items
    )
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>S1 targeted provisional {version}</title><style>
body{{font:15px/1.55 system-ui;background:#0d1724;color:#eef;max-width:2000px;margin:auto;padding:18px}}header{{background:#17283c;padding:18px 18px 28px;position:sticky;top:0;z-index:2;height:30vh;min-height:110px;max-height:90vh;overflow:auto;resize:vertical;box-sizing:border-box}}header::after{{content:"↕ 오른쪽 아래 모서리를 드래그하여 상단 영역 높이 조절";position:absolute;right:24px;bottom:3px;color:#9fdcff;background:#17283c;padding:1px 8px;font-size:12px;pointer-events:none}}a{{color:#7bddff;margin-right:9px}}article{{background:#18283d;margin:20px 0;padding:16px;border-radius:12px}}h2{{margin:0 0 8px}}small{{color:#7ff0bd}}img{{width:100%;height:auto;border-radius:7px}}section,details{{padding:12px;background:#102033;margin-top:10px}}.pipeline{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.pipeline div{{background:#0d1724;padding:10px;border-left:4px solid #7bddff}}table{{width:100%;border-collapse:collapse;margin:12px 0;background:#102033}}th,td{{border:1px solid #456;padding:6px 8px;text-align:left;vertical-align:top}}th{{background:#203752;position:sticky;top:0}}.pass{{color:#9af0b8}}.fail{{color:#ff9c9c}}input,select{{font:inherit;padding:7px;background:#0d1724;color:#eef;border:1px solid #567}}nav{{font-size:13px}}@media(max-width:900px){{.pipeline{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Extension Track S1 — {'full-160 회귀' if full_160 else 'targeted 자율 판단'} ({version})</h1><p>{html.escape(version_note)} 후보가 보여도 8번 패널이 NONE이면 최종적으로 그리지 않습니다.</p><div class="pipeline"><div><b>1. 후보 생성</b><br>constructible이면 V6 분류와 무관하게 진단 실행</div><div><b>2. 독립 정지</b><br>구조 끝·최대 거리·측면 frame·최소 해상도 폭에서 정지</div><div><b>3. 검증</b><br>v3와 동일한 구조 지속성·signed containment</div><div><b>4. 최종 선택</b><br>모든 조건을 통과한 경우만 실제 S1 마스크로 선택</div></div><p>전체 계산={len(items)} · 변경분 렌더링={len(rendered_sample_ids)} · counts={html.escape(str(dict(counts)))} · S2 eligible={handoff_count} · V6 mismatches={summary['v6_reference_mismatch_count']} · full160={str(full_160).lower()}</p><table><thead><tr><th>V6 최상위 분류</th><th>하위분류</th><th>입력</th><th>constructible</th><th>non-constructible</th><th>S1 결과</th><th>정확한 근거</th></tr></thead><tbody>{breakdown_rows}</tbody></table>{f'<h2>등급 판정 전체 목록</h2><table><thead><tr><th>sample</th><th>grade</th><th>reason</th><th>metrics</th></tr></thead><tbody>{grading_rows}</tbody></table>' if grade_v4_probe else ''}<p><input id="search" placeholder="sample ID 검색"> <select id="status"><option value="">모든 결과</option><option>selected</option><option>selected_partial</option><option>ambiguous_hold</option><option>unchanged</option><option>failed</option><option>no_extension_needed</option></select> <b id="visible"></b></p><nav>{nav}</nav></header>{''.join(cards)}<script>
const aa=[...document.querySelectorAll('article')],q=document.querySelector('#search'),s=document.querySelector('#status'),v=document.querySelector('#visible');function f(){{let n=0;for(const a of aa){{const ok=(!q.value||a.id.toLowerCase().includes(q.value.toLowerCase()))&&(!s.value||a.dataset.status===s.value);a.style.display=ok?'':'none';if(ok)n++;}}v.textContent=`표시 ${{n}}/${{aa.length}}`;}}q.oninput=s.onchange=f;f();
</script></body></html>'''
    (output / "results.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = output / "review.html"
    review.write_text(page, encoding="utf-8")
    return review


def main() -> None:
    """Run the accepted Phase 7-A S1 boundary workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4a", "v4b", "v4c"), default="v1")
    parser.add_argument("--full-160", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--changed-only-against", type=Path)
    parser.add_argument("--include-banner-occluder", action="store_true")
    parser.add_argument("--grade-v4-probe", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--comparison-baseline-variant",
        choices=("v4a", "v4b", "v4b_banner", "v4c", "v4c_banner"),
    )
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()
    destination = args.output or (
        V4A_FULL160_OUTPUT
        if args.full_160 and args.version == "v4a"
        else V4B_FULL160_OUTPUT
        if args.full_160 and args.version == "v4b"
        else V4C_FULL160_OUTPUT
        if args.full_160 and args.version == "v4c"
        else V3_FULL160_OUTPUT
        if args.full_160
        else V4A_OUTPUT
        if args.version == "v4a"
        else V4B_OUTPUT
        if args.version == "v4b"
        else V4C_OUTPUT
        if args.version == "v4c"
        else V3_OUTPUT
        if args.version == "v3"
        else V2_OUTPUT
        if args.version == "v2"
        else DEFAULT_OUTPUT
    )
    selected_targets = tuple(args.sample_id) if args.sample_id else (
        tuple(
            item["sample_id"]
            for item in json.loads((REFERENCE / "results.json").read_text(encoding="utf-8"))
        )
        if args.full_160
        else TARGETS
    )
    print(
        run(
            destination,
            version=args.version,
            targets=selected_targets,
            full_160=args.full_160,
            changed_only_against=args.changed_only_against,
            include_banner_occluder=args.include_banner_occluder,
            grade_v4_probe=args.grade_v4_probe,
            comparison_baseline_variant=args.comparison_baseline_variant,
            render_images=not args.no_render,
        ).resolve()
    )


if __name__ == "__main__":
    main()
