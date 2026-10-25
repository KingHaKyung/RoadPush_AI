from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "v2_2a2_rowwise_counterfactual_boundary_completion_probe_40_v1"
AUDIT = HERE / "model_audit_40" / "audit_manifest.json"
TITLE = "V2.2-A2 — Rowwise Counterfactual Boundary Completion"



from roadpush_ai.legacy_policy import grounded_footprint as grounded
integrated = grounded.integrated
outer = grounded.outer
policy = grounded.policy


def row_spans(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = component.shape[0]
    valid = component.any(axis=1)
    left = np.full(h, -1, dtype=np.int32)
    right = np.full(h, -1, dtype=np.int32)
    for y in np.flatnonzero(valid):
        xs = np.flatnonzero(component[y])
        left[y], right[y] = int(xs[0]), int(xs[-1])
    return valid, left, right


def blocker_supported_boundaries(
    component: np.ndarray,
    ground_blocked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Complete only boundary notches supported by an existing ground blocker.

    No sidewalk pixels are written.  The returned arrays are temporary rowwise
    coordinates used only by the width analysis.
    """
    valid, left, right = row_spans(component)
    inferred_left = left.copy()
    inferred_right = right.copy()
    h, w = component.shape
    support = cv2.dilate(
        ground_blocked.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 21)),
    ).astype(bool)
    left_candidate = np.zeros(h, dtype=bool)
    right_candidate = np.zeros(h, dtype=bool)
    for y in np.flatnonzero(valid):
        lx, rx = int(left[y]), int(right[y])
        radius = max(8, round((rx - lx + 1) * 0.08))
        left_candidate[y] = support[y, max(0, lx - radius):min(w, lx + radius + 1)].any()
        right_candidate[y] = support[y, max(0, rx - radius):min(w, rx + radius + 1)].any()

    records: list[dict] = []
    max_search = max(25, round(h * 0.16))

    def trusted_neighbor(y: int, direction: int, candidates: np.ndarray):
        for distance in range(1, max_search + 1):
            yy = y + direction * distance
            if yy < 0 or yy >= h:
                break
            if valid[yy] and not candidates[yy]:
                return yy
        return None

    for side, current, candidates, inward_sign in (
        ("left", left, left_candidate, 1),
        ("right", right, right_candidate, -1),
    ):
        target = inferred_left if side == "left" else inferred_right
        for y in np.flatnonzero(valid & candidates):
            above = trusted_neighbor(int(y), -1, candidates)
            below = trusted_neighbor(int(y), 1, candidates)
            if above is None or below is None or below == above:
                continue
            ratio = (y - above) / (below - above)
            predicted = int(round(float(current[above]) + ratio * float(current[below] - current[above])))
            deviation = (int(current[y]) - predicted) * inward_sign
            local_width = int(right[y] - left[y] + 1)
            minimum = max(3, round(local_width * 0.012))
            maximum = max(18, round(local_width * 0.28))
            if deviation < minimum or deviation > maximum:
                continue
            target[y] = predicted
            records.append({
                "y": int(y), "side": side, "visible": int(current[y]),
                "inferred": predicted, "extension_px": int(deviation),
                "support_above_y": int(above), "support_below_y": int(below),
            })
    return left, right, inferred_left, inferred_right, records


def span_mask(valid: np.ndarray, left: np.ndarray, right: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y in np.flatnonzero(valid):
        if left[y] >= 0 and right[y] >= left[y]:
            mask[y, int(left[y]):int(right[y]) + 1] = True
    return mask


def analyse_rows(
    corridor_mask: np.ndarray,
    free_mask: np.ndarray,
    labels: np.ndarray,
    path: list[tuple[int, int]],
) -> tuple[np.ndarray, list[dict]]:
    component = outer.selected_component(corridor_mask, path)
    if not component.any():
        component = corridor_mask
    records = outer.sample_outer_boundaries(component, path, labels)
    for row in records:
        segments = outer.base.free_segments(free_mask, row["y"], row["left"], row["right"])
        selected = outer.choose_route_segment(segments, row["anchor_x"])
        row["free_segments"] = [list(x) for x in segments]
        row["selected_free_segment"] = list(selected) if selected else None
        row["selected_free_width"] = selected[1] - selected[0] + 1 if selected else 0
    return component, records


def draw_inference(image: np.ndarray, visible_left, visible_right, inferred_left, inferred_right, sampled_y) -> np.ndarray:
    panel = image.copy()
    for y in sampled_y:
        if visible_left[y] < 0:
            continue
        cv2.line(panel, (int(visible_left[y]), y), (int(visible_right[y]), y), (235, 70, 220), 2, cv2.LINE_AA)
        if inferred_left[y] != visible_left[y]:
            cv2.line(panel, (int(inferred_left[y]), y), (int(visible_left[y]), y), (255, 145, 35), 6, cv2.LINE_AA)
        if inferred_right[y] != visible_right[y]:
            cv2.line(panel, (int(visible_right[y]), y), (int(inferred_right[y]), y), (255, 145, 35), 6, cv2.LINE_AA)
    return panel


def process(item: dict, model_holder: dict) -> dict:
    sid = item["sample_id"]
    image, labels = integrated.load_inputs(item)
    repaired, walkway = integrated.selected_walkway(labels)
    blocked_ids = tuple(policy.probe.v2.OCCUPANCY) + tuple(policy.probe.v2.OCCLUSION)
    raw_objects = np.isin(labels, blocked_ids)
    raw_hazards = np.isin(labels, tuple(policy.probe.v2.HAZARD))
    structural, _ = outer.base.obstacle_transparent_structure(repaired, raw_objects | raw_hazards)
    baseline_fp, baseline_rows, yolo_meta, _ = integrated.integrated_footprints(labels, structural, image, model_holder)

    natural_fp, natural_source, _, _, natural_rows, _ = grounded.universal_hole_candidates(
        labels, repaired, structural, walkway, baseline_fp
    )
    candidate_structural, _ = outer.base.obstacle_transparent_structure(
        repaired, raw_objects | raw_hazards | natural_source
    )
    if natural_source.any() and not (candidate_structural & ~structural).any():
        natural_fp[:] = False
        natural_source[:] = False
        for row in natural_rows:
            if row.get("accepted"):
                row["accepted"] = False
                row["rejection_reason"] = "no_new_occluded_sidewalk_restoration"

    ground_blocked = raw_hazards | baseline_fp | natural_fp
    _, curb_suspect, _, _ = integrated.curb_probe.analyze(labels)
    route_surface = repaired & ~curb_suspect
    if not route_surface.any():
        route_surface = repaired
    _, path = outer.route_path(route_surface)
    base_component = outer.selected_component(structural & ~curb_suspect, path)
    if not base_component.any():
        base_component = structural & ~curb_suspect

    valid, visible_left, visible_right = row_spans(base_component)
    _, _, inferred_left, inferred_right, inference_rows = blocker_supported_boundaries(
        base_component, ground_blocked
    )
    visible_corridor = span_mask(valid, visible_left, visible_right, labels.shape)
    inferred_corridor = span_mask(valid, inferred_left, inferred_right, labels.shape)

    # Counterfactual corridor is a temporary geometric hypothesis.  It is not
    # written back to repaired/structural masks.  Existing blockers remain.
    before_free = structural & ~ground_blocked & ~curb_suspect
    after_free = inferred_corridor & ~ground_blocked & ~curb_suspect
    before_component, before_rows = grounded.boundary_analysis(
        structural & ~curb_suspect, before_free, labels, path
    )
    after_component, after_rows = analyse_rows(inferred_corridor, after_free, labels, path)
    before_by_y = {r["y"]: r for r in before_rows}
    after_by_y = {r["y"]: r for r in after_rows}
    free_changed = sum(
        1 for y in set(before_by_y) | set(after_by_y)
        if y not in before_by_y or y not in after_by_y or
        before_by_y[y].get("selected_free_segment") != after_by_y[y].get("selected_free_segment")
    )
    sampled_y = sorted(set(before_by_y) | set(after_by_y))

    base_view = grounded.paint(image, repaired, (45, 210, 95), 0.38)
    base_view = grounded.paint(base_view, ground_blocked, (255, 195, 35), 0.88)
    inference_view = draw_inference(
        image, visible_left, visible_right, inferred_left, inferred_right, sampled_y
    )
    before_view = grounded.draw_boundary_rows(image, before_component, before_rows)
    hypothesis_view = grounded.paint(image, inferred_corridor, (255, 145, 35), 0.22)
    hypothesis_view = grounded.paint(hypothesis_view, ground_blocked, (255, 195, 35), 0.90)
    after_view = grounded.draw_boundary_rows(image, after_component, after_rows)
    free_view = grounded.draw_free_rows(image, after_free, after_rows)

    art = OUT / "artifacts" / sid
    art.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(art / "01_visible_row_corridor.png"), visible_corridor.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02_counterfactual_row_corridor.png"), inferred_corridor.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "03_ground_blocked.png"), ground_blocked.astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = [base_view, inference_view, before_view, hypothesis_view, after_view, free_view]
    titles = [
        "1 BASE: V2.1 grounded-footprint-only (yellow=blocked)",
        "2 Rowwise boundary inference (magenta=visible, orange=completed)",
        "3 BEFORE: structural boundary + width rows",
        "4 Temporary row-corridor hypothesis only (not written to mask)",
        "5 AFTER: counterfactual boundary + width rows",
        "6 AFTER: free rows after existing blockers",
    ]
    for ax, title, panel in zip(axes.flat, titles, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{sid} | {TITLE} | completed rows {len(inference_rows)} | free rows changed {free_changed}",
        fontsize=14,
    )
    fig.savefig(art / "summary.png", dpi=115)
    plt.close(fig)

    before_widths = [r["selected_free_width"] for r in before_rows if r["selected_free_width"]]
    after_widths = [r["selected_free_width"] for r in after_rows if r["selected_free_width"]]
    payload = {
        "sample_id": sid,
        "order": item.get("order"),
        "split": item.get("split"),
        "completed_boundary_rows": len(inference_rows),
        "left_completed_rows": sum(r["side"] == "left" for r in inference_rows),
        "right_completed_rows": sum(r["side"] == "right" for r in inference_rows),
        "free_width_rows_changed": free_changed,
        "median_selected_free_width_before_px": float(np.median(before_widths)) if before_widths else None,
        "median_selected_free_width_after_px": float(np.median(after_widths)) if after_widths else None,
        "natural_grounded_footprints": sum(bool(r.get("accepted")) for r in natural_rows),
        "yolo_state": (
            "split_confirmed" if yolo_meta.get("split_confirmed_count") else
            "triggered_fallback" if yolo_meta.get("triggered") else "not_triggered"
        ),
        "boundary_inference": inference_rows,
        "policy": {
            "base": "V2.1 grounded-footprint-only",
            "probe_only": True,
            "structural_mask_modified": False,
            "pixel_restoration_used": False,
            "counterfactual_boundary_is_rowwise_only": True,
            "boundary_completion_requires_ground_blocker_support": True,
            "existing_ground_blockers_subtracted": True,
            "v2_2_b_hard_veto_applied": False,
        },
    }
    (art / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    model_holder: dict = {}
    results = [process(item, model_holder) for item in items]
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "order", "sample_id", "split", "completed_boundary_rows", "left_completed_rows",
        "right_completed_rows", "free_width_rows_changed", "median_selected_free_width_before_px",
        "median_selected_free_width_after_px", "natural_grounded_footprints", "yolo_state",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in results)

    completed = [r["sample_id"] for r in results if r["completed_boundary_rows"]]
    free_changed = [r["sample_id"] for r in results if r["free_width_rows_changed"]]
    yolo_samples = [r["sample_id"] for r in results if r["yolo_state"] != "not_triggered"]
    failures = []
    if len(results) != 40:
        failures.append("sample_count")
    if yolo_samples != ["IMG_4548", "commons_04"]:
        failures.append(f"unexpected_yolo_samples:{yolo_samples}")
    validation = {
        "status": "execution_valid_visual_review_pending_not_integrated" if not failures else "failed_execution_validation",
        "title": TITLE,
        "base": "V2.1 grounded-footprint-only",
        "expected_samples": 40,
        "processed_samples": len(results),
        "boundary_completed_samples": completed,
        "free_width_changed_samples": free_changed,
        "structural_mask_modified": False,
        "conditional_yolo_samples": yolo_samples,
        "failures": failures,
    }
    (OUT / "completion_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cards = []
    for row in results:
        sid = row["sample_id"]
        cls = "changed" if row["free_width_rows_changed"] else "unchanged"
        cards.append(
            f'<section class="{cls}"><h2>{int(row.get("order") or 0):02d}. {html.escape(sid)} '
            f'<small>{html.escape(str(row.get("split", "")))}</small></h2>'
            f'<p>boundary rows completed {row["completed_boundary_rows"]} · free rows changed '
            f'{row["free_width_rows_changed"]} · width {row["median_selected_free_width_before_px"]} → '
            f'{row["median_selected_free_width_after_px"]}px · YOLO {html.escape(row["yolo_state"])}</p>'
            f'<a href="artifacts/{html.escape(sid)}/result.json">result.json</a>'
            f'<img loading="lazy" src="artifacts/{html.escape(sid)}/summary.png"></section>'
        )
    page = (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(TITLE)}</title>'
        '<style>body{font:16px system-ui;background:#101722;color:#eef;max-width:1550px;margin:auto;padding:18px}'
        'section{background:#192331;margin:24px 0;padding:18px;border-radius:12px}'
        'section.changed{border:3px solid #43ded5}img{width:100%;height:auto}small{color:#8ff}a{color:#8df}</style>'
        f'<h1>{html.escape(TITLE)} — 40 images</h1>'
        '<p><b>기준:</b> V2.1 grounded-footprint-only · <b>옵션:</b> A2 rowwise counterfactual boundary · '
        '<b>structural mask:</b> 변경 없음 · <b>A1:</b> 보존 · <b>판정:</b> 시각 검토 대기·미편입</p>' + ''.join(cards)
    )
    (OUT / "review.html").write_text(page, encoding="utf-8")
    (OUT / "REPORT.md").write_text(
        f"# {TITLE} — 40 images\n\n"
        "- 기준: V2.1 grounded-footprint-only\n"
        "- A1 pixel-mask restoration과 독립된 두 번째 옵션\n"
        "- repaired/structural mask는 변경하지 않음\n"
        "- ground blocker가 지지하는 외곽경계의 안쪽 패임만 인접 신뢰 행으로 보간\n"
        "- 폭 계산용 임시 row corridor에서 기존 footprint/hazard를 차감\n"
        "- V2.2-B exact-row hard veto 및 향후 silhouette/footprint selector는 적용하지 않음\n"
        f"- Boundary-completed samples: {', '.join(completed) if completed else '없음'}\n"
        f"- Free-width changed samples: {', '.join(free_changed) if free_changed else '없음'}\n"
        f"- Validation: {validation['status']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
