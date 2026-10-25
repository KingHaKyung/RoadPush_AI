from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import textwrap
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from roadpush_ai.phase7a.extension_s1 import mask_sha256
from roadpush_ai.phase7b.handoff import Phase7AAdapter, Phase7BHandoff
from roadpush_ai.phase7b.occupancy_builder import (
    ExistingOccupancyEvidenceIndex,
    Phase7BOccupancyResult,
)
from roadpush_ai.phase7b.pipeline import (
    Phase7BPipelineBundle,
    compose_phase7_b_result,
    run_phase7_b_sample,
    validate_phase7_b_record,
)


DEFAULT_REAL_CONTROLS = (
    "IMG_4282",
    "MP_SEL_PN000695",
    "MP_SEL_PN000601",
    "alley_003",
    "alley_014",
    "MP_SEL_SUR_001386",
    "IMG_4508",
    "commons_01",
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_bundle(sample_id: str, kind: str) -> tuple[Phase7BPipelineBundle, np.ndarray]:
    height, width = 240, 320
    structural = np.zeros((height, width), dtype=bool)
    structural[30:220, 40:280] = True
    occupied = np.zeros_like(structural)
    if kind == "central_split":
        occupied[30:220, 150:170] = True
    elif kind == "staggered_no_teleport":
        occupied[30:125, 150:280] = True
        occupied[125:220, 40:170] = True
    elif kind == "one_row_notch":
        occupied[125, 80:280] = True
    elif kind == "sustained_narrowing":
        occupied[124:126, 160:280] = True
    elif kind == "full_closure":
        occupied[125, 40:280] = True
    elif kind == "split_merge_best_right":
        occupied[31:219, 110:150] = True
    else:
        raise ValueError(kind)
    rows = tuple(range(30, 220))
    handoff = Phase7BHandoff(
        sample_id=sample_id,
        selected_surface="sidewalk",
        phase7a_policy_id="synthetic_focused_control_v1",
        phase7a_status="selected",
        measurement_scope="full",
        measurement_authority="allowed",
        result_authority="authoritative",
        human_review_required=False,
        authorized_row_range=(30, 219),
        observed_rows=rows,
        generated_rows=(),
        structural_mask_sha256=mask_sha256(structural),
        structural_area_px=int(structural.sum()),
        provenance={"control_kind": kind, "source_image": None},
        structural_mask=structural,
        typed_null_result=None,
    )
    occupancy = Phase7BOccupancyResult(
        sample_id=sample_id,
        coverage_status="synthetic_verified",
        abstention_reasons=(),
        occupancy_mask=occupied,
        occupancy_mask_sha256=mask_sha256(occupied),
        occupancy_pixels=int(occupied.sum()),
        base_blocker_pixels=int(occupied.sum()),
        added_completion_pixels=0,
        clipped_blocker_pixels=0,
        instance_records=(),
        evidence_source="synthetic_focused_control",
        upstream_hashes_before={},
        upstream_hashes_after={},
    )
    base = np.full((height, width, 3), (66, 66, 66), dtype=np.uint8)
    for y in range(height):
        base[y, :, :] = (45 + y // 8, 48 + y // 10, 52 + y // 12)
    cv2.rectangle(base, (40, 30), (279, 219), (125, 125, 125), 2)
    return compose_phase7_b_result(handoff, occupancy), base


def _blend(base: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    panel = base.copy()
    if mask is not None:
        logical = np.asarray(mask, dtype=bool)
        panel[logical] = (
            panel[logical].astype(np.float32) * (1.0 - alpha)
            + np.asarray(color, dtype=np.float32) * alpha
        ).astype(np.uint8)
    return panel


def _put(panel: np.ndarray, text: str, y: int, *, color=(245, 245, 245), scale=0.55) -> None:
    cv2.putText(
        panel,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _headed(panel: np.ndarray, title: str, size=(500, 300)) -> np.ndarray:
    resized = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
    result = np.zeros((size[1] + 32, size[0], 3), dtype=np.uint8)
    result[:32] = (35, 23, 16)
    result[32:] = resized
    _put(result, title, 22, scale=0.52)
    return result


def _source_image(bundle: Phase7BPipelineBundle) -> np.ndarray:
    source = bundle.handoff.provenance.get("source_image")
    if source:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        target = bundle.handoff.structural_mask
        if target is None:
            target = bundle.occupancy.occupancy_mask
        if target is not None and image.shape[:2] != target.shape:
            image = cv2.resize(
                image,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return image
    if bundle.handoff.structural_mask is None:
        return np.full((360, 640, 3), (55, 55, 55), dtype=np.uint8)
    h, w = bundle.handoff.structural_mask.shape
    return np.full((h, w, 3), (55, 55, 55), dtype=np.uint8)


def _short(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


_KOREAN_FONT = Path(r"C:\Windows\Fonts\malgun.ttf")
_KOREAN_BOLD_FONT = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = _KOREAN_BOLD_FONT if bold and _KOREAN_BOLD_FONT.is_file() else _KOREAN_FONT
    if not path.is_file():
        raise FileNotFoundError("A Korean-capable Malgun Gothic font is required")
    return ImageFont.truetype(str(path), size=size)


def _percent(value: float | None) -> str:
    return "계산 불가" if value is None else f"{value * 100:.1f}%"


def _median_representative_row(
    rows: Sequence[int], values: Sequence[float], median: float, center: int
) -> int:
    if len(rows) != len(values) or not rows:
        raise ValueError("row/value evidence must be non-empty and aligned")
    exact = [
        int(row)
        for row, value in zip(rows, values)
        if np.isclose(float(value), float(median), rtol=1e-12, atol=1e-12)
    ]
    candidates = exact or [
        int(rows[int(np.argmin(np.abs(np.asarray(values, dtype=float) - median)))])
    ]
    return min(candidates, key=lambda row: (abs(row - int(center)), row))


def _logical_runs(row: np.ndarray, left: int, right: int) -> list[list[int]]:
    xs = np.flatnonzero(np.asarray(row[left : right + 1], dtype=bool)) + left
    if not len(xs):
        return []
    breaks = np.flatnonzero(np.diff(xs) > 1) + 1
    return [
        [int(group[0]), int(group[-1])]
        for group in np.split(xs, breaks)
        if len(group)
    ]


def _dashboard_panel_data(bundle: Phase7BPipelineBundle) -> dict[str, Any]:
    """Return the exact representative-row geometry used by panel 6."""

    record = bundle.record
    data: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "selected_path": list(record["selected_corridor_path"]),
        "bottleneck_row": None,
        "selected_interval": None,
        "selected_row_ratio": None,
        "other_free_intervals": [],
        "obstruction_row": None,
        "obstacle_intervals": [],
    }
    bottleneck = bundle.metrics.bottleneck
    if bottleneck and bottleneck.get("kind") == "three_row_robust_bottleneck":
        row = _median_representative_row(
            bottleneck["row_band"],
            bottleneck["row_passable_ratios"],
            bottleneck["three_row_median_ratio"],
            bottleneck["center_row"],
        )
        selected_by_row = {
            int(item["y"]): item for item in bundle.metrics.selected_corridor_path
        }
        selected = list(selected_by_row[row]["interval"])
        span = bundle.rows.structural_spans[row]
        span_width = span[1] - span[0] + 1
        all_intervals = [list(value) for value in bundle.topology.row_intervals[row]]
        others = [
            {
                "interval": interval,
                "row_relative_ratio": (interval[1] - interval[0] + 1) / span_width,
            }
            for interval in all_intervals
            if interval != selected
        ]
        data.update(
            {
                "bottleneck_row": row,
                "selected_interval": selected,
                "selected_row_ratio": float(selected_by_row[row]["passable_ratio"]),
                "other_free_intervals": others,
            }
        )
    obstruction = bundle.metrics.maximum_obstruction
    if obstruction and bundle.handoff.structural_mask is not None and bundle.occupancy.occupancy_mask is not None:
        row = _median_representative_row(
            obstruction["row_band"],
            obstruction["row_ratios"],
            obstruction["three_row_median_ratio"],
            obstruction["center_row"],
        )
        left, right = bundle.rows.structural_spans[row]
        occupied = np.asarray(bundle.occupancy.occupancy_mask, dtype=bool)
        structural = np.asarray(bundle.handoff.structural_mask, dtype=bool)
        data["obstruction_row"] = row
        data["obstacle_intervals"] = _logical_runs(
            occupied[row] & structural[row], left, right
        )
    separate_obstruction_row = bool(
        data["bottleneck_row"] is not None
        and data["obstruction_row"] is not None
        and data["bottleneck_row"] != data["obstruction_row"]
    )
    data["obstruction_is_separate_row"] = separate_obstruction_row
    data["obstacle_legend_label"] = (
        "장애물 최대" if separate_obstruction_row else "장애물 점유"
    )
    data["maximum_obstruction_summary_label"] = (
        "최대 장애물 점유 폭(별도 위치)"
        if separate_obstruction_row
        else "최대 장애물 점유 폭"
    )
    return data


def _capped_segment(
    draw: ImageDraw.ImageDraw,
    interval: Sequence[int],
    y: int,
    *,
    source_width: int,
    source_height: int,
    color: tuple[int, int, int],
    line_width: int = 5,
) -> None:
    sx = lambda x: int(round(float(x) / max(1, source_width - 1) * 299))
    sy = int(round(float(y) / max(1, source_height - 1) * 299))
    left, right = sx(interval[0]), sx(interval[1])
    draw.line((left, sy, right, sy), fill=color, width=line_width)
    cap = 6
    draw.line((left, sy - cap, left, sy + cap), fill=color, width=line_width)
    draw.line((right, sy - cap, right, sy + cap), fill=color, width=line_width)


def _output_contract_panel(
    bundle: Phase7BPipelineBundle, base: np.ndarray | None = None
) -> np.ndarray:
    """Render a Korean review dashboard at the final 500x300 resolution."""

    record = bundle.record
    base = _source_image(bundle) if base is None else base
    source_height, source_width = base.shape[:2]
    mini = cv2.resize(base, (300, 300), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(mini, cv2.COLOR_BGR2RGB)).convert("RGBA")
    path_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    path_draw = ImageDraw.Draw(path_layer)
    data = _dashboard_panel_data(bundle)
    for item in data["selected_path"]:
        y = int(round(item["y"] / max(1, source_height - 1) * 299))
        left = int(round(item["interval"][0] / max(1, source_width - 1) * 299))
        right = int(round(item["interval"][1] / max(1, source_width - 1) * 299))
        path_draw.line((left, y, right, y), fill=(42, 210, 95, 92), width=2)
    image = Image.alpha_composite(image, path_layer)
    panel = Image.new("RGB", (500, 300), (23, 29, 38))
    panel.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(panel)
    if data["selected_interval"] is not None:
        _capped_segment(
            draw,
            data["selected_interval"],
            data["bottleneck_row"],
            source_width=source_width,
            source_height=source_height,
            color=(45, 232, 111),
        )
        for other in data["other_free_intervals"]:
            _capped_segment(
                draw,
                other["interval"],
                data["bottleneck_row"],
                source_width=source_width,
                source_height=source_height,
                color=(86, 153, 214),
            )
    for interval in data["obstacle_intervals"]:
        _capped_segment(
            draw,
            interval,
            data["obstruction_row"],
            source_width=source_width,
            source_height=source_height,
            color=(247, 72, 75),
        )

    # Keep the legend away from the representative lines so their capped
    # geometry stays directly visible even for far/top bottlenecks.
    representative_ys = [
        value
        for value in (data["bottleneck_row"], data["obstruction_row"])
        if value is not None
    ]
    scaled_representative_ys = [
        int(round(value / max(1, source_height - 1) * 299))
        for value in representative_ys
    ]
    legend_y = 229 if any(value < 70 for value in scaled_representative_ys) else 6
    # The compact legend labels the exact colored representative-row geometry.
    draw.rounded_rectangle(
        (6, legend_y, 294, legend_y + 59), radius=5, fill=(12, 18, 27)
    )
    draw.line((14, legend_y + 13, 34, legend_y + 13), fill=(45, 232, 111), width=5)
    draw.text(
        (40, legend_y + 4),
        f"선택 통행 {_percent(record['minimum_passable_ratio'])}",
        font=_font(12, bold=True),
        fill=(225, 255, 235),
    )
    other_text = "경로 미선택 없음"
    if data["other_free_intervals"]:
        other_text = "경로 미선택 " + ", ".join(
            _percent(item["row_relative_ratio"])
            for item in data["other_free_intervals"]
        )
    draw.line((14, legend_y + 32, 34, legend_y + 32), fill=(86, 153, 214), width=5)
    draw.text((40, legend_y + 23), other_text, font=_font(11), fill=(216, 231, 247))
    draw.line((14, legend_y + 50, 34, legend_y + 50), fill=(247, 72, 75), width=5)
    draw.text(
        (40, legend_y + 41),
        (
            f"{data['obstacle_legend_label']} "
            f"{_percent(record['maximum_obstructed_width_ratio'])}"
            + (" · 별도 행" if data["obstruction_is_separate_row"] else "")
        ),
        font=_font(11),
        fill=(255, 225, 225),
    )

    x = 310
    provisional = record["result_authority"] == "provisional_review"
    if provisional:
        draw.text(
            (x, 5), "검토용 가계산", font=_font(12, bold=True), fill=(255, 188, 79)
        )
        draw.text(
            (x, 21), "영역 승인 필요", font=_font(12, bold=True), fill=(255, 188, 79)
        )
    else:
        draw.text(
            (x, 10), "통행 가능 폭 계산", font=_font(13, bold=True), fill=(231, 241, 250)
        )
    summary = [
        ("상대 통행 가능 폭(주값)", _percent(record["minimum_passable_ratio"]), (75, 232, 128)),
        ("단일 행 최저(진단)", _percent(record["raw_minimum_passable_ratio"]), (210, 220, 230)),
        (data["maximum_obstruction_summary_label"], _percent(record["maximum_obstructed_width_ratio"]), (255, 112, 112)),
        ("전체 장애물 점유 면적", _percent(record["obstacle_area_ratio"]), (235, 205, 145)),
    ]
    y = 43 if provisional else 38
    for label, value, color in summary:
        draw.text((x, y), label, font=_font(10), fill=(177, 194, 210))
        draw.text((x, y + 13), value, font=_font(15, bold=True), fill=color)
        y += 43
    draw.text(
        (x, 215),
        f"범위 {record['measurement_scope']} · 검토 {'필요' if record['human_review_required'] else '불필요'}",
        font=_font(10),
        fill=(206, 216, 226),
    )
    draw.text(
        (x, 233),
        f"권한 {record['result_authority']}",
        font=_font(10),
        fill=(255, 188, 79) if provisional else (206, 216, 226),
    )
    draw.text(
        (x, 258),
        "절대 폭: 입력값 없음 · 계산 불가",
        font=_font(10),
        fill=(170, 184, 198),
    )
    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


def _headed_korean(panel: np.ndarray, title: str) -> np.ndarray:
    result = np.zeros((332, 500, 3), dtype=np.uint8)
    result[:32] = (35, 23, 16)
    result[32:] = panel
    image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(
        (12, 7), title, font=_font(13, bold=True), fill=(245, 245, 245)
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _render_panel(bundle: Phase7BPipelineBundle, base: np.ndarray) -> np.ndarray:
    structural = bundle.handoff.structural_mask
    occupancy = bundle.occupancy.occupancy_mask
    p1 = _blend(base, structural, (50, 220, 70), 0.45)
    p2 = _blend(base, structural, (80, 120, 80), 0.25)
    p2 = _blend(p2, occupancy, (35, 45, 240), 0.70)

    p3 = _blend(base, structural, (120, 120, 120), 0.25)
    for y in bundle.rows.eligible_rows:
        if y in bundle.rows.structural_spans:
            left, right = bundle.rows.structural_spans[y]
            cv2.line(p3, (left, y), (right, y), (50, 235, 70), 1)
    exclusion_colors = {
        "undersized": (0, 90, 255),
        "left_frame_clipped": (0, 0, 255),
        "right_frame_clipped": (0, 0, 255),
        "unresolved_boundary_support": (210, 80, 210),
        "out_of_scope": (255, 160, 20),
    }
    for reason, ys in bundle.rows.excluded_rows.items():
        color = exclusion_colors.get(reason, (0, 0, 255))
        for y in ys:
            cv2.line(p3, (0, y), (p3.shape[1] - 1, y), color, 1)

    p4 = _blend(base, structural, (100, 130, 100), 0.22)
    for y, intervals in bundle.topology.row_intervals.items():
        for left, right in intervals:
            cv2.line(p4, (left, y), (right, y), (200, 130, 40), 1)
    for row in bundle.metrics.selected_corridor_path:
        left, right = row["interval"]
        cv2.line(p4, (left, row["y"]), (right, row["y"]), (40, 240, 240), 2)

    p5 = p4.copy()
    bottleneck = bundle.metrics.bottleneck
    if bottleneck and bottleneck.get("row_band"):
        for y in bottleneck["row_band"]:
            if y is not None:
                cv2.line(p5, (0, y), (p5.shape[1] - 1, y), (0, 210, 255), 2)
        raw_y = bottleneck.get("raw_minimum_row")
        if raw_y is not None:
            cv2.line(p5, (0, raw_y), (p5.shape[1] - 1, raw_y), (255, 255, 0), 1)

    p6 = _output_contract_panel(bundle, base)

    panels = [
        _headed(p1, "1 Phase7-A structural domain"),
        _headed(p2, "2 measurement occupancy (red)"),
        _headed(p3, "3 eligible green / excluded rows"),
        _headed(p4, "4 all intervals blue / selected yellow"),
        _headed(p5, "5 robust band orange / raw row cyan"),
        _headed_korean(p6, "6 통행 가능 폭 요약 · 검토 계약"),
    ]
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


def build_focused_review(
    output: Path,
    *,
    real_sample_ids: Sequence[str] = DEFAULT_REAL_CONTROLS,
) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite focused review: {output}")
    panel_dir = output / "panels"
    panel_dir.mkdir(parents=True)
    adapter = Phase7AAdapter()
    evidence = ExistingOccupancyEvidenceIndex()
    controls: list[tuple[str, str, Phase7BPipelineBundle, np.ndarray]] = []
    for sample_id in real_sample_ids:
        bundle = run_phase7_b_sample(
            sample_id, adapter=adapter, evidence_index=evidence
        )
        controls.append((sample_id, "real", bundle, _source_image(bundle)))
    synthetic_kinds = (
        "central_split",
        "staggered_no_teleport",
        "one_row_notch",
        "sustained_narrowing",
        "full_closure",
        "split_merge_best_right",
    )
    for kind in synthetic_kinds:
        sample_id = f"SYNTH_{kind}"
        bundle, base = _synthetic_bundle(sample_id, kind)
        controls.append((sample_id, "synthetic", bundle, base))

    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    cards: list[str] = []
    for index, (sample_id, control_type, bundle, base) in enumerate(controls, 1):
        validate_phase7_b_record(bundle.record)
        records.append(bundle.record)
        panel_path = panel_dir / f"{sample_id}.png"
        if not cv2.imwrite(str(panel_path), _render_panel(bundle, base)):
            raise OSError(panel_path)
        manifest.append(
            {
                "sample_id": sample_id,
                "control_type": control_type,
                "panel": str(panel_path.relative_to(output)).replace("\\", "/"),
                "panel_sha256": _file_hash(panel_path),
            }
        )
        record = bundle.record
        cards.append(
            f'''<section class="card" id="c{index}"><h2>{index}. {html.escape(sample_id)} <span>{control_type}</span></h2>
<p>Phase7-A={html.escape(record['phase7a_status'])} · scope={record['measurement_scope']} · confidence={record['measurement_confidence']} · state={record['passability_state']} · robust={record['minimum_passable_ratio']} · raw={record['raw_minimum_passable_ratio']}</p>
<img src="panels/{html.escape(sample_id)}.png" alt="{html.escape(sample_id)} Phase 7-B focused audit panels"></section>'''
        )

    results_path = output / "results.json"
    results_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = output / "focused_controls.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['measurement_scope']}:{record['passability_state']}"
        counts[key] = counts.get(key, 0) + 1
    review = output / "review.html"
    review.write_text(
        f'''<!doctype html><html><head><meta charset="utf-8"><title>Phase 7-B focused integration gate</title><style>
body{{margin:0;background:#0e2034;color:#edf4ff;font-family:Arial,sans-serif}}header{{padding:24px;position:sticky;top:0;background:#13283fdd;backdrop-filter:blur(8px);z-index:2}}h1{{margin:0 0 8px}}header p{{margin:6px 0}}nav a{{color:#7ee7d1;margin-right:12px}}.card{{margin:18px;border:1px solid #31506f;border-radius:12px;overflow:hidden;background:#142a42}}.card h2,.card p{{padding:0 16px}}.card h2 span{{font-size:14px;color:#77e5cf}}img{{display:block;width:100%;height:auto}}code{{color:#ffd479}}
</style></head><body><header><h1>Phase 7-B focused integration gate</h1><p>Canonical Steps 1–6 integration only. Physical widths are intentionally unsupplied, so absolute columns remain <code>null</code>. No full-160 regression has run.</p><p>Panels: structural domain · measurement occupancy · row eligibility · connected path · raw/robust bottleneck · output contract.</p><p>records={len(records)} · real={len(real_sample_ids)} · synthetic={len(synthetic_kinds)} · counts={html.escape(str(counts))}</p><nav>{''.join(f'<a href="#c{i}">{html.escape(sample_id)}</a>' for i,(sample_id,_,_,_) in enumerate(controls,1))}</nav></header>{''.join(cards)}</body></html>''',
        encoding="utf-8",
    )
    validation = {
        "schema_version": "phase7_b_focused_validation_v1",
        "record_count": len(records),
        "real_count": len(real_sample_ids),
        "synthetic_count": len(synthetic_kinds),
        "all_records_valid": True,
        "all_default_physical_inputs_null": all(
            record["surface_width_m"] is None
            and record["agent_width_m"] is None
            and record["safety_factor"] is None
            and record["absolute_passable_width_m"] is None
            and record["effective_blocked_width_m"] is None
            for record in records
        ),
        "results_sha256": _file_hash(results_path),
        "review_sha256": _file_hash(review),
        "focused_controls_sha256": _file_hash(manifest_path),
        "counts": counts,
    }
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return review


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append")
    arguments = parser.parse_args()
    print(
        build_focused_review(
            arguments.output,
            real_sample_ids=(
                tuple(dict.fromkeys(arguments.sample_id))
                if arguments.sample_id
                else DEFAULT_REAL_CONTROLS
            ),
        ).resolve()
    )
