from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import html
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = get_settings().project_root
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "model_audit_40" / "audit_manifest.json"
OUT = HERE / "outer_boundary_width_probe_8_r1"
IDS = [
    "IMG_4697", "IMG_4364", "MP_SEL_SUR_006472", "kr_bicycle_01",
    "commons_01", "IMG_4282", "kr_case_candidate_04", "kr_bollard_01",
]
MAX_SIDE = 700



from roadpush_ai.legacy_policy import boundary_width as base
v2 = base.v2
direction = base.direction


def fit(w: int, h: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def route_path(repaired: np.ndarray) -> tuple[dict, list[tuple[int, int]]]:
    attempt = base.solve_route(repaired)
    index = attempt["main_index"]
    return attempt, attempt["paths"][index] if index is not None else []


def selected_component(structural: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    seed = path[0] if path else None
    if seed is None or not structural[seed]:
        padded = np.pad(structural.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        seed = direction.choose_start(structural, distance, 0.94)
        if seed is None:
            seed = direction.choose_start(structural, distance, 0.88)
    return direction.component_from_seed(structural, seed) if seed else np.zeros_like(structural)


def path_x(path: list[tuple[int, int]], y: int, fallback: float) -> float:
    if not path:
        return fallback
    points = np.asarray(path, np.float32)
    return float(points[int(np.argmin(np.abs(points[:, 0] - y))), 1])


def sample_outer_boundaries(component: np.ndarray, path: list[tuple[int, int]], labels: np.ndarray) -> list[dict]:
    h, w = component.shape
    ys = np.flatnonzero(component.any(axis=1))
    if not len(ys):
        return []
    if path:
        bottom, top = max(y for y, _ in path), min(y for y, _ in path)
        top = min(top, int(ys[0] + 0.12 * (bottom - ys[0])))
    else:
        bottom, top = min(h - 1, round(h * 0.94)), max(int(ys[0]), round(h * 0.45))
    if bottom - top < round(h * 0.10):
        return []
    levels = np.unique(np.linspace(bottom, top, 18).round().astype(int))[::-1]
    band = max(2, round(h * 0.007))
    records = []
    for y in levels:
        y0, y1 = max(0, y - band), min(h, y + band + 1)
        row = component[y0:y1].sum(axis=0) >= max(1, round((y1 - y0) * 0.45))
        xs = np.flatnonzero(row)
        if not len(xs):
            continue
        left, right = int(xs[0]), int(xs[-1])
        left_kind = base.endpoint_kind(labels, int(y), left, "left")
        right_kind = base.endpoint_kind(labels, int(y), right, "right")
        if left_kind not in ("frame_truncated", "obstacle_occlusion") and right_kind not in ("frame_truncated", "obstacle_occlusion"):
            state = "complete"
        elif left_kind == "frame_truncated" or right_kind == "frame_truncated":
            state = "lower_bound"
        else:
            state = "occlusion_limited"
        records.append({
            "y": int(y), "left": left, "right": right, "width": right - left + 1,
            "anchor_x": path_x(path, int(y), w / 2), "left_kind": left_kind,
            "right_kind": right_kind, "width_state": state,
        })
    return records


def clipped_free_segments(free: np.ndarray, record: dict) -> list[tuple[int, int]]:
    return base.free_segments(free, record["y"], record["left"], record["right"])


def choose_route_segment(segments: list[tuple[int, int]], anchor: float) -> tuple[int, int] | None:
    containing = [segment for segment in segments if segment[0] <= anchor <= segment[1]]
    if containing:
        return max(containing, key=lambda p: p[1] - p[0])
    return max(segments, key=lambda p: p[1] - p[0]) if segments else None


def draw_path(image: np.ndarray, path: list[tuple[int, int]], color, width=4):
    if len(path) >= 2:
        points = np.asarray([(x, y) for y, x in path], np.int32)
        cv2.polylines(image, [points], False, color, width, cv2.LINE_AA)


def process(item: dict) -> dict:
    sid = item["sample_id"]
    with Image.open(ROOT / item["relative_path"]) as source:
        size = fit(source.width, source.height)
        image = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    native_labels = cv2.imread(str(ROOT / item["class_map_path"]), cv2.IMREAD_UNCHANGED)
    if native_labels is None:
        raise FileNotFoundError(item["class_map_path"])
    repaired_native, _, _ = v2.repair_surface(native_labels)
    labels = cv2.resize(native_labels.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    repaired = cv2.resize(repaired_native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    blocked_ids = tuple(v2.OCCUPANCY) + tuple(v2.OCCLUSION) + tuple(v2.HAZARD)
    blocked = np.isin(labels, blocked_ids)
    structural, restored = base.obstacle_transparent_structure(repaired, blocked)
    free = structural & ~blocked
    route, path = route_path(repaired)
    component = selected_component(structural, path)
    records = sample_outer_boundaries(component, path, labels)

    for record in records:
        segments = clipped_free_segments(free, record)
        selected = choose_route_segment(segments, record["anchor_x"])
        record["free_segments"] = [[a, b] for a, b in segments]
        record["selected_free_segment"] = list(selected) if selected else None
        record["selected_free_width"] = selected[1] - selected[0] + 1 if selected else 0
        record["widest_free_width"] = max((b - a + 1 for a, b in segments), default=0)

    complete = [r for r in records if r["width_state"] == "complete"]
    lower = [r for r in records if r["width_state"] == "lower_bound"]
    occluded = [r for r in records if r["width_state"] == "occlusion_limited"]
    status = "complete_width_candidate" if len(complete) >= 6 else (
        "partial_width_only" if lower or occluded else "measurement_unavailable"
    )

    full_view = paint(image, repaired, (45, 210, 95), 0.42)
    structure_view = paint(full_view, restored, (255, 145, 35), 0.82)
    boundary_view = structure_view.copy()
    contour_view = full_view.copy()
    width_view = full_view.copy()
    free_view = paint(image, free, (85, 185, 235), 0.28)

    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contour_view, contours, -1, (255, 65, 210), 4, cv2.LINE_AA)
    draw_path(contour_view, path, (35, 110, 255) if route["high_confidence"] else (125, 125, 125), 4)
    state_colors = {"complete": (40, 215, 255), "lower_bound": (155, 155, 155), "occlusion_limited": (255, 145, 35)}
    endpoint_colors = {"physical_boundary": (255, 65, 210), "semantic_boundary": (255, 65, 210),
                       "obstacle_occlusion": (255, 145, 35), "frame_truncated": (155, 155, 155)}
    for record in records:
        y, left, right = record["y"], record["left"], record["right"]
        cv2.line(boundary_view, (left, y), (right, y), (235, 235, 235), 1, cv2.LINE_AA)
        cv2.circle(boundary_view, (left, y), 4, endpoint_colors[record["left_kind"]], -1)
        cv2.circle(boundary_view, (right, y), 4, endpoint_colors[record["right_kind"]], -1)
        cv2.line(width_view, (left, y), (right, y), state_colors[record["width_state"]], 3, cv2.LINE_AA)
        for a, b in record["free_segments"]:
            cv2.line(free_view, (a, y), (b, y), (90, 205, 255), 2, cv2.LINE_AA)
        if record["selected_free_segment"]:
            a, b = record["selected_free_segment"]
            cv2.line(free_view, (a, y), (b, y), (35, 85, 255), 5, cv2.LINE_AA)

    art = OUT / "artifacts" / sid
    art.mkdir(parents=True)
    cv2.imwrite(str(art / "01_full_sidewalk.png"), repaired.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02_structural_surface.png"), structural.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "03_restored_object_footprints.png"), restored.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "04_free_space.png"), free.astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [full_view, structure_view, boundary_view, contour_view, width_view, free_view]
    titles = [
        "1 Full sidewalk (preserved)", "2 Structural surface (orange=temporary restoration)",
        "3 Raw outer endpoints at each row", "4 Actual outer contour + blue route",
        "5 Width states (cyan=complete, gray=lower, orange=occluded)",
        "6 Free gaps (light blue=all, dark blue=route-selected)",
    ]
    for ax, title, panel in zip(axes.flat, titles, panels):
        ax.imshow(panel); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{sid} | outer-boundary width probe | {status} | no branch removal", fontsize=16)
    fig.savefig(art / "summary.png", dpi=115)
    plt.close(fig)

    structural_widths = [r["width"] for r in complete]
    selected_widths = [r["selected_free_width"] for r in records if r["selected_free_width"]]
    payload = {
        "sample_id": sid, "order": item["order"], "split": item["split"], "status": status,
        "sampled_rows": len(records), "complete_rows": len(complete), "lower_bound_rows": len(lower),
        "occlusion_limited_rows": len(occluded),
        "median_complete_structural_width_px": float(np.median(structural_widths)) if structural_widths else None,
        "median_route_selected_free_width_px": float(np.median(selected_widths)) if selected_widths else None,
        "restored_object_pixels": int(restored.sum()), "rows": records,
        "branch_removal": False, "straight_boundary_fit": False, "new_model_inference": False,
        "destructive_mask_edit": False, "full_sidewalk_preserved": True,
        "warning": "Outermost connected sidewalk includes side branches by design; widths are image-space pixels, not meters.",
    }
    (art / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in manifest["samples"]}
    rows = [process(by_id[sid]) for sid in IDS]
    summary_fields = [key for key in rows[0] if key != "rows"]
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields); writer.writeheader()
        writer.writerows([{key: row[key] for key in summary_fields} for row in rows])
    counts = {status: sum(row["status"] == status for row in rows) for status in (
        "complete_width_candidate", "partial_width_only", "measurement_unavailable")}
    cards = []
    for row in rows:
        sid = html.escape(row["sample_id"])
        structural = "n/a" if row["median_complete_structural_width_px"] is None else f'{row["median_complete_structural_width_px"]:.1f}px'
        free = "n/a" if row["median_route_selected_free_width_px"] is None else f'{row["median_route_selected_free_width_px"]:.1f}px'
        cards.append(
            f'<section><h2>{row["order"]:02d}. {sid} <small>{row["split"]}</small></h2>'
            f'<p>{row["status"]} · complete {row["complete_rows"]} · lower {row["lower_bound_rows"]} · '
            f'occluded {row["occlusion_limited_rows"]} · structure median {structural} · selected free median {free}</p>'
            f'<a href="artifacts/{sid}/summary.png"><img src="artifacts/{sid}/summary.png"></a></section>'
        )
    review = (
        '<!doctype html><meta charset="utf-8"><title>Outer-boundary width probe 8 r1</title>'
        '<style>body{font:16px system-ui;background:#10151d;color:#eef;max-width:1500px;margin:auto;padding:18px}'
        'section{background:#19212c;margin:24px 0;padding:18px;border-radius:12px}img{width:100%;height:auto}small{color:#8ff}</style>'
        '<h1>Raw outer-boundary + obstacle-transparent width probe — 8 images</h1>'
        '<p>No straight boundary fit and no branch removal. The actual outer contour of the connected structural surface is used as-is. '
        'Objects are restored only for structural width and subtracted again for free gaps.</p>'
        f'<p>Complete {counts["complete_width_candidate"]} · Partial {counts["partial_width_only"]} · Unavailable {counts["measurement_unavailable"]}</p>'
        + ''.join(cards)
    )
    (OUT / "review.html").write_text(review, encoding="utf-8")
    validation = {
        "expected_samples": 8, "processed_samples": len(rows),
        "all_artifacts_present": all((OUT / "artifacts" / row["sample_id"] / "summary.png").exists() for row in rows),
        "new_model_inference": False, "existing_outputs_overwritten": False,
        "straight_boundary_fit": False, "branch_removal": False,
        "destructive_mask_edit": False, "full_sidewalk_preserved": True, "status_counts": counts,
    }
    (OUT / "completion_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text(
        "# Outer-boundary width probe 8 r1\n\n"
        "- Cached semantic maps only; no new model inference\n"
        "- No straight-line or polynomial boundary fitting\n"
        "- No branch detection or removal\n"
        "- Uses the actual outer edge of the selected connected structural surface\n"
        "- Internal obstacle footprints may be restored only in the structural copy\n"
        "- Free gaps are calculated after obstacles are subtracted again\n"
        "- Frame-truncated widths are lower bounds, never complete widths\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "samples": len(rows), "status_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
