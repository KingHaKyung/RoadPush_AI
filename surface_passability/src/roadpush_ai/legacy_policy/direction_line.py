from __future__ import annotations

import csv
import heapq
import html
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pilot_8_v2" / "artifacts"
OUT = HERE / "direction_line_probe_8_v2"
IDS = [
    "IMG_4697", "IMG_4364", "MP_SEL_SUR_006472", "kr_bicycle_01",
    "commons_01", "IMG_4282", "kr_case_candidate_04", "kr_bollard_01",
]
ANALYSIS_MAX = 700


def fit(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, ANALYSIS_MAX / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    # Padding is essential when a sidewalk touches an image edge. Without it,
    # OpenCV's morphology treats that open edge poorly and creates edge lines.
    pad = max(8, round(min(mask.shape) * 0.02))
    work = np.pad(mask.astype(np.uint8) * 255, pad)
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton[pad:-pad, pad:-pad] > 0


def preprocess(mask: np.ndarray) -> np.ndarray:
    # Only suppress tiny jagged gaps. Do not fill semantic object holes wholesale.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def choose_start(mask: np.ndarray, distance: np.ndarray, height_fraction: float) -> tuple[int, int] | None:
    h, w = mask.shape
    center_x = (w - 1) / 2
    # A narrow gate near the viewer replaces the old full bottom-15% start region.
    preferred_y = min(h - 2, round(h * height_fraction))
    for offset in range(0, max(3, round(h * 0.05))):
        for y in ([preferred_y - offset] if offset == 0 else [preferred_y - offset, preferred_y + offset]):
            if y < round(h * 0.70) or y >= h - 1:
                continue
            xs = np.flatnonzero(mask[y])
            if not len(xs):
                continue
            central = xs[(xs >= round(w * 0.25)) & (xs <= round(w * 0.75))]
            candidates = central if len(central) else xs
            score = distance[y, candidates] - 0.08 * np.abs(candidates - center_x)
            x = int(candidates[int(np.argmax(score))])
            return y, x
    return None


def component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask)
    label = int(labels[seed])
    return labels == label if label else np.zeros_like(mask)


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(row.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(a), int(b - 1)) for a, b in zip(starts, ends)]


def choose_goals(component: np.ndarray, distance: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    h, w = component.shape
    ys = np.flatnonzero(component.any(axis=1))
    if not len(ys):
        return 0, []
    top, bottom = int(ys[0]), int(ys[-1])
    target_y = min(bottom, top + max(4, round((bottom - top) * 0.06)))
    while target_y <= bottom and not component[target_y].any():
        target_y += 1
    candidates = []
    for left, right in runs(component[target_y]):
        if right - left + 1 < 3:
            continue
        xs = np.arange(left, right + 1)
        local = distance[target_y, xs]
        best_value = float(local.max())
        tied = xs[np.isclose(local, best_value)]
        x = int(tied[np.argmin(np.abs(tied - (w - 1) / 2))])
        candidates.append((target_y, x))
    # Retain at most five genuine exits, preferring wider/safer points.
    candidates.sort(key=lambda p: float(distance[p]), reverse=True)
    return target_y, candidates[:5]


def shortest_paths(
    component: np.ndarray,
    distance: np.ndarray,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    h, w = component.shape
    costs = np.full((h, w), np.inf, np.float64)
    previous = np.full((h, w, 2), -1, np.int32)
    costs[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    remaining = set(goals)
    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while queue and remaining:
        current, y, x = heapq.heappop(queue)
        if current != costs[y, x]:
            continue
        remaining.discard((y, x))
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not component[ny, nx]:
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            edge_penalty = 1.0 + 12.0 / (float(distance[ny, nx]) + 1.0)
            direction_penalty = 3.0 if dy > 0 else (1.08 if dy == 0 else 1.0)
            proposed = current + step * edge_penalty * direction_penalty
            if proposed < costs[ny, nx]:
                costs[ny, nx] = proposed
                previous[ny, nx] = (y, x)
                heapq.heappush(queue, (proposed, ny, nx))
    paths, totals = [], []
    for goal in goals:
        if not np.isfinite(costs[goal]):
            continue
        path = [goal]
        cursor = goal
        while cursor != start:
            py, px = map(int, previous[cursor])
            if py < 0:
                path = []
                break
            cursor = (py, px)
            path.append(cursor)
        if path:
            path.reverse()
            paths.append(path)
            totals.append(float(costs[goal]))
    return paths, totals


def smooth_path(path: list[tuple[int, int]], component: np.ndarray) -> list[tuple[int, int]]:
    if len(path) < 9:
        return path
    points = np.asarray(path, np.float32)
    window = max(5, min(31, (len(path) // 18) | 1))
    kernel = np.ones(window, np.float32) / window
    ys = np.convolve(points[:, 0], kernel, mode="same")
    xs = np.convolve(points[:, 1], kernel, mode="same")
    half = window // 2
    ys[:half], xs[:half] = points[:half, 0], points[:half, 1]
    ys[-half:], xs[-half:] = points[-half:, 0], points[-half:, 1]
    result = []
    for original, y, x in zip(path, ys, xs):
        candidate = (int(round(y)), int(round(x)))
        result.append(candidate if component[candidate] else original)
    return result


def solve_attempt(clean: np.ndarray, distance: np.ndarray, start_fraction: float) -> dict[str, object]:
    start = choose_start(clean, distance, start_fraction)
    component = component_from_seed(clean, start) if start else np.zeros_like(clean)
    target_y, goals = choose_goals(component, distance) if start else (0, [])
    paths, totals = shortest_paths(component, distance, start, goals) if start and goals else ([], [])
    paths = [smooth_path(path, component) for path in paths]
    main_index = int(np.argmin([cost / max(1, len(path)) for cost, path in zip(totals, paths)])) if paths else None
    angle = None
    vertical_progress = 0.0
    if main_index is not None and len(paths[main_index]) >= 2:
        p0, p1 = paths[main_index][0], paths[main_index][-1]
        vertical_progress = float((p0[0] - p1[0]) / clean.shape[0])
        angle = math.degrees(math.atan2(p1[1] - p0[1], max(1, p0[0] - p1[0])))
    high_confidence = bool(
        main_index is not None
        and vertical_progress >= 0.20
        and angle is not None
        and abs(angle) <= 60.0
    )
    quality = vertical_progress - (abs(angle) / 360.0 if angle is not None else 1.0)
    return {
        "requested_start_fraction": start_fraction,
        "start": start, "component": component, "target_y": target_y, "goals": goals,
        "paths": paths, "totals": totals, "main_index": main_index,
        "angle": angle, "vertical_progress": vertical_progress,
        "high_confidence": high_confidence, "quality": quality,
    }


def paint_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1 - alpha) + np.asarray(color) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_path(image: np.ndarray, path: list[tuple[int, int]], color: tuple[int, int, int], width: int = 4) -> None:
    if len(path) < 2:
        return
    points = np.asarray([(x, y) for y, x in path], np.int32)
    cv2.polylines(image, [points], False, color, width, cv2.LINE_AA)


def process(sample_id: str) -> dict[str, object]:
    source = SOURCE / sample_id
    photo = np.asarray(Image.open(source / "01_original.jpg").convert("RGB"))
    native = cv2.imread(str(source / "05_repaired_surface.png"), cv2.IMREAD_GRAYSCALE) > 0
    size = fit(photo.shape[1], photo.shape[0])
    image = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
    resized = cv2.resize(native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    clean = preprocess(resized)
    padded = np.pad(clean.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    raw_skeleton = morphological_skeleton(resized)
    clean_skeleton = morphological_skeleton(clean)
    primary_attempt = solve_attempt(clean, distance, 0.94)
    # Fallback is exclusively for a missing start seed. A weak route at 94%
    # remains weak; it must not be made to pass by moving the seed upward.
    selected_attempt = (
        solve_attempt(clean, distance, 0.88)
        if primary_attempt["start"] is None
        else primary_attempt
    )
    start = selected_attempt["start"]
    component = selected_attempt["component"]
    target_y = selected_attempt["target_y"]
    goals = selected_attempt["goals"]
    paths = selected_attempt["paths"]
    main_index = selected_attempt["main_index"]
    high_confidence = bool(selected_attempt["high_confidence"])
    fallback_used = bool(selected_attempt["requested_start_fraction"] == 0.88)

    base_view = paint_mask(image, resized, (45, 210, 95), 0.42)
    raw_view = base_view.copy()
    raw_view[raw_skeleton] = (245, 70, 70)
    clean_view = paint_mask(image, clean, (45, 210, 95), 0.42)
    clean_view[clean_skeleton] = (245, 70, 70)
    seed_view = base_view.copy()
    if start:
        cv2.circle(seed_view, (start[1], start[0]), 9, (20, 220, 255), -1, cv2.LINE_AA)
        cv2.line(seed_view, (0, start[0]), (size[0] - 1, start[0]), (20, 220, 255), 1, cv2.LINE_AA)
    for goal in goals:
        cv2.circle(seed_view, (goal[1], goal[0]), 8, (255, 80, 220), -1, cv2.LINE_AA)
    all_view = base_view.copy()
    for index, path in enumerate(paths):
        selected_color = (45, 120, 255) if high_confidence else (135, 135, 135)
        draw_path(all_view, path, (255, 150, 25) if index != main_index else selected_color, 5)
    main_view = base_view.copy()
    for index, path in enumerate(paths):
        draw_path(main_view, path, (255, 165, 30), 3)
    if main_index is not None:
        selected_color = (40, 110, 255) if high_confidence else (135, 135, 135)
        draw_path(main_view, paths[main_index], selected_color, 6)
        cv2.circle(main_view, (start[1], start[0]), 8, (20, 220, 255), -1, cv2.LINE_AA)

    direction_angle = selected_attempt["angle"]
    vertical_progress = float(selected_attempt["vertical_progress"])
    status = "high_confidence_candidate" if high_confidence else "low_confidence_or_undetermined"

    sample_out = OUT / "artifacts" / sample_id
    sample_out.mkdir(parents=True)
    cv2.imwrite(str(sample_out / "01_clean_surface.png"), clean.astype(np.uint8) * 255)
    cv2.imwrite(str(sample_out / "02_raw_skeleton.png"), raw_skeleton.astype(np.uint8) * 255)
    cv2.imwrite(str(sample_out / "03_padded_clean_skeleton.png"), clean_skeleton.astype(np.uint8) * 255)
    cv2.imwrite(str(sample_out / "04_reachable_component.png"), component.astype(np.uint8) * 255)
    cv2.imwrite(str(sample_out / "05_direction_candidates.jpg"), cv2.cvtColor(main_view, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    titles = [
        "1 Repaired surface", "2 Previous-style raw skeleton", "3 Padded + lightly cleaned skeleton",
        "4 Narrow start gate + distant goals", "5 Center-seeking route candidates", "6 Selected direction candidate",
    ]
    panels = [base_view, raw_view, clean_view, seed_view, all_view, main_view]
    for ax, title, panel in zip(axes.flat, titles, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{sample_id} | direction-line probe | non-destructive", fontsize=16)
    fig.savefig(sample_out / "summary.png", dpi=140)
    plt.close(fig)

    payload = {
        "sample_id": sample_id,
        "analysis_width": size[0], "analysis_height": size[1],
        "start_yx": list(start) if start else None,
        "requested_start_fraction": selected_attempt["requested_start_fraction"],
        "start_gate_fraction_from_top": (start[0] / size[1]) if start else None,
        "fallback_used": fallback_used,
        "target_row": target_y, "goals_yx": [list(goal) for goal in goals],
        "candidate_count": len(paths), "selected_candidate_index": main_index,
        "direction_angle_degrees_from_image_up": direction_angle,
        "vertical_progress_fraction": vertical_progress,
        "confidence_rules": {"minimum_vertical_progress": 0.20, "maximum_absolute_angle_degrees": 60.0},
        "status": status,
        "destructive_mask_edit": False,
        "notes": [
            "The primary cyan start seed is near 94% image height; 88% is evaluated as fallback when needed.",
            "Blue is a high-confidence center-seeking route; gray is low-confidence; orange paths are alternatives.",
            "No sidewalk pixels are deleted by this probe.",
        ],
    }
    (sample_out / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    (OUT / "artifacts").mkdir(parents=True)
    rows = [process(sample_id) for sample_id in IDS]
    with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "sample_id", "status", "candidate_count", "direction_angle_degrees_from_image_up",
            "vertical_progress_fraction", "requested_start_fraction", "start_gate_fraction_from_top",
            "fallback_used", "destructive_mask_edit",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    cards = []
    for row in rows:
        sample_id = str(row["sample_id"])
        angle = row["direction_angle_degrees_from_image_up"]
        angle_text = "n/a" if angle is None else f"{angle:.1f}°"
        cards.append(
            f'<section><h2>{html.escape(sample_id)}</h2>'
            f'<p>Status {row["status"]} · candidates {row["candidate_count"]} · angle from image-up {angle_text} · '
            f'vertical progress {row["vertical_progress_fraction"]:.1%} · fallback {row["fallback_used"]} · no deletion</p>'
            f'<img src="artifacts/{html.escape(sample_id)}/summary.png"></section>'
        )
    review = (
        '<!doctype html><meta charset="utf-8"><title>Direction-line probe 8 v2</title>'
        '<style>body{font:16px system-ui;background:#10151d;color:#eef;max-width:1500px;margin:auto}'
        'section{background:#19212c;margin:24px;padding:18px;border-radius:12px}img{width:100%;height:auto}'
        'small{color:#8ff}</style><h1>Direction-line probe — 8 images</h1>'
        '<p>Cached repaired surfaces only. Cyan=start seed, magenta=goal, blue=high-confidence candidate, '
        'gray=low-confidence, orange=alternatives. '
        'This probe never deletes sidewalk pixels.</p>' + "".join(cards)
    )
    (OUT / "review.html").write_text(review, encoding="utf-8")
    report = """# Direction-line probe 8 v2

- Input: cached V2 repaired surfaces; no model inference
- Output is separate and non-destructive
- Old full bottom-15% start region replaced by one narrow seed near 94% image height
- 88% fallback is used only when no start seed exists near 94%; low confidence never triggers fallback
- Black padding added before morphological skeletonization
- Route candidates are solved on the connected surface with an edge-distance penalty and upward-motion preference
- Blue requires at least 20% vertical progress and no more than 60 degrees from image-up
- Gray is a retained low-confidence route; orange lines are alternatives
- This is image-space directional evidence, not a final mask-deletion rule or metric navigation result
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "samples": len(rows), "statuses": {r['sample_id']: r['status'] for r in rows}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
