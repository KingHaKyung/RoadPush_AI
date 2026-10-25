from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import html
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = get_settings().project_root
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "model_audit_40" / "audit_manifest.json"
OUT = HERE / "boundary_branch_probe_8_v5"
IDS = [
    "IMG_4697", "IMG_4364", "MP_SEL_SUR_006472", "kr_bicycle_01",
    "commons_01", "IMG_4282", "kr_case_candidate_04", "kr_bollard_01",
]
MAX_SIDE = 700



from roadpush_ai.legacy_policy import base_surface as v2
from roadpush_ai.legacy_policy import direction_line as direction


def fit(w: int, h: int) -> tuple[int, int]:
    scale = min(1.0, MAX_SIDE / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def structure_copy(repaired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = repaired.shape
    k = max(5, round(min(h, w) * 0.014)) | 1
    closed = cv2.morphologyEx(
        repaired.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    ).astype(bool)
    inverse = (~closed).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    result = closed.copy()
    limit = max(16, round(h * w * 0.012))
    for label in range(1, count):
        x, y, ww, hh, area = map(int, stats[label])
        border = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not border and area <= limit:
            result[labels == label] = True
    return result, result & ~repaired


def obstacle_transparent_structure(repaired: np.ndarray, blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restore only obstacle runs bracketed by sidewalk on both horizontal sides.

    This makes poles/bollards/bicycles transparent for structural-width analysis,
    while a large car that reaches the frame or has no sidewalk visible on its
    far side remains an occlusion rather than invented sidewalk.
    """
    structure, base_fill = structure_copy(repaired)
    h, w = repaired.shape
    additions = base_fill.copy()
    max_bridge = max(10, round(w * 0.18))
    blocked_u8 = blocked.astype(np.uint8)
    for y in range(h):
        for left, right in direction.runs(blocked_u8[y] > 0):
            if right - left + 1 > max_bridge:
                continue
            left_window = structure[y, max(0, left - max_bridge):left]
            right_window = structure[y, right + 1:min(w, right + 1 + max_bridge)]
            if left_window.any() and right_window.any():
                structure[y, left:right + 1] = True
                additions[y, left:right + 1] = True
    # Join row-wise restorations into a stable geometry copy without changing the source mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    structure = cv2.morphologyEx(structure.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    additions |= structure & ~repaired
    return structure, additions


def solve_route(repaired: np.ndarray) -> dict:
    clean = direction.preprocess(repaired)
    padded = np.pad(clean.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    attempt = direction.solve_attempt(clean, distance, 0.94)
    if attempt["start"] is None:
        attempt = direction.solve_attempt(clean, distance, 0.88)
    return attempt


def route_x_at(path: list[tuple[int, int]], y: int, default_x: float) -> float:
    if not path:
        return default_x
    points = np.asarray(path, np.float32)
    index = int(np.argmin(np.abs(points[:, 0] - y)))
    return float(points[index, 1])


def choose_run(row: np.ndarray, anchor_x: float, tolerance: int) -> tuple[int, int] | None:
    choices = direction.runs(row)
    if not choices:
        return None
    containing = [r for r in choices if r[0] - tolerance <= anchor_x <= r[1] + tolerance]
    pool = containing if containing else choices
    return min(pool, key=lambda r: abs((r[0] + r[1]) / 2 - anchor_x))


def boundary_support(gray: np.ndarray, labels: np.ndarray, y: int, x: int) -> dict:
    h, w = gray.shape
    y0, y1 = max(0, y - 5), min(h, y + 6)
    x0, x1 = max(0, x - 4), min(w, x + 5)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    patch = np.abs(sobel[y0:y1, x0:x1])
    gradient = float(np.percentile(patch, 80) / 255.0) if patch.size else 0.0
    context = labels[y0:y1, x0:x1]
    curb = float(np.isin(context, (2, 9)).mean()) if context.size else 0.0
    return {"gradient": min(1.0, gradient), "curb": curb}


def endpoint_kind(labels: np.ndarray, y: int, x: int, side: str) -> str:
    h, w = labels.shape
    if x <= 1 or x >= w - 2:
        return "frame_truncated"
    pad_y = max(3, round(h * 0.006))
    pad_x = max(3, round(w * 0.006))
    if side == "left":
        x0, x1 = max(0, x - pad_x), min(w, x + 2)
    else:
        x0, x1 = max(0, x - 1), min(w, x + pad_x + 1)
    y0, y1 = max(0, y - pad_y), min(h, y + pad_y + 1)
    context = labels[y0:y1, x0:x1]
    blocked_ids = tuple(v2.OCCUPANCY) + tuple(v2.OCCLUSION) + tuple(v2.HAZARD)
    if context.size and np.isin(context, blocked_ids).mean() >= 0.08:
        return "obstacle_occlusion"
    if context.size and np.isin(context, (2, 9, 10, 13, 14, 16, 17, 18, 25, 29, 30)).mean() >= 0.08:
        return "physical_boundary"
    return "semantic_boundary"


def collect_rows(structural: np.ndarray, route_path: list[tuple[int, int]], labels: np.ndarray,
                 gray: np.ndarray) -> list[dict]:
    h, w = structural.shape
    if route_path:
        bottom = int(max(y for y, _ in route_path))
        top = int(min(y for y, _ in route_path))
        # Boundary evidence may extend a little beyond a short route.
        component_ys = np.flatnonzero(structural.any(axis=1))
        if len(component_ys):
            top = min(top, int(component_ys[0] + 0.12 * (bottom - component_ys[0])))
    else:
        bottom, top = round(h * 0.94), round(h * 0.45)
    if bottom - top < round(h * 0.12):
        return []
    levels = np.unique(np.linspace(bottom, top, 18).round().astype(int))[::-1]
    band = max(2, round(h * 0.007))
    records = []
    anchor = w / 2
    for y in levels:
        anchor = route_x_at(route_path, int(y), anchor)
        y0, y1 = max(0, y - band), min(h, y + band + 1)
        row = structural[y0:y1].sum(axis=0) >= max(1, round((y1 - y0) * 0.50))
        selected = choose_run(row, anchor, max(3, round(w * 0.018)))
        if selected is None:
            continue
        left, right = selected
        ls = boundary_support(gray, labels, int(y), left)
        rs = boundary_support(gray, labels, int(y), right)
        records.append({
            "y": int(y), "left": left, "right": right, "width": right - left + 1,
            "anchor_x": anchor, "left_gradient": ls["gradient"], "right_gradient": rs["gradient"],
            "left_curb": ls["curb"], "right_curb": rs["curb"],
            "left_kind": endpoint_kind(labels, int(y), left, "left"),
            "right_kind": endpoint_kind(labels, int(y), right, "right"),
        })
    return records


def fit_independent_boundary(records: list[dict], side: str, image_width: int) -> dict:
    """Fit a conservative continuation from the far/main-corridor portion.

    The fit is never written into the source sidewalk mask. It is evidence for a
    derived main-corridor candidate only.
    """
    if len(records) < 5:
        return {"coeff": None, "predicted": [], "eligible": 0}
    key = side
    kind_key = f"{side}_kind"
    # records are ordered from the lower/near portion toward the upper/far portion.
    eligible = [i for i, r in enumerate(records) if r[kind_key] not in ("frame_truncated", "obstacle_occlusion")]
    if len(eligible) < 4:
        return {"coeff": None, "predicted": [], "eligible": len(eligible)}
    # Prefer the far half: it normally shows the corridor before a side branch fans out.
    far_limit = max(6, math.ceil(len(records) * 0.55))
    far = [i for i in eligible if i >= len(records) - far_limit]
    chosen = far if len(far) >= 4 else eligible
    y = np.asarray([records[i]["y"] for i in chosen], np.float64)
    x = np.asarray([records[i][key] for i in chosen], np.float64)
    degree = 1
    keep = np.ones(len(chosen), bool)
    for _ in range(3):
        coeff = np.polyfit(y[keep], x[keep], degree)
        residual = np.abs(x - np.polyval(coeff, y))
        threshold = max(5.0, image_width * 0.025, float(np.median(residual[keep]) * 2.8))
        updated = residual <= threshold
        if updated.sum() < 4 or np.array_equal(updated, keep):
            break
        keep = updated
    coeff = np.polyfit(y[keep], x[keep], degree)
    all_y = np.asarray([r["y"] for r in records], np.float64)
    predicted = np.polyval(coeff, all_y)
    return {"coeff": coeff.tolist(), "predicted": predicted.tolist(), "eligible": len(eligible)}


def branch_and_width_analysis(records: list[dict], image_width: int, blocked: np.ndarray) -> dict:
    left_model = fit_independent_boundary(records, "left", image_width)
    right_model = fit_independent_boundary(records, "right", image_width)
    widths = np.asarray([r["width"] for r in records], np.float64) if records else np.zeros(0)
    threshold = max(10.0, float(np.median(widths) * 0.12)) if len(widths) else 10.0
    left_pred = left_model["predicted"]
    right_pred = right_model["predicted"]
    for i, r in enumerate(records):
        left_delta = (left_pred[i] - r["left"]) if left_pred else 0.0
        right_delta = (r["right"] - right_pred[i]) if right_pred else 0.0
        opposite_right_ok = (
            not right_pred or r["right_kind"] == "frame_truncated"
            or abs(r["right"] - right_pred[i]) <= threshold
        )
        opposite_left_ok = (
            not left_pred or r["left_kind"] == "frame_truncated"
            or abs(r["left"] - left_pred[i]) <= threshold
        )
        left_candidate = bool(left_pred and left_delta > threshold and opposite_right_ok)
        right_candidate = bool(right_pred and right_delta > threshold and opposite_left_ok)
        # A deviation next to an object is an occlusion/indentation, never branch evidence.
        y0, y1 = max(0, r["y"] - 8), min(blocked.shape[0], r["y"] + 9)
        if left_candidate:
            x0, x1 = max(0, r["left"] - 6), min(blocked.shape[1], int(round(left_pred[i])) + 7)
            if x1 > x0 and blocked[y0:y1, x0:x1].any():
                left_candidate = False
        if right_candidate:
            x0, x1 = max(0, int(round(right_pred[i])) - 6), min(blocked.shape[1], r["right"] + 7)
            if x1 > x0 and blocked[y0:y1, x0:x1].any():
                right_candidate = False
        r["left_branch_candidate"] = left_candidate
        r["right_branch_candidate"] = right_candidate
        r["main_left"] = int(round(left_pred[i])) if r["left_branch_candidate"] else r["left"]
        r["main_right"] = int(round(right_pred[i])) if r["right_branch_candidate"] else r["right"]
        r["main_left"] = max(r["left"], min(r["right"], r["main_left"]))
        r["main_right"] = min(r["right"], max(r["left"], r["main_right"]))
        left_complete = r["left_kind"] not in ("frame_truncated", "obstacle_occlusion")
        right_complete = r["right_kind"] not in ("frame_truncated", "obstacle_occlusion")
        if left_complete and right_complete:
            r["width_state"] = "complete"
        elif r["left_kind"] == "frame_truncated" or r["right_kind"] == "frame_truncated":
            r["width_state"] = "lower_bound"
        elif r["left_kind"] == "obstacle_occlusion" or r["right_kind"] == "obstacle_occlusion":
            r["width_state"] = "occlusion_limited"
        else:
            r["width_state"] = "unavailable"
        r["main_width"] = max(0, r["main_right"] - r["main_left"] + 1)
    return {"left_model": left_model, "right_model": right_model, "branch_threshold_px": threshold}


def persistent_pair(records: list[dict], image_width: int) -> tuple[np.ndarray, dict]:
    n = len(records)
    if n < 4:
        return np.zeros(n, bool), {"left_coeff": None, "right_coeff": None, "residual": None}
    y = np.asarray([r["y"] for r in records], np.float64)
    left = np.asarray([r["left"] for r in records], np.float64)
    right = np.asarray([r["right"] for r in records], np.float64)
    widths = right - left
    threshold = max(5.0, float(np.median(widths) * 0.085))
    best = None
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(y[j] - y[i]) < 2:
                continue
            lc = np.polyfit(y[[i, j]], left[[i, j]], 1)
            rc = np.polyfit(y[[i, j]], right[[i, j]], 1)
            residual = np.maximum(np.abs(left - np.polyval(lc, y)), np.abs(right - np.polyval(rc, y)))
            mask = residual <= threshold
            # Image edges and curb labels are supporting evidence, not mandatory.
            support = np.asarray([
                max(r["left_gradient"], r["right_gradient"], min(1.0, 8 * max(r["left_curb"], r["right_curb"])))
                for r in records
            ])
            value = (int(mask.sum()), float(support[mask].sum()), -float(residual[mask].mean()) if mask.any() else -1e9)
            if best is None or value > best[0]:
                best = (value, mask)
    mask = best[1] if best else np.zeros(n, bool)
    if mask.sum() >= 3:
        lc = np.polyfit(y[mask], left[mask], 1)
        rc = np.polyfit(y[mask], right[mask], 1)
        residual = np.maximum(np.abs(left - np.polyval(lc, y)), np.abs(right - np.polyval(rc, y)))
        mask = residual <= threshold
        # Reject a fitted pair that crosses or expands upward implausibly.
        fitted_width = np.polyval(rc, y) - np.polyval(lc, y)
        if np.any(fitted_width <= max(3, image_width * 0.015)):
            mask[:] = False
    else:
        lc = rc = None
        residual = np.full(n, np.inf)
    return mask, {
        "left_coeff": lc.tolist() if lc is not None else None,
        "right_coeff": rc.tolist() if rc is not None else None,
        "threshold_px": threshold,
        "residual": residual.tolist(),
    }


def free_segment(free: np.ndarray, y: int, left: int, right: int, anchor_x: float) -> tuple[int, int] | None:
    h, w = free.shape
    band = max(2, round(h * 0.006))
    y0, y1 = max(0, y - band), min(h, y + band + 1)
    row = free[y0:y1].sum(axis=0) >= max(1, round((y1 - y0) * 0.45))
    clipped = np.zeros(w, bool)
    clipped[max(0, left):min(w, right + 1)] = row[max(0, left):min(w, right + 1)]
    choices = direction.runs(clipped)
    if not choices:
        return None
    containing = [r for r in choices if r[0] <= anchor_x <= r[1]]
    return max(containing or choices, key=lambda r: r[1] - r[0])


def free_segments(free: np.ndarray, y: int, left: int, right: int) -> list[tuple[int, int]]:
    h, w = free.shape
    band = max(2, round(h * 0.006))
    y0, y1 = max(0, y - band), min(h, y + band + 1)
    row = free[y0:y1].sum(axis=0) >= max(1, round((y1 - y0) * 0.45))
    clipped = np.zeros(w, bool)
    clipped[max(0, left):min(w, right + 1)] = row[max(0, left):min(w, right + 1)]
    return direction.runs(clipped)


def draw_path(image: np.ndarray, path: list[tuple[int, int]], color, width=4):
    if len(path) >= 2:
        pts = np.asarray([(x, y) for y, x in path], np.int32)
        cv2.polylines(image, [pts], False, color, width, cv2.LINE_AA)


def process(item: dict) -> dict:
    sid = item["sample_id"]
    with Image.open(ROOT / item["relative_path"]) as source:
        size = fit(source.width, source.height)
        image = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    native_labels = cv2.imread(str(ROOT / item["class_map_path"]), cv2.IMREAD_UNCHANGED)
    repaired_native, _, _ = v2.repair_surface(native_labels)
    labels = cv2.resize(native_labels.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    repaired = cv2.resize(repaired_native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    occupancy_ids = tuple(v2.OCCUPANCY) + tuple(v2.OCCLUSION) + tuple(v2.HAZARD)
    blocked = np.isin(labels, occupancy_ids)
    structural, temp_fill = obstacle_transparent_structure(repaired, blocked)
    free = structural & ~blocked
    route = solve_route(repaired)
    route_index = route["main_index"]
    route_path = route["paths"][route_index] if route_index is not None else []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    records = collect_rows(structural, route_path, labels, gray)
    analysis = branch_and_width_analysis(records, image.shape[1], blocked)

    for record in records:
        record["accepted"] = record["width_state"] in ("complete", "lower_bound", "occlusion_limited")
        segment = free_segment(free, record["y"], record["main_left"], record["main_right"], record["anchor_x"])
        alternatives = free_segments(free, record["y"], record["main_left"], record["main_right"])
        record["free_left"] = segment[0] if segment else None
        record["free_right"] = segment[1] if segment else None
        record["free_width"] = segment[1] - segment[0] + 1 if segment else 0
        record["free_segments"] = [[a, b] for a, b in alternatives]
        record["widest_free_width"] = max((b - a + 1 for a, b in alternatives), default=0)

    accepted_rows = [r for r in records if r["accepted"]]
    complete_rows = [r for r in records if r["width_state"] == "complete"]
    lower_rows = [r for r in records if r["width_state"] == "lower_bound"]
    occluded_rows = [r for r in records if r["width_state"] == "occlusion_limited"]
    branch_rows = [r for r in records if r["left_branch_candidate"] or r["right_branch_candidate"]]
    span = ((max((r["y"] for r in accepted_rows), default=0) - min((r["y"] for r in accepted_rows), default=0)) / image.shape[0])
    inlier_fraction = len(accepted_rows) / len(records) if records else 0.0
    support = np.median([
        max(r["left_gradient"], r["right_gradient"], min(1.0, 8 * max(r["left_curb"], r["right_curb"])))
        for r in accepted_rows
    ]) if accepted_rows else 0.0
    if len(complete_rows) >= 6:
        status = "complete_width_candidate"
    elif len(lower_rows) or len(occluded_rows):
        status = "partial_width_only"
    else:
        status = "measurement_unavailable"

    base = paint(image, repaired, (45, 210, 95), 0.42)
    orientation = paint(base, temp_fill, (255, 150, 35), 0.82)
    candidates = orientation.copy()
    tracked = base.copy()
    structural_width = base.copy()
    free_width = paint(image, free, (60, 185, 245), 0.34)
    kind_colors = {
        "physical_boundary": (255, 65, 205), "semantic_boundary": (255, 65, 205),
        "obstacle_occlusion": (255, 145, 35), "frame_truncated": (145, 145, 145),
    }
    for record in records:
        y, left, right = record["y"], record["left"], record["right"]
        cv2.line(candidates, (left, y), (right, y), (230, 230, 230), 1, cv2.LINE_AA)
        cv2.circle(candidates, (left, y), 4, kind_colors[record["left_kind"]], -1)
        cv2.circle(candidates, (right, y), 4, kind_colors[record["right_kind"]], -1)
        if record["left_branch_candidate"]:
            cv2.line(candidates, (left, y), (record["main_left"], y), (255, 165, 30), 5, cv2.LINE_AA)
        if record["right_branch_candidate"]:
            cv2.line(candidates, (record["main_right"], y), (right, y), (255, 165, 30), 5, cv2.LINE_AA)
        state_color = {
            "complete": (40, 215, 255), "lower_bound": (165, 165, 165),
            "occlusion_limited": (255, 145, 35), "unavailable": (245, 65, 70),
        }[record["width_state"]]
        thickness = 3 if record["accepted"] else 1
        cv2.line(structural_width, (record["main_left"], y), (record["main_right"], y), state_color, thickness, cv2.LINE_AA)
        if record["accepted"]:
            for a, b in record["free_segments"]:
                cv2.line(free_width, (a, y), (b, y), (90, 200, 255), 2, cv2.LINE_AA)
        if record["accepted"] and record["free_left"] is not None:
            cv2.line(free_width, (record["free_left"], y), (record["free_right"], y), (35, 90, 255), 5, cv2.LINE_AA)
    if records:
        y0, y1 = max(r["y"] for r in records), min(r["y"] for r in records)
        ys = np.arange(y0, y1 - 1, -2)
        for boundary_model, color in ((analysis["left_model"], (255, 70, 210)), (analysis["right_model"], (255, 70, 210))):
            coeff = boundary_model["coeff"]
            if coeff is None:
                continue
            xs = np.rint(np.polyval(np.asarray(coeff), ys)).astype(int)
            valid = (xs >= 0) & (xs < image.shape[1])
            pts = np.asarray([(x, y) for x, y in zip(xs[valid], ys[valid])], np.int32)
            if len(pts) >= 2:
                cv2.polylines(tracked, [pts], False, color, 5, cv2.LINE_AA)
    draw_path(tracked, route_path, (35, 110, 255) if route["high_confidence"] else (120, 120, 120), 4)

    art = OUT / "artifacts" / sid
    art.mkdir(parents=True)
    cv2.imwrite(str(art / "01_repaired_surface.png"), repaired.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02_structure_copy.png"), structural.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02b_obstacle_transparent_additions.png"), temp_fill.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "03_free_space.png"), free.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "04_boundary_overlay.jpg"), cv2.cvtColor(tracked, cv2.COLOR_RGB2BGR))

    branch_candidate = np.zeros_like(repaired)
    main_corridor = np.zeros_like(repaired)
    if records:
        ordered = sorted(records, key=lambda r: r["y"])
        ys_known = np.asarray([r["y"] for r in ordered], np.float64)
        left_known = np.asarray([r["main_left"] for r in ordered], np.float64)
        right_known = np.asarray([r["main_right"] for r in ordered], np.float64)
        raw_left_known = np.asarray([r["left"] for r in ordered], np.float64)
        raw_right_known = np.asarray([r["right"] for r in ordered], np.float64)
        y_min, y_max = int(ys_known.min()), int(ys_known.max())
        for y in range(y_min, y_max + 1):
            ml = int(round(np.interp(y, ys_known, left_known)))
            mr = int(round(np.interp(y, ys_known, right_known)))
            rl = int(round(np.interp(y, ys_known, raw_left_known)))
            rr = int(round(np.interp(y, ys_known, raw_right_known)))
            main_corridor[y, max(0, ml):min(image.shape[1], mr + 1)] = repaired[y, max(0, ml):min(image.shape[1], mr + 1)]
            if rl < ml:
                branch_candidate[y, max(0, rl):min(image.shape[1], ml)] = repaired[y, max(0, rl):min(image.shape[1], ml)]
            if mr < rr:
                branch_candidate[y, max(0, mr + 1):min(image.shape[1], rr + 1)] = repaired[y, max(0, mr + 1):min(image.shape[1], rr + 1)]
    cv2.imwrite(str(art / "05_main_corridor_candidate.png"), main_corridor.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "06_branch_candidate.png"), branch_candidate.astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [base, orientation, candidates, tracked, structural_width, free_width]
    titles = [
        "1 Full sidewalk (preserved)", "2 Structural surface (orange=temporary object restoration)", "3 Boundary types + branch candidates",
        "4 Independent boundary continuations + blue selector", "5 Width states (cyan=complete, gray=lower, orange=occluded)",
        "6 Free gaps (light blue=all, dark blue=route-selected)",
    ]
    for ax, title, panel in zip(axes.flat, titles, panels):
        ax.imshow(panel); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{sid} | boundary-width probe | {status} | kept {len(accepted_rows)}/{len(records)}", fontsize=16)
    fig.savefig(art / "summary.png", dpi=115)
    plt.close(fig)

    structure_widths = [r["main_width"] for r in complete_rows]
    free_widths = [r["free_width"] for r in accepted_rows if r["free_width"] > 0]
    payload = {
        "sample_id": sid, "order": item["order"], "split": item["split"], "status": status,
        "candidate_rows": len(records), "accepted_rows": len(accepted_rows), "accepted_span_fraction": span,
        "inlier_fraction": inlier_fraction, "median_boundary_support": float(support),
        "complete_rows": len(complete_rows), "lower_bound_rows": len(lower_rows),
        "occlusion_limited_rows": len(occluded_rows), "branch_candidate_rows": len(branch_rows),
        "branch_candidate_pixels": int(branch_candidate.sum()),
        "temporary_object_restoration_pixels": int(temp_fill.sum()),
        "median_structural_width_px": float(np.median(structure_widths)) if structure_widths else None,
        "median_free_width_px": float(np.median(free_widths)) if free_widths else None,
        "analysis_image_width_px": image.shape[1], "analysis_image_height_px": image.shape[0],
        "new_model_inference": False, "uses_structural_axis": False, "destructive_mask_edit": False,
        "full_sidewalk_preserved": True, "main_corridor_is_derived_candidate": True,
        "structural_surface_is_obstacle_transparent": True,
        "rows": records,
        "warning": "Pixel widths are image-space evidence, not metric sidewalk widths.",
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
        writer.writerows([{k: r[k] for k in summary_fields} for r in rows])
    counts = {status: sum(r["status"] == status for r in rows) for status in (
        "complete_width_candidate", "partial_width_only", "measurement_unavailable")}
    cards = []
    for row in rows:
        sid = html.escape(row["sample_id"])
        sw = "n/a" if row["median_structural_width_px"] is None else f'{row["median_structural_width_px"]:.1f}px'
        fw = "n/a" if row["median_free_width_px"] is None else f'{row["median_free_width_px"]:.1f}px'
        cards.append(
            f'<section><h2>{row["order"]:02d}. {sid} <small>{row["split"]}</small></h2>'
            f'<p>{row["status"]} · kept {row["accepted_rows"]}/{row["candidate_rows"]} rows · '
            f'complete {row["complete_rows"]} · lower {row["lower_bound_rows"]} · occluded {row["occlusion_limited_rows"]} · '
            f'branch rows {row["branch_candidate_rows"]} · structural median {sw} · free median {fw} · no deletion</p>'
            f'<a href="artifacts/{sid}/summary.png"><img src="artifacts/{sid}/summary.png"></a></section>'
        )
    review = (
        '<!doctype html><meta charset="utf-8"><title>Boundary branch probe 8 r5</title>'
        '<style>body{font:16px system-ui;background:#10151d;color:#eef;max-width:1500px;margin:auto;padding:18px}'
        'section{background:#19212c;margin:24px 0;padding:18px;border-radius:12px}img{width:100%;height:auto}small{color:#8ff}</style>'
        '<h1>Obstacle-transparent structure + branch candidate + free-gap probe — 8 images</h1>'
        '<p>No structural axis. Full sidewalk is preserved. Objects are restored only in the structural copy, then subtracted again for free space. '
        'Magenta lines are independent boundary continuations; orange bars in panel 3 mark branch evidence. Cyan means complete structural width, '
        'gray is a lower bound, orange in panel 5 is obstacle-limited. Light blue shows every free gap and dark blue is the route-selected gap.</p>'
        f'<p>Complete {counts["complete_width_candidate"]} · Partial only {counts["partial_width_only"]} · Unavailable {counts["measurement_unavailable"]}</p>'
        + ''.join(cards)
    )
    (OUT / "review.html").write_text(review, encoding="utf-8")
    validation = {
        "expected_samples": 8, "processed_samples": len(rows),
        "all_artifacts_present": all((OUT / "artifacts" / r["sample_id"] / "summary.png").exists() for r in rows),
        "new_model_inference": False, "existing_outputs_overwritten": False,
        "uses_structural_axis": False, "destructive_mask_edit": False,
        "full_sidewalk_preserved": True, "main_corridor_is_derived_candidate": True,
        "structural_surface_is_obstacle_transparent": True,
        "status_counts": counts,
    }
    (OUT / "completion_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text(
        "# Boundary + branch probe 8 r5\n\n"
        "- Cached Mapillary argmax maps only; no new inference\n"
        "- No structural/reference axis\n"
        "- Blue route only selects the relevant sidewalk run\n"
        "- Left/right boundaries are classified and continued independently\n"
        "- Object footprints bracketed by sidewalk are restored only in the structural copy\n"
        "- Object-adjacent deviations are vetoed as branch evidence\n"
        "- All free gaps and the blue-route-selected gap are retained separately\n"
        "- One-sided outward expansion is emitted as a branch candidate\n"
        "- Full sidewalk remains untouched; main corridor is a separate derived candidate\n"
        "- Complete, lower-bound, and obstacle-limited widths are distinguished\n"
        "- Pixel widths are not real-world metric widths\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "samples": len(rows), "status_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
