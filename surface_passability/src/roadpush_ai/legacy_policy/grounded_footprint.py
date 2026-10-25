from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import csv
import html
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = get_settings().project_root
OUT = HERE / "overhead_silhouette_footprint_boundary_probe_40_v6"
AUDIT = HERE / "model_audit_40" / "audit_manifest.json"
RUN_LABEL = "overhead-silhouette footprint + boundary probe"
PROBE_ONLY = True
APPLY_NATURAL_RESTORATION = True

# Mapillary has no Tree class.  Trees, hedges, and bushes all arrive as
# Vegetation.  Mountain/Terrain are included as auditable candidates, but the
# same walkway-ownership rules normally reject them.
NATURAL_GROUNDED_CANDIDATES = (25, 29, 30)
CLASS_NAMES = {25: "Mountain", 29: "Terrain", 30: "Vegetation"}



from roadpush_ai.legacy_policy import integrated_sidewalk as integrated
outer = integrated.outer
policy = integrated.policy


def paint(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.65) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * (1.0 - alpha) + np.asarray(color, np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def filled_outer_envelope(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    envelope = np.zeros_like(component, dtype=np.uint8)
    if contours:
        cv2.drawContours(envelope, contours, -1, 1, cv2.FILLED)
    return envelope.astype(bool)


def horizontal_bracket_rows(
    intrusion: np.ndarray, source_walkway: np.ndarray, search: int
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Return rows where the candidate is visibly bracketed by sidewalk.

    This rejects boundary vegetation/terrain.  Only a non-walk region with
    visible sidewalk on both horizontal sides is eligible for restoration.
    """
    h, w = intrusion.shape
    rows: list[tuple[int, int, int]] = []
    restorable = np.zeros_like(intrusion, dtype=bool)
    for y in np.flatnonzero(intrusion.any(axis=1)):
        xs = np.flatnonzero(intrusion[y])
        if not len(xs):
            continue
        left, right = int(xs[0]), int(xs[-1])
        left_support = source_walkway[y, max(0, left - search):left].any()
        right_support = source_walkway[y, right + 1:min(w, right + 1 + search)].any()
        if left_support and right_support:
            rows.append((int(y), left, right))
            restorable[y, left:right + 1] = True
    return rows, restorable


def natural_ground_footprint(
    component: np.ndarray,
    corridor_envelope: np.ndarray,
    source_walkway: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    component_index: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = component.shape
    inside = component & corridor_envelope
    ys, xs = np.nonzero(inside)
    metadata = {
        "source": "natural_intrusion",
        "class_id": int(class_id),
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "component_index": int(component_index),
        "source_pixels": int(component.sum()),
        "corridor_intrusion_pixels": int(inside.sum()),
        "accepted": False,
    }
    empty = np.zeros_like(component, dtype=bool)
    minimum = max(18, round(component.size * 0.00005))
    if len(xs) < minimum:
        metadata["rejection_reason"] = "too_little_corridor_intrusion"
        return empty, empty, metadata

    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    metadata["intrusion_bbox"] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    search = max(10, round(w * 0.10))
    bracketed, restorable = horizontal_bracket_rows(inside, source_walkway, search)
    metadata["bracketed_rows"] = len(bracketed)
    metadata["intrusion_rows"] = int(len(np.unique(ys)))
    metadata["bracket_fraction"] = len(bracketed) / max(1, metadata["intrusion_rows"])
    if len(bracketed) < 4 or metadata["bracket_fraction"] < 0.12:
        metadata["rejection_reason"] = "not_an_internal_walkway_intrusion"
        return empty, empty, metadata

    # The ground hole is only an anchor.  Estimate the footprint from the object
    # standing immediately above it, never from the hole/flower-bed width itself.
    bottom = max(row[0] for row in bracketed)
    top = min(row[0] for row in bracketed)
    natural = np.isin(labels, np.asarray(NATURAL_GROUNDED_CANDIDATES, dtype=np.uint8))
    x0, x1 = int(xs.min()), int(xs.max())
    anchor_y = max(0, top - 1)
    minimum_vertical_run = max(8, round((bottom - top + 1) * 0.50))
    overhead_columns: list[int] = []
    overhead_runs: dict[int, int] = {}
    for x in range(x0, x1 + 1):
        run = 0
        for y in range(anchor_y, -1, -1):
            if natural[y, x]:
                run += 1
            else:
                break
        if run >= minimum_vertical_run:
            overhead_columns.append(x)
            overhead_runs[x] = run

    source = np.zeros_like(component, dtype=bool)
    if overhead_columns:
        # Keep the longest contiguous overhead group.  This isolates the trunk
        # from the much wider terrain/root patch touching its base.
        groups: list[list[int]] = []
        for x in overhead_columns:
            if not groups or x > groups[-1][-1] + 1:
                groups.append([x])
            else:
                groups[-1].append(x)
        selected_columns = max(groups, key=len)
        left, right = selected_columns[0], selected_columns[-1]
        source_top = max(0, anchor_y - max(overhead_runs[x] for x in selected_columns) + 1)
        source[source_top:bottom + 1, left:right + 1] = natural[source_top:bottom + 1, left:right + 1]
        strategy = "overhead_vertical_silhouette_projected_to_ground"
    else:
        # Some trunks/supports begin exactly at the corridor hole and therefore
        # have no separate pixels above it.  Use the upper half of that upright
        # component, not its flared lower contact patch.
        upper_bottom = top + max(1, (bottom - top + 1) // 2)
        upper_rows = [row for row in bracketed if row[0] <= upper_bottom]
        left = int(round(np.percentile([row[1] for row in upper_rows], 20)))
        right = int(round(np.percentile([row[2] for row in upper_rows], 80)))
        source = component.copy()
        strategy = "upper_upright_silhouette_projected_to_ground"

    stable = float(right - left + 1)
    margin = max(2, round(w * 0.004), round((right - left + 1) * 0.05))
    left, right = max(0, left - margin), min(w - 1, right + margin)
    depth = max(6, min(round((bottom - top + 1) * 0.20), round(h * 0.065)))
    fp_top, fp_bottom = max(0, bottom - depth + 1), min(h - 1, bottom + 2)
    footprint = np.zeros_like(component, dtype=bool)
    footprint[fp_top:fp_bottom + 1, left:right + 1] = True

    # Footprint must touch visible source sidewalk around its base.  Proximity
    # to the later restored surface is deliberately insufficient evidence.
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring = cv2.dilate(footprint.astype(np.uint8), ring_kernel).astype(bool) & ~footprint
    support_pixels = int((ring & source_walkway).sum())
    support_fraction = support_pixels / max(1, int(ring.sum()))
    metadata.update({
        "stable_lower_width_px": stable,
        "silhouette_strategy": strategy,
        "overhead_vertical_run_threshold": minimum_vertical_run,
        "overhead_supported_columns": len(overhead_columns),
        "source_silhouette_pixels": int(source.sum()),
        "footprint_bbox": [left, fp_top, right - left + 1, fp_bottom - fp_top + 1],
        "footprint_pixels": int(footprint.sum()),
        "ground_ring_support_pixels": support_pixels,
        "ground_ring_support_fraction": support_fraction,
    })
    if support_pixels < 8 or support_fraction < 0.08:
        metadata["rejection_reason"] = "no_visible_walkway_ground_support"
        return empty, empty, metadata

    footprint &= corridor_envelope
    metadata["footprint_pixels"] = int(footprint.sum())
    metadata["accepted"] = bool(footprint.any())
    metadata["rejection_reason"] = None if footprint.any() else "footprint_outside_corridor"
    return footprint, source, metadata


def row_span_envelope(component: np.ndarray) -> np.ndarray:
    """Fill between the outermost pixels of the selected corridor on each row.

    Unlike a semantic-class component, this recovers a tree silhouette even when
    the model fragments it into Building/Fence/Road/Vegetation/Terrain labels.
    """
    envelope = np.zeros_like(component, dtype=bool)
    for y in np.flatnonzero(component.any(axis=1)):
        xs = np.flatnonzero(component[y])
        if len(xs):
            envelope[y, int(xs[0]):int(xs[-1]) + 1] = True
    return envelope


def universal_hole_candidates(
    labels: np.ndarray,
    repaired: np.ndarray,
    structural: np.ndarray,
    source_walkway: np.ndarray,
    baseline_footprints: np.ndarray,
):
    corridor = outer.selected_component(structural, outer.route_path(source_walkway)[1])
    if not corridor.any():
        corridor = structural
    envelope = row_span_envelope(corridor)
    holes = envelope & ~repaired
    # Join semantic fragments belonging to one physical occluder, but do not
    # bridge separate nearby obstacles with a wide kernel.
    holes = cv2.morphologyEx(
        holes.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    footprints = np.zeros_like(labels, dtype=bool)
    source_occluders = np.zeros_like(labels, dtype=bool)
    accepted_sources = np.zeros_like(labels, dtype=bool)
    rejected_sources = np.zeros_like(labels, dtype=bool)
    records: list[dict] = []
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    transient_ids = np.asarray(tuple(integrated.TRANSIENT_IDS), dtype=np.uint8)
    existing_policy_ids = np.asarray(
        tuple(policy.probe.v2.OCCUPANCY) + tuple(policy.probe.v2.OCCLUSION),
        dtype=np.uint8,
    )
    natural_ids = np.asarray(NATURAL_GROUNDED_CANDIDATES, dtype=np.uint8)
    covered = cv2.dilate(
        baseline_footprints.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    minimum = max(16, round(labels.size * 0.00004))
    for component_index in range(1, count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) < minimum:
            continue
        component = component_labels == component_index
        component_labels_here = labels[component]
        ids, counts = np.unique(component_labels_here, return_counts=True)
        dominant = int(ids[int(np.argmax(counts))])
        class_mix = {str(int(i)): int(c) for i, c in zip(ids, counts)}
        transient_fraction = float(np.isin(component_labels_here, transient_ids).mean())
        existing_policy_fraction = float(np.isin(component_labels_here, existing_policy_ids).mean())
        natural_fraction = float(np.isin(component_labels_here, natural_ids).mean())
        component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[component_index, cv2.CC_STAT_AREA])
        record_base = {
            "source": "class_agnostic_corridor_hole",
            "component_index": component_index,
            "source_pixels": int(component.sum()),
            "dominant_class_id": dominant,
            "class_mix": class_mix,
            "transient_fraction": transient_fraction,
            "existing_policy_fraction": existing_policy_fraction,
            "natural_fraction": natural_fraction,
            "component_bbox": [
                int(stats[component_index, cv2.CC_STAT_LEFT]),
                int(stats[component_index, cv2.CC_STAT_TOP]),
                component_width,
                component_height,
            ],
            "accepted": False,
        }
        if transient_fraction >= 0.10:
            record_base["rejection_reason"] = "person_or_rider_occlusion"
            records.append(record_base)
            rejected_sources |= component
            continue
        # Existing V2 object classes already pass through the ownership,
        # off-walkway and footprint rules.  The generic hole recovery must not
        # bypass a rejection made by that policy (cars outside the walkway were
        # the clearest failure mode in v3).
        if existing_policy_fraction >= 0.10:
            record_base["rejection_reason"] = "owned_by_existing_object_policy"
            records.append(record_base)
            rejected_sources |= component
            continue
        overlap = int((component & covered).sum())
        if overlap >= max(4, round(component.sum() * 0.015)):
            record_base["rejection_reason"] = "already_covered_by_current_v2_footprint"
            record_base["current_v2_coverage_pixels"] = overlap
            records.append(record_base)
            rejected_sources |= component
            continue
        # A shallow row fragment is normally a paving seam, curb edge or
        # segmentation crack, not an upright occluder with hidden sidewalk
        # behind it.  Require both meaningful vertical extent and area.
        min_height = max(12, round(labels.shape[0] * 0.05))
        min_area = max(80, round(labels.size * 0.00040))
        if component_height < min_height or component_area < min_area:
            record_base["rejection_reason"] = "shallow_or_tiny_surface_fragment"
            records.append(record_base)
            rejected_sources |= component
            continue
        # This supplement exists for grounded natural occluders fragmented by
        # Mapillary's lack of a Tree class.  Recognised man-made objects remain
        # under the current universal V2 object policy above.
        if natural_fraction < 0.60:
            record_base["rejection_reason"] = "insufficient_natural_occluder_support"
            records.append(record_base)
            rejected_sources |= component
            continue
        footprint, source_occluder, metadata = natural_ground_footprint(
            component, envelope, source_walkway, labels, dominant, component_index
        )
        metadata.update(record_base)
        metadata["accepted"] = bool(footprint.any())
        metadata["rejection_reason"] = None if footprint.any() else metadata.get("rejection_reason")
        records.append(metadata)
        if metadata["accepted"]:
            footprints |= footprint
            source_occluders |= source_occluder
            accepted_sources |= component
        else:
            rejected_sources |= component
    return footprints, source_occluders, accepted_sources, rejected_sources, records, envelope


def boundary_analysis(
    surface: np.ndarray,
    free: np.ndarray,
    labels: np.ndarray,
    path: list[tuple[int, int]],
) -> tuple[np.ndarray, list[dict]]:
    component = outer.selected_component(surface, path)
    records = outer.sample_outer_boundaries(component, path, labels)
    for record in records:
        segments = outer.base.free_segments(free, record["y"], record["left"], record["right"])
        selected = outer.choose_route_segment(segments, record["anchor_x"])
        record["free_segments"] = [[a, b] for a, b in segments]
        record["selected_free_segment"] = list(selected) if selected else None
        record["selected_free_width"] = selected[1] - selected[0] + 1 if selected else 0
    return component, records


def draw_boundary_rows(image: np.ndarray, component: np.ndarray, records: list[dict]) -> np.ndarray:
    panel = image.copy()
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(panel, contours, -1, (255, 65, 210), 3, cv2.LINE_AA)
    colors = {"complete": (40, 215, 255), "lower_bound": (155, 155, 155), "occlusion_limited": (255, 145, 35)}
    for record in records:
        cv2.line(
            panel,
            (record["left"], record["y"]),
            (record["right"], record["y"]),
            colors.get(record["width_state"], (235, 235, 235)),
            2,
            cv2.LINE_AA,
        )
    return panel


def draw_free_rows(image: np.ndarray, free: np.ndarray, records: list[dict]) -> np.ndarray:
    panel = paint(image, free, (85, 185, 235), 0.26)
    for record in records:
        y = record["y"]
        for left, right in record["free_segments"]:
            cv2.line(panel, (left, y), (right, y), (90, 205, 255), 2, cv2.LINE_AA)
        if record["selected_free_segment"]:
            left, right = record["selected_free_segment"]
            cv2.line(panel, (left, y), (right, y), (35, 85, 255), 5, cv2.LINE_AA)
    return panel


def process(item: dict, model_holder: dict) -> dict:
    sid = item["sample_id"]
    image, labels = integrated.load_inputs(item)
    repaired, walkway = integrated.selected_walkway(labels)
    blocked_ids = tuple(policy.probe.v2.OCCUPANCY) + tuple(policy.probe.v2.OCCLUSION)
    raw_blocked = np.isin(labels, blocked_ids)
    raw_hazards = np.isin(labels, tuple(policy.probe.v2.HAZARD))
    structural, _ = outer.base.obstacle_transparent_structure(repaired, raw_blocked | raw_hazards)

    baseline_fp, baseline_rows, yolo_meta, _ = integrated.integrated_footprints(
        labels, structural, image, model_holder
    )
    natural_fp, natural_source, accepted_natural, rejected_natural, natural_rows, envelope = universal_hole_candidates(
        labels, repaired, structural, walkway, baseline_fp
    )
    candidate_structural, _ = outer.base.obstacle_transparent_structure(
        repaired, raw_blocked | raw_hazards | natural_source
    )
    candidate_restore = candidate_structural & ~structural
    if natural_source.any() and not candidate_restore.any():
        # A footprint without any newly recovered occluded sidewalk only adds a
        # duplicate blockage.  It cannot improve the boundary reconstruction.
        for row in natural_rows:
            if row.get("accepted"):
                row["accepted"] = False
                row["rejection_reason"] = "no_new_occluded_sidewalk_restoration"
        natural_fp[:] = False
        natural_source[:] = False
        candidate_structural = structural.copy()
        candidate_restore[:] = False
    probe_structural = candidate_structural if APPLY_NATURAL_RESTORATION else structural.copy()
    natural_restore = candidate_restore if APPLY_NATURAL_RESTORATION else np.zeros_like(structural, dtype=bool)
    probe_fp = baseline_fp | natural_fp
    current_free = structural & ~raw_hazards & ~baseline_fp
    probe_free = probe_structural & ~raw_hazards & ~probe_fp

    _, curb_suspect, _, _ = integrated.curb_probe.analyze(labels)
    route_surface = repaired & ~curb_suspect
    if not route_surface.any():
        route_surface = repaired
    _, path = outer.route_path(route_surface)
    current_component, current_records = boundary_analysis(
        structural & ~curb_suspect, current_free & ~curb_suspect, labels, path
    )
    probe_component, probe_records = boundary_analysis(
        probe_structural & ~curb_suspect, probe_free & ~curb_suspect, labels, path
    )
    current_by_y = {record["y"]: record for record in current_records}
    probe_by_y = {record["y"]: record for record in probe_records}
    boundary_changed_rows = sum(
        1 for y in set(current_by_y) | set(probe_by_y)
        if y not in current_by_y or y not in probe_by_y or
        current_by_y[y]["left"] != probe_by_y[y]["left"] or
        current_by_y[y]["right"] != probe_by_y[y]["right"] or
        current_by_y[y].get("selected_free_segment") != probe_by_y[y].get("selected_free_segment")
    )
    structural_boundary_changed_rows = sum(
        1 for y in set(current_by_y) | set(probe_by_y)
        if y not in current_by_y or y not in probe_by_y or
        current_by_y[y]["left"] != probe_by_y[y]["left"] or
        current_by_y[y]["right"] != probe_by_y[y]["right"]
    )
    free_width_rows_changed = sum(
        1 for y in set(current_by_y) | set(probe_by_y)
        if y not in current_by_y or y not in probe_by_y or
        current_by_y[y].get("selected_free_segment") != probe_by_y[y].get("selected_free_segment")
    )

    current_view = paint(image, repaired, (45, 210, 95), 0.42)
    source_view = paint(image, natural_source, (180, 80, 225), 0.62)
    source_view = paint(source_view, natural_fp, (255, 195, 35), 0.90)
    restore_view = paint(image, probe_structural, (45, 210, 95), 0.30)
    restore_view = paint(restore_view, natural_restore, (255, 145, 35), 0.78)
    restore_view = paint(restore_view, probe_fp, (255, 195, 35), 0.88)
    current_boundary_view = draw_boundary_rows(image, current_component, current_records)
    probe_boundary_view = draw_boundary_rows(image, probe_component, probe_records)
    probe_free_view = draw_free_rows(image, probe_free & ~curb_suspect, probe_records)

    accepted = [row for row in natural_rows if row["accepted"]]
    art = OUT / "artifacts" / sid
    art.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(art / "01_current_v2_footprints.png"), baseline_fp.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "02_natural_ground_footprints.png"), natural_fp.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "03_source_occluders.png"), natural_source.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "04_natural_restoration.png"), natural_restore.astype(np.uint8) * 255)
    cv2.imwrite(str(art / "05_probe_free_space.png"), probe_free.astype(np.uint8) * 255)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = [current_view, source_view, current_boundary_view, restore_view, probe_boundary_view, probe_free_view]
    titles = [
        "1 Current V2 repaired sidewalk",
        "2 Overhead source silhouette (purple) + projected footprint (yellow)",
        "3 BEFORE: current outer boundary + width rows",
        ("4 AFTER: silhouette-only restoration (orange) + all footprints" if APPLY_NATURAL_RESTORATION
         else "4 AFTER: V2 structural sidewalk unchanged + all footprints"),
        "5 AFTER: recomputed outer boundary + width rows",
        "6 AFTER: recomputed free-width rows (dark blue=route-selected)",
    ]
    for ax, title, panel in zip(axes.flat, titles, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{sid} | {RUN_LABEL} | new accepted {len(accepted)}", fontsize=15)
    fig.savefig(art / "summary.png", dpi=115)
    plt.close(fig)

    payload = {
        "sample_id": sid,
        "order": item.get("order"),
        "split": item.get("split"),
        "current_v2_accepted_footprints": sum(bool(row.get("accepted")) for row in baseline_rows),
        "natural_candidates": len(natural_rows),
        "natural_accepted": len(accepted),
        "natural_footprint_pixels": int(natural_fp.sum()),
        "natural_restored_pixels": int(natural_restore.sum()),
        "candidate_restored_pixels_not_applied": int(candidate_restore.sum()) if not APPLY_NATURAL_RESTORATION else 0,
        "natural_source_silhouette_pixels": int(natural_source.sum()),
        "boundary_rows_before": len(current_records),
        "boundary_rows_after": len(probe_records),
        "boundary_or_free_width_rows_changed": boundary_changed_rows,
        "structural_boundary_rows_changed": structural_boundary_changed_rows,
        "free_width_rows_changed": free_width_rows_changed,
        "median_selected_free_width_before_px": float(np.median([
            r["selected_free_width"] for r in current_records if r["selected_free_width"]
        ])) if any(r["selected_free_width"] for r in current_records) else None,
        "median_selected_free_width_after_px": float(np.median([
            r["selected_free_width"] for r in probe_records if r["selected_free_width"]
        ])) if any(r["selected_free_width"] for r in probe_records) else None,
        "yolo_state": (
            "split_confirmed" if yolo_meta.get("split_confirmed_count") else
            "triggered_fallback" if yolo_meta.get("triggered") else "not_triggered"
        ),
        "natural_components": natural_rows,
        "policy": {
            "probe_only": PROBE_ONLY,
            "existing_v2_modified": not PROBE_ONLY,
            "people_riders_excluded": True,
            "off_walkway_rejected": True,
            "existing_v2_footprints_preserved": True,
            "natural_overhead_restoration_applied": APPLY_NATURAL_RESTORATION,
            "new_candidate_source": "class_agnostic_corridor_holes",
            "width_direction_branch_logic_changed": False,
        },
    }
    (art / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    manifest = json.loads(AUDIT.read_text(encoding="utf-8"))
    items = manifest["samples"] if isinstance(manifest, dict) else manifest
    model_holder: dict = {}
    results = [process(item, model_holder) for item in items]

    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "order", "sample_id", "split", "current_v2_accepted_footprints",
            "natural_candidates", "natural_accepted", "natural_footprint_pixels",
            "natural_restored_pixels", "natural_source_silhouette_pixels",
            "boundary_rows_before", "boundary_rows_after",
            "boundary_or_free_width_rows_changed",
            "median_selected_free_width_before_px", "median_selected_free_width_after_px",
            "yolo_state",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fields})

    cards = []
    for result in results:
        sid = result["sample_id"]
        cards.append(
            '<section class="card">'
            f'<h2>{int(result.get("order") or 0):02d}. {html.escape(sid)} '
            f'<span>{html.escape(str(result.get("split", "")))}</span></h2>'
            f'<p>current footprints {result["current_v2_accepted_footprints"]} · '
            f'natural candidates {result["natural_candidates"]} · '
            f'natural accepted {result["natural_accepted"]} · '
            f'changed rows {result["boundary_or_free_width_rows_changed"]} · '
            f'width median {result["median_selected_free_width_before_px"]} → '
            f'{result["median_selected_free_width_after_px"]}px · '
            f'YOLO {html.escape(result["yolo_state"])}</p>'
            f'<a href="artifacts/{html.escape(sid)}/result.json">result.json</a>'
            f'<img loading="lazy" src="artifacts/{html.escape(sid)}/summary.png">'
            '</section>'
        )
    review = (
        '<!doctype html><meta charset="utf-8"><title>Overhead silhouette footprint + boundary probe 40 v6</title>'
        '<style>body{margin:0;background:#111a26;color:#eef5ff;font-family:system-ui;padding:18px}'
        'h1{margin:0 0 8px}.note{color:#b9c9dc;margin-bottom:20px}.card{background:#192433;'
        'border-radius:14px;padding:16px;margin:0 0 18px}.card span{color:#64e7df}.card p{font-size:18px}'
        'a{color:#68d9ff}.card img{width:100%;display:block;margin-top:10px;background:white}</style>'
        '<h1>Overhead silhouette footprint + boundary probe — 40 images (v6)</h1>'
        '<p class="note">Probe only. The ground hole is used only as an anchor. Purple is the '
        'overhead object silhouette and yellow is its projected footprint. Panels 3 and 5 show '
        'the actual outer-boundary/width rows before and after silhouette-only restoration. '
        'Current V2, direction, and branch policy remain unchanged.</p>' + ''.join(cards)
    )
    (OUT / "review.html").write_text(review, encoding="utf-8")
    validation = {
        "status": "completed_probe_not_accuracy_pass",
        "expected_samples": 40,
        "processed_samples": len(results),
        "all_artifacts_present": all((OUT / "artifacts" / r["sample_id"] / "summary.png").exists() for r in results),
        "existing_v2_modified": False,
        "existing_outputs_overwritten": False,
        "cached_mapillary_maps_reused": True,
        "conditional_yolo_samples": [r["sample_id"] for r in results if r["yolo_state"] != "not_triggered"],
        "samples_with_natural_acceptance": [r["sample_id"] for r in results if r["natural_accepted"]],
        "natural_accepted_total": sum(r["natural_accepted"] for r in results),
    }
    (OUT / "completion_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(
        "# Overhead silhouette footprint + boundary probe 40 v6\n\n"
        "- Probe only; integrated V2 v4 remains unchanged\n"
        "- Preserves all current V2 accepted footprints and ownership filters\n"
        "- People/riders remain excluded; off-walkway objects remain rejected\n"
        "- Current object classes stay under the existing universal ownership/off-walkway policy\n"
        "- Audits missing natural non-walk components only when they intrude inside the selected walkway corridor\n"
        "- Rejects shallow surface fragments and candidates without strong natural-occluder support\n"
        "- Requires visible sidewalk on both horizontal sides before accepting an occlusion anchor\n"
        "- Estimates footprint width from the upright object silhouette above the anchor, not the ground-hole width\n"
        "- Restores only the accepted overhead silhouette and retains its projected ground footprint\n"
        "- Recomputes and visualizes the actual outer boundary and horizontal width rows after restoration\n"
        "- Does not change width, direction, branch, or final service policy\n"
        "- Completion is not an accuracy pass; visual review is required\n",
        encoding="utf-8",
    )
    print(json.dumps(validation | {"output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
