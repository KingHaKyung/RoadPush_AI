from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "policy_aligned_clearance_probe_40_r6"
BANNER_ID = 32
TRANSIENT_IDS = {19, 20, 21, 22}
MOBILE_BODY_IDS = set(range(52, 63))



from roadpush_ai.legacy_policy import class_aware_clearance as r3
probe = r3.probe
probe.OUT = OUT
probe.SUMMARY_LABEL = "V2-policy-aligned clearance probe"
probe.FOOTPRINT_PANEL_LABEL = "3 Accepted walkway clearance envelopes (yellow)"
probe.WIDTH_PANEL_LABEL = "6 Width free space (walkway-owned envelopes only)"


def bbox_distance(a: list[int], b: list[int]) -> tuple[int, int]:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw - 1, ay0 + ah - 1
    bx1, by1 = bx0 + bw - 1, by0 + bh - 1
    dx = max(0, max(ax0, bx0) - min(ax1, bx1) - 1)
    dy = max(0, max(ay0, by0) - min(ay1, by1) - 1)
    return dx, dy


def ground_support(component: np.ndarray, footprint: np.ndarray, walkway: np.ndarray) -> tuple[np.ndarray, int, float]:
    h, w = component.shape
    ys, xs = np.nonzero(component)
    if not len(xs):
        return np.zeros_like(footprint), 0, 0.0
    bottom = int(ys.max())
    depth = max(5, round(h * 0.035))
    y0, y1 = max(0, bottom - 1), min(h, bottom + depth + 1)
    support_rows = walkway[y0:y1]
    supported_columns = support_rows.any(axis=0)
    kernel_width = max(3, round(w * 0.012)) | 1
    supported_columns = cv2.dilate(
        supported_columns.astype(np.uint8)[None, :], np.ones((1, kernel_width), np.uint8)
    )[0].astype(bool)
    effective = footprint & supported_columns[None, :]
    support_pixels = int(effective.sum())
    fraction = support_pixels / max(1, int(footprint.sum()))
    return effective, support_pixels, fraction


def source_contact_support(component: np.ndarray, walkway: np.ndarray) -> tuple[int, int, float, bool]:
    """Check the object's original lowest contact columns before projecting its width.

    A road vehicle can overhang the sidewalk in image space even though its wheels are
    on the road.  Testing only the projected rectangle therefore creates false walkway
    obstacles.  Bottom-truncated objects are retained conservatively because their true
    contact point is outside the image.
    """
    h, w = component.shape
    ys, xs = np.nonzero(component)
    if not len(xs):
        return 0, 0, 0.0, False
    bottom = int(ys.max())
    if bottom >= h - 2:
        return 0, 0, 1.0, True

    bbox_height = int(ys.max() - ys.min() + 1)
    band_top = max(int(ys.min()), bottom - max(3, round(bbox_height * 0.10)) + 1)
    contact = component.copy()
    contact[:band_top] = False
    contact_columns = np.flatnonzero(contact.any(axis=0))
    if not len(contact_columns):
        return 0, 0, 0.0, False

    # Stay immediately below the visible contact.  A deeper vertical probe can land
    # on nearer foreground sidewalk even when a distant vehicle stands on the road.
    depth = 3
    supported = 0
    for x in contact_columns:
        local_bottom = int(np.flatnonzero(component[:, x]).max())
        y0, y1 = local_bottom + 1, min(h, local_bottom + depth + 1)
        x0, x1 = max(0, int(x) - 2), min(w, int(x) + 3)
        if y0 < y1 and walkway[y0:y1, x0:x1].any():
            supported += 1
    total = int(len(contact_columns))
    ratio = supported / max(1, total)
    return supported, total, ratio, False


def policy_estimate_footprints(labels: np.ndarray, structural: np.ndarray):
    # Recompute the conservative V2 source surface at analysis resolution.  Ground
    # ownership is tested against this source, never mere proximity to a restored surface.
    repaired, _, _ = probe.v2.repair_surface(labels)
    route, path = probe.outer.route_path(repaired)
    walkway = probe.outer.selected_component(repaired, path)
    if not walkway.any():
        walkway = repaired

    object_ids = tuple(probe.v2.OCCUPANCY)
    footprints = np.zeros_like(labels, dtype=bool)
    cycle_footprints = np.zeros_like(labels, dtype=bool)
    rows: list[dict] = []
    accepted_sources: list[tuple[np.ndarray, np.ndarray, dict]] = []

    for class_id in object_ids:
        if class_id in TRANSIENT_IDS or class_id == BANNER_ID:
            continue
        binary = (labels == class_id).astype(np.uint8)
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_index in range(1, count):
            if int(stats[component_index, cv2.CC_STAT_AREA]) < 4:
                continue
            component = component_labels == component_index
            estimated = r3.class_aware_component_footprint(component, class_id)
            if estimated is None:
                continue
            candidate, metadata = estimated
            effective, support_pixels, support_fraction = ground_support(component, candidate, walkway)
            minimum_support = max(6, round(candidate.sum() * 0.04))
            source_supported, source_total, source_ratio, bottom_truncated = source_contact_support(component, walkway)
            source_contact_ok = (
                class_id not in MOBILE_BODY_IDS
                or bottom_truncated
                or (source_supported >= 2 and source_ratio >= 0.35)
            )
            accepted = support_pixels >= minimum_support and source_contact_ok
            if accepted:
                if class_id in probe.CYCLE_IDS:
                    cycle_footprints |= effective
                else:
                    footprints |= effective
                accepted_sources.append((component, effective, metadata))
            metadata.update({
                "component_index": component_index,
                "ground_support_pixels": support_pixels,
                "ground_support_fraction": support_fraction,
                "minimum_ground_support_pixels": minimum_support,
                "source_contact_supported_columns": source_supported,
                "source_contact_total_columns": source_total,
                "source_contact_support_fraction": source_ratio,
                "source_bottom_truncated": bottom_truncated,
                "source_contact_required": class_id in MOBILE_BODY_IDS,
                "accepted": accepted,
                "rejection_reason": (
                    None if accepted else
                    "off_walkway_source_contact" if not source_contact_ok else
                    "no_selected_walkway_ground_support"
                ),
            })
            rows.append(metadata)

    # Close only narrow same-depth gaps between cycle-like envelopes.  This prevents a
    # few-pixel pseudo-passage between wheels or fragmented bicycle/motorcycle components.
    close_width = max(7, round(labels.shape[1] * 0.05)) | 1
    merged_cycles = cv2.morphologyEx(
        cycle_footprints.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 3)),
    ).astype(bool)
    cycle_gap_fill = merged_cycles & ~cycle_footprints
    footprints |= merged_cycles

    # Promote Banner only when it physically connects to an already accepted grounded
    # support/base.  Overhead banners remain context-only.
    banner_binary = (labels == BANNER_ID).astype(np.uint8)
    count, banner_labels, stats, _ = cv2.connectedComponentsWithStats(banner_binary, 8)
    for component_index in range(1, count):
        component = banner_labels == component_index
        if int(component.sum()) < 4:
            continue
        ys, xs = np.nonzero(component)
        banner_bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        supporters = []
        for source_component, source_footprint, source_metadata in accepted_sources:
            dx, dy = bbox_distance(banner_bbox, source_metadata["source_bbox"])
            if dx <= max(6, round(labels.shape[1] * 0.025)) and dy <= max(8, round(labels.shape[0] * 0.04)):
                supporters.append((source_footprint, source_metadata, dx + dy))
        if not supporters:
            rows.append({
                "class_id": BANNER_ID, "kind": "conditional_grounded_banner",
                "source_bbox": banner_bbox, "accepted": False,
                "rejection_reason": "no_grounded_support_connection",
            })
            continue
        support_footprint, support_metadata, _ = min(supporters, key=lambda value: value[2])
        fy, fx = np.nonzero(support_footprint)
        if not len(fx):
            continue
        left = max(0, int(xs.min()) - max(2, round(labels.shape[1] * 0.004)))
        right = min(labels.shape[1] - 1, int(xs.max()) + max(2, round(labels.shape[1] * 0.004)))
        top, bottom = int(fy.min()), int(fy.max())
        banner_envelope = np.zeros_like(labels, dtype=bool)
        banner_envelope[top:bottom + 1, left:right + 1] = True
        banner_envelope &= cv2.dilate(walkway.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        footprints |= banner_envelope
        rows.append({
            "class_id": BANNER_ID, "kind": "conditional_grounded_banner",
            "clearance_strategy": "banner_width_projected_to_grounded_base",
            "source_bbox": banner_bbox,
            "footprint_bbox": [left, top, right - left + 1, bottom - top + 1],
            "footprint_pixels": int(banner_envelope.sum()),
            "ground_support_class_id": support_metadata["class_id"],
            "accepted": bool(banner_envelope.any()),
            "rejection_reason": None if banner_envelope.any() else "projection_outside_selected_walkway",
        })

    if cycle_gap_fill.any():
        rows.append({
            "class_id": -1, "kind": "cycle_gap_closure", "accepted": True,
            "footprint_pixels": int(cycle_gap_fill.sum()), "close_width_px": close_width,
        })
    return footprints, rows


def grounded_banner_visual(labels: np.ndarray, footprints: np.ndarray) -> np.ndarray:
    banner = (labels == BANNER_ID).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(banner, 8)
    result = np.zeros_like(labels, dtype=bool)
    expanded_footprints = cv2.dilate(
        footprints.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    ).astype(bool)
    for component_id in range(1, count):
        component = component_labels == component_id
        if (cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & expanded_footprints).any():
            result |= component
    return result


def relabel_outputs() -> None:
    review = OUT / "review.html"
    text = review.read_text(encoding="utf-8")
    text = text.replace("Ground footprint probe 40 r1", "V2 policy-aligned clearance probe 40 r6")
    text = text.replace("Stable lower-span ground-footprint probe", "V2 policy-aligned clearance-envelope probe")
    text = text.replace(
        "2-D ground footprints estimated from repeated wide spans in each object lower band. The absolute widest row is not used.",
        "walkway-owned class-aware envelopes. Transient people/riders do not create fixed width envelopes; grounded banners are conditional; off-walkway objects are rejected.",
    )
    review.write_text(text, encoding="utf-8")

    validation_path = OUT / "completion_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update({
        "ground_width_uses_stable_lower_span_footprints": False,
        "v2_policy_aligned": True,
        "people_riders_fixed_envelopes": False,
        "off_walkway_ground_support_required": True,
        "mobile_source_contact_required_before_projection": True,
        "grounded_banner_conditional": True,
        "cycle_narrow_gap_closure": True,
        "strict_catch_basin_repair": True,
    })
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text(
        "# V2 policy-aligned clearance probe 40 r6\n\n"
        "- Reused 40 cached argmax maps; no model inference\n"
        "- Requires ground support from the selected source walkway, not mere proximity\n"
        "- Requires vehicle/cycle source contact on the walkway before body-width projection\n"
        "- Clips accepted envelopes to ground-supported walkway columns\n"
        "- Does not create fixed clearance envelopes for Person/Rider classes\n"
        "- Closes narrow same-depth gaps between cycle-like components\n"
        "- Promotes Banner only when connected to an accepted grounded support/base\n"
        "- Uses strict small/enclosed Catch Basin repair from the updated V2 policy\n"
        "- Traffic Cone and taxonomy-missed debris remain explicit model limitations\n",
        encoding="utf-8",
    )


def main() -> None:
    probe.component_footprint = r3.class_aware_component_footprint
    probe.estimate_footprints = policy_estimate_footprints
    probe.conditional_visual_obstacles = grounded_banner_visual
    probe.main()
    relabel_outputs()


if __name__ == "__main__":
    main()
