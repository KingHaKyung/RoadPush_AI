from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "class_aware_clearance_probe_40_r3"

# Tall supports may have a wide lamp/sign far above the pedestrian passage.
LOWER_SUPPORT_IDS = {44, 45, 46, 47, 48, 49, 50}
# These classes need their body width projected to the ground, not only tire contacts.
WIDE_BODY_IDS = {19, 20, 21, 22, 33, 34, 38, 39, 40, 42, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62}



from roadpush_ai.legacy_policy import ground_footprint as probe
probe.OUT = OUT
probe.SUMMARY_LABEL = "class-aware clearance-envelope probe"
probe.FOOTPRINT_PANEL_LABEL = "3 Clearance envelopes (yellow): body width at ground position"
probe.WIDTH_PANEL_LABEL = "6 Width free space (clearance envelopes + hazards removed)"


def row_spans(component: np.ndarray, top: int, bottom: int) -> list[tuple[int, int, int, int]]:
    rows = []
    for y in range(top, bottom + 1):
        xs = np.flatnonzero(component[y])
        if len(xs):
            rows.append((y, int(xs[0]), int(xs[-1]), int(xs[-1] - xs[0] + 1)))
    return rows


def robust_span(rows: list[tuple[int, int, int, int]], percentile: float, ratio: float) -> tuple[int, int, float, int]:
    widths = np.asarray([row[3] for row in rows], np.float32)
    target = max(1.0, float(np.percentile(widths, percentile)))
    selected = [row for row in rows if row[3] >= max(1.0, target * ratio)]
    if not selected:
        selected = [max(rows, key=lambda row: row[3])]
    left = int(round(np.percentile([row[1] for row in selected], 10)))
    right = int(round(np.percentile([row[2] for row in selected], 90)))
    return left, right, target, len(selected)


def class_aware_component_footprint(component: np.ndarray, class_id: int):
    image_h, image_w = component.shape
    ys, xs = np.nonzero(component)
    minimum = max(4, round(component.size * 0.000008))
    if len(xs) < minimum:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    object_h = y1 - y0 + 1
    object_w = x1 - x0 + 1

    if class_id in LOWER_SUPPORT_IDS:
        # Only inspect the lower support/base.  A high lamp or sign flare is ignored.
        band_depth = max(6, round(object_h * 0.25))
        band_top = max(y0, y1 - band_depth + 1)
        rows = row_spans(component, band_top, y1)
        left, right, stable_width, selected_rows = robust_span(rows, 85, 0.65)
        patch_depth = max(4, min(round(object_h * 0.16), round(image_h * 0.07)))
        strategy = "lower_support_or_pedestal"
    else:
        # A bicycle, vehicle, person, bench, etc. blocks with its body width.  Use a
        # repeated near-maximum span, then project that width onto a shallow ground band.
        rows = row_spans(component, y0, y1)
        left, right, stable_width, selected_rows = robust_span(rows, 90, 0.72)
        patch_depth = max(4, min(round(object_h * 0.10), round(image_h * 0.035)))
        band_top = y0
        strategy = "wide_body_projected_to_ground" if class_id in WIDE_BODY_IDS else "generic_body_projected_to_ground"

    margin_x = max(2, round(image_w * 0.004), round((right - left + 1) * 0.06))
    margin_y = max(2, round(image_h * 0.004))
    left = max(0, left - margin_x)
    right = min(image_w - 1, right + margin_x)
    top = max(0, y1 - patch_depth + 1 - margin_y)
    bottom = min(image_h - 1, y1 + margin_y)

    footprint = np.zeros_like(component, dtype=np.uint8)
    cv2.rectangle(footprint, (left, top), (right, bottom), 1, cv2.FILLED)
    return footprint.astype(bool), {
        "class_id": int(class_id),
        "source_pixels": int(component.sum()),
        "source_bbox": [x0, y0, object_w, object_h],
        "lower_band_top": int(band_top),
        "stable_width_px": float(stable_width),
        "qualifying_rows": int(selected_rows),
        "selected_rows": int(selected_rows),
        "footprint_bbox": [left, top, right - left + 1, bottom - top + 1],
        "footprint_pixels": int(footprint.sum()),
        "kind": "pole_or_bollard" if class_id == probe.POLE_ID else (
            "cycle_like" if class_id in probe.CYCLE_IDS else "generic_obstacle"
        ),
        "clearance_strategy": strategy,
    }


def relabel_outputs() -> None:
    review = OUT / "review.html"
    text = review.read_text(encoding="utf-8")
    text = text.replace("Ground footprint probe 40 r1", "Class-aware clearance probe 40 r3")
    text = text.replace("Stable lower-span ground-footprint probe", "Class-aware clearance-envelope probe")
    text = text.replace(
        "2-D ground footprints estimated from repeated wide spans in each object lower band. The absolute widest row is not used.",
        "class-aware clearance envelopes: wide bodies use a robust silhouette width projected to the ground, while tall supports use only their lower base/pedestal.",
    )
    review.write_text(text, encoding="utf-8")

    validation_path = OUT / "completion_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["ground_width_uses_stable_lower_span_footprints"] = False
    validation["class_aware_clearance_envelopes"] = True
    validation["wide_body_width_projected_to_ground"] = True
    validation["tall_support_upper_flare_ignored"] = True
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (OUT / "REPORT.md").write_text(
        "# Class-aware clearance probe 40 r3\n\n"
        "- Reused 40 cached native argmax maps; no model inference\n"
        "- Full visible obstacle silhouettes remain excluded from visual free space\n"
        "- Bicycle, motorcycle, vehicle, person, and furniture-like objects use a robust near-maximum body width\n"
        "- The body width is projected onto a shallow band at the object's ground position\n"
        "- Pole, street-light, utility-pole, and sign-support classes use only the lower support/base\n"
        "- Multi-row square pedestals are retained; high lamp/sign flares are ignored\n"
        "- No branch removal, boundary fitting, model inference, or destructive edit\n",
        encoding="utf-8",
    )


def main() -> None:
    probe.component_footprint = class_aware_component_footprint
    probe.main()
    relabel_outputs()


if __name__ == "__main__":
    main()
