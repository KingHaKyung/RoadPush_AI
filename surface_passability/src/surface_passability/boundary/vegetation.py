"""Grounded-vegetation evidence and conservative boundary-cap gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from surface_passability.boundary.direction import (
    rebuild_selected_boundary_bands,
    reject_straight_boundary_candidate,
)
from surface_passability.policies.vegetation import (
    GroundedVegetationCapPolicy,
)


POLICY_ID = "phase7_trapezoid_boundary_retry_v5_probe_v_grounded_vegetation_cap"


def boundary_endpoint_column(band: dict[str, Any], side: str) -> int:
    """Return the integer image column for one fitted band endpoint."""

    return int(round(float(band[side])))


def diagnose_grounded_vegetation_cap(
    result: dict[str, Any],
    semantic_labels: np.ndarray,
    policy: GroundedVegetationCapPolicy | None = None,
) -> dict[str, Any]:
    """Measure sustained same-component vegetation at near and far caps."""

    policy = policy or GroundedVegetationCapPolicy()
    labels = np.asarray(semantic_labels)
    bands = rebuild_selected_boundary_bands(result)
    if (
        result.get("envelope_fit", {}).get("status") != "accepted"
        or len(bands) < policy.cap_band_count
        or labels.ndim != 2
    ):
        return {
            "status": "not_applicable",
            "gate_would_reject": False,
            "gate_applied": False,
            "failing_caps": [],
            "cap_diagnostics": [],
            "thresholds": asdict(policy),
        }

    vegetation = labels == policy.vegetation_class_id
    component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        vegetation.astype(np.uint8),
        connectivity=8,
    )
    height, width = labels.shape
    cap_groups = {
        "far": bands[: policy.cap_band_count],
        "near": bands[-policy.cap_band_count :],
    }
    diagnostics = []
    failures = []
    contact_mask = np.zeros_like(vegetation)
    for cap_name, cap_bands in cap_groups.items():
        for side in ("left", "right"):
            contacts = []
            for band in cap_bands:
                row_y = int(band["center_y"])
                column_x = boundary_endpoint_column(band, side)
                radius = policy.endpoint_contact_radius_px
                y0 = max(0, row_y - radius)
                y1 = min(height, row_y + radius + 1)
                x0 = max(0, column_x - radius)
                x1 = min(width, column_x + radius + 1)
                patch_vegetation = vegetation[y0:y1, x0:x1]
                fraction = float(np.mean(patch_vegetation))
                patch_components = component_map[y0:y1, x0:x1]
                component_values = patch_components[patch_vegetation]
                component_id = 0
                if component_values.size:
                    positive = component_values[component_values > 0]
                    if positive.size:
                        values, counts = np.unique(
                            positive,
                            return_counts=True,
                        )
                        component_id = int(values[int(np.argmax(counts))])
                qualifies = (
                    fraction >= policy.minimum_local_vegetation_fraction
                    and component_id > 0
                )
                if qualifies:
                    contact_mask[y0:y1, x0:x1] |= patch_vegetation
                contacts.append(
                    {
                        "band_center": row_y,
                        "vegetation_fraction": fraction,
                        "component_id": component_id,
                        "qualifies": qualifies,
                    }
                )

            qualifying_ids = [
                contact["component_id"]
                for contact in contacts
                if contact["qualifies"]
            ]
            component_id = 0
            sustained_count = 0
            if qualifying_ids:
                values, counts = np.unique(
                    qualifying_ids,
                    return_counts=True,
                )
                best = int(np.argmax(counts))
                component_id = int(values[best])
                sustained_count = int(counts[best])

            component = None
            vertical = False
            grounded_through_support = False
            if 0 < component_id < component_count:
                x0, y0, component_width, component_height, area = [
                    int(value) for value in stats[component_id]
                ]
                aspect = float(component_height / max(1, component_width))
                height_fraction = float(component_height / max(1, height))
                support_bottom = int(bands[-1]["center_y"])
                component_bottom = y0 + component_height - 1
                vertical = (
                    aspect >= policy.minimum_component_vertical_aspect
                    and height_fraction
                    >= policy.minimum_component_image_height_fraction
                )
                grounded_through_support = component_bottom >= support_bottom
                component = {
                    "component_id": component_id,
                    "bbox": [
                        x0,
                        y0,
                        component_width,
                        component_height,
                    ],
                    "area_px": area,
                    "vertical_aspect": aspect,
                    "image_height_fraction": height_fraction,
                    "bottom_y": component_bottom,
                }

            gate = (
                sustained_count >= policy.minimum_sustained_contact_bands
                and vertical
                and grounded_through_support
            )
            item = {
                "cap": cap_name,
                "side": side,
                "contacts": contacts,
                "dominant_component": component,
                "sustained_contact_band_count": sustained_count,
                "vertical_component_gate_passed": vertical,
                "grounded_through_support_gate_passed": (
                    grounded_through_support
                ),
                "gate_would_reject": gate,
            }
            diagnostics.append(item)
            if gate:
                failures.append(
                    {
                        "cap": cap_name,
                        "side": side,
                        "component_id": component_id,
                    }
                )

    return {
        "status": "grounded_vegetation_cap" if failures else "clear",
        "gate_would_reject": bool(failures),
        "gate_applied": False,
        "failing_caps": failures,
        "cap_diagnostics": diagnostics,
        "thresholds": asdict(policy),
        "vegetation_mask": vegetation,
        "contact_mask": contact_mask,
    }


def apply_grounded_vegetation_cap(
    result: dict[str, Any],
    masks: dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    policy: GroundedVegetationCapPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Attach vegetation evidence and reject only a grounded cap failure."""

    policy = policy or GroundedVegetationCapPolicy()
    diagnostics = diagnose_grounded_vegetation_cap(
        result,
        semantic_labels,
        policy,
    )
    vegetation_mask = diagnostics.pop(
        "vegetation_mask",
        np.zeros_like(semantic_labels, dtype=bool),
    )
    contact_mask = diagnostics.pop(
        "contact_mask",
        np.zeros_like(semantic_labels, dtype=bool),
    )
    output = deepcopy(result)
    output_masks = {name: value.copy() for name, value in masks.items()}
    output_masks["probe_v_vegetation"] = vegetation_mask
    output_masks["probe_v_cap_contact"] = contact_mask
    output["policy_id"] = POLICY_ID
    output["probe_v_grounded_vegetation_cap"] = diagnostics
    output["probe_v_contract"] = {
        "mode": "candidate_gate",
        "vegetation_only_at_selected_caps": True,
        "sustained_same_component_contact_required": True,
        "vertical_component_required": True,
        "ground_contact_through_selected_support_required": True,
        "middle_vegetation_not_rejected": True,
        "vegetation_not_added_to_global_obstacle_silhouette": True,
        "candidate_ranking_changed": False,
        "blocker_logic_changed": False,
        "extension_or_curve_added": False,
        "threshold_status": "provisional_targeted_review_candidate",
    }
    if diagnostics["gate_would_reject"]:
        output, output_masks = reject_straight_boundary_candidate(
            output,
            output_masks,
        )
        output["envelope_fit"]["rejection_reasons"] = list(
            dict.fromkeys(
                [
                    *output["envelope_fit"]["rejection_reasons"],
                    "probe_v_grounded_vegetation_cap",
                ]
            )
        )
        diagnostics["gate_applied"] = True
        output["probe_v_grounded_vegetation_cap"] = diagnostics
    return output, output_masks
