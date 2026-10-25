from __future__ import annotations

import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.surface.preprocessing import SharedPreprocessing


def evaluate_rows(
    boundary: np.ndarray,
    blockers: dict,
    shared: SharedPreprocessing,
    route: dict | None = None,
    route_path: list[tuple[int, int]] | None = None,
) -> dict:
    route = route or shared.route
    route_path = shared.route_path if route_path is None else route_path
    component = legacy.outer.selected_component(boundary, route_path)
    if not component.any():
        component = boundary.copy()
    free = component & ~blockers["final_mask"]
    rows = legacy.outer.sample_outer_boundaries(component, route_path, shared.labels)
    branch_uncertain = max(0, len(route.get("paths", [])) - 1) > 0
    for row in rows:
        left, right, y = row["left"], row["right"], row["y"]
        segments = legacy.base.free_segments(free, y, left, right)
        selected = legacy.outer.choose_route_segment(segments, row["anchor_x"])
        transient = bool(blockers["transient_mask"][y, left:right + 1].any())
        row.update({
            "structural_sidewalk_span": [left, right],
            "free_segments": [[a, b] for a, b in segments],
            "selected_route_segment": list(selected) if selected else None,
            "transient_occlusion": transient,
            "boundary_reliability": "reliable" if row["width_state"] == "complete" else "uncertain",
            "branch_reliability": "uncertain" if branch_uncertain else "no_alternative_branch_detected",
        })
    return {
        "component": component,
        "free": free,
        "rows": rows,
        "transient_occluded_rows": sum(r["transient_occlusion"] for r in rows),
        "boundary_uncertain_rows": sum(r["boundary_reliability"] == "uncertain" for r in rows),
        "branch_uncertain_rows": sum(r["branch_reliability"] == "uncertain" for r in rows),
    }

