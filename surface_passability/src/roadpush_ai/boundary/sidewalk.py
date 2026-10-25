from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.surface.preprocessing import SharedPreprocessing


class BoundaryMode(str, Enum):
    TEST1_OBSERVED_ONLY = "test1_observed_only"
    TEST2_LOCAL_BRIDGE = "test2_local_bridge"
    TEST3_A2_ROW_SPAN = "test3_a2_row_span"


@dataclass
class BoundaryResult:
    mask: np.ndarray
    additions: np.ndarray
    mode: BoundaryMode
    calls: dict[str, bool]


def construct_boundary(shared: SharedPreprocessing, mode: BoundaryMode) -> BoundaryResult:
    empty = np.zeros_like(shared.labels, dtype=bool)
    calls = {
        "repair_surface": True,
        "structure_copy": True,
        "route_component_selection_before_completion": False,
        "v2_horizontal_obstacle_bridge": False,
        "v2_1_natural_restoration": False,
        "a1_corridor_hole_restoration": False,
        "a2_row_span_completion": False,
        "all_candidate_outer_row_span": False,
    }

    if mode == BoundaryMode.TEST1_OBSERVED_ONLY:
        return BoundaryResult(shared.observed_component.copy(), empty, mode, calls)

    if mode == BoundaryMode.TEST2_LOCAL_BRIDGE:
        blocked_ids = tuple(legacy.policy.probe.v2.OCCUPANCY) + tuple(
            legacy.policy.probe.v2.OCCLUSION
        ) + tuple(legacy.policy.probe.v2.HAZARD)
        raw_blocked = np.isin(shared.labels, blocked_ids)
        bridged, _ = legacy.base.obstacle_transparent_structure(shared.repaired, raw_blocked)
        bridged &= ~shared.curb_suspect
        component = legacy.outer.selected_component(bridged, shared.route_path)
        calls["v2_horizontal_obstacle_bridge"] = True
        calls["route_component_selection_before_completion"] = True
        final = component if component.any() else bridged
        return BoundaryResult(final, final & ~shared.observed_component, mode, calls)

    if mode == BoundaryMode.TEST3_A2_ROW_SPAN:
        # The revised V3 Test3 is intentionally broader than the historical
        # selected-component A2 probe: every observed structure_copy candidate
        # in the row contributes to the outer endpoints.  Route identity is
        # imposed only after this boundary is complete.
        candidates = shared.structure_copy & ~shared.curb_suspect
        valid, left, right = legacy.a2.row_spans(candidates)
        span = legacy.a2.span_mask(valid, left, right, shared.labels.shape)
        span &= ~shared.curb_suspect
        calls["a2_row_span_completion"] = True
        calls["all_candidate_outer_row_span"] = True
        return BoundaryResult(span, span & ~shared.observed_component, mode, calls)

    raise ValueError(f"Unsupported boundary mode: {mode}")

