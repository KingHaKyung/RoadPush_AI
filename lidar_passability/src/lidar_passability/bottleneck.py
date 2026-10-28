"""Successor prototype: unsmoothed bottleneck, metres first; no model execution.

Legacy P3 remains unchanged for reproducibility. This module uses its geometric
input validator, but never its median-based path to select the new result.
"""

from copy import deepcopy
from pathlib import Path
import sys
from .width_measure import measure as legacy_measure, overlap
from .p2_strip_adapter import adapt_p2

VERSION = "roadpush.bottleneck/0.2-draft"
POLICY = "max_path_min_width_m_ratio_at_same_bottleneck_section"


def select(layers):
    """Widest connected path in the layered interval graph, lexicographic tie.

    Threshold reachability preserves all terminal nodes, avoiding a greedy local
    choice at splits/merges. No filtering or point-touch connectivity is added.
    """
    candidates = sorted({b - a for layer in layers for a, b in layer})

    def feasible(threshold):
        states = {}
        for i, layer in enumerate(layers):
            nxt = {}
            for j, (a, b) in enumerate(layer):
                if b - a < threshold:
                    continue
                if i == 0:
                    nxt[j] = (j,)
                else:
                    paths = [
                        (*path, j)
                        for k, path in states.items()
                        if overlap(layers[i - 1][k], (a, b))
                    ]
                    if paths:
                        nxt[j] = min(paths)
            states = nxt
        return min(states.values()) if states else None

    lo, hi, best = (0, len(candidates) - 1, None)
    while lo <= hi:
        mid = (lo + hi) // 2
        path = feasible(candidates[mid])
        if path is None:
            hi = mid - 1
        else:
            best = path
            lo = mid + 1
    return best


def measure(payload):
    validated = legacy_measure(payload)
    result = dict(
        schema_version=VERSION,
        policy=POLICY,
        frame_id=payload["sample_id"],
        scope=payload["scope"],
        minimum_passable_ratio=None,
        bottleneck_width_m=None,
        same_section_full_width_m=None,
        raw_minimum_ratio_on_selected_path=None,
        selected_corridor_path=[],
        bottleneck=None,
        reasons=[
            r
            for r in validated["abstention_reasons"]
            if r != "fewer_than_5_sections_draft_policy"
        ],
        authority="none",
        production_eligible=False,
        safety_verdict=None,
        smoothing_applied=False,
        error_rejection_applied=False,
    )
    if not payload["sections"]:
        result["reasons"].append("no_sections")
    if result["reasons"]:
        return result
    rows = payload["sections"]
    layers = [r["free_intervals_m"] for r in rows]
    path = select(layers)
    result["authority"] = "provisional_review"
    if path is None:
        result.update(
            minimum_passable_ratio=0.0,
            bottleneck_width_m=0.0,
            raw_minimum_ratio_on_selected_path=0.0,
            bottleneck={"kind": "known_disconnection_in_sampled_strip"},
        )
        return result
    selected = []
    for i, j in enumerate(path):
        a, b = layers[i][j]
        left, right = rows[i]["structural_span_m"]
        selected.append(
            dict(
                section_index=i,
                s_m=rows[i]["s_m"],
                interval_m=[a, b],
                width_m=b - a,
                full_width_m=right - left,
                ratio=(b - a) / (right - left),
            )
        )
    narrow = min(selected, key=lambda r: (r["width_m"], r["ratio"], r["section_index"]))
    result.update(
        minimum_passable_ratio=narrow["ratio"],
        bottleneck_width_m=narrow["width_m"],
        same_section_full_width_m=narrow["full_width_m"],
        raw_minimum_ratio_on_selected_path=min((r["ratio"] for r in selected)),
        selected_corridor_path=selected,
        bottleneck={
            "kind": "minimum_width_section",
            "section_index": narrow["section_index"],
            "s_m": narrow["s_m"],
        },
    )
    return result


def from_p2(record, spec):
    old = adapt_p2(record, spec)
    spatial = old["spatial_evidence"]
    rows = spatial["sections"]
    payload = dict(
        schema_version="p3_width_draft_v1",
        sample_id=record["metadata"]["frame_id"],
        scope="partial",
        upstream_contract_version=record["schema_version"],
        frame_kind="common_straight_strip_st",
        station_spacing_m=record["config"]["resolution_m"],
        evidence_mode="surface_candidate_review",
        sections=rows,
    )
    result = measure(payload)
    if result["minimum_passable_ratio"] == 0 and spatial["unknown_space_present"]:
        for k in (
            "minimum_passable_ratio",
            "bottleneck_width_m",
            "same_section_full_width_m",
            "raw_minimum_ratio_on_selected_path",
        ):
            result[k] = None
        result.update(authority="none", selected_corridor_path=[], bottleneck=None)
        result["reasons"].append("candidate_disconnected_with_unknown_space")
    result.update(
        input_sha256=record["metadata"]["input_sha256"],
        spatial_evidence=deepcopy(spatial),
        geometric_parameters=deepcopy(spec),
    )
    return result
