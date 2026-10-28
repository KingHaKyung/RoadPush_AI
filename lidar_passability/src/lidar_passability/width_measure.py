"""Review-only station/interval prototype. No PCD, learned model or safety verdict.

Intervals are continuous metres in one common (s,t) strip frame. This is an
independent implementation of the photo metric's optimization, not a drop-in
photo pipeline. See CONTRACT.md for important geometric differences.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import median

VERSION = "p3_width_draft_v1"


def number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or (not math.isfinite(value))
    ):
        raise ValueError("expected finite number, excluding bool")
    return float(value)


def overlap(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def objective(ratios):
    return (
        min((median(ratios[i : i + 3]) for i in range(len(ratios) - 2))),
        min(ratios),
    )


def select_path(layers, denominators):
    """Exact layered reachability with 2-bit history, then raw tie-break.

    A median >= t iff at least two values of its triple are >= t. Keeping
    only one locally best prefix is incorrect at a split/merge; retain the
    last two threshold booleans instead. Complexity is polynomial in nodes.
    """
    ratios = [[(b - a) / d for a, b in layer] for layer, d in zip(layers, denominators)]
    candidates = sorted({v for layer in ratios for v in layer})
    if not candidates:
        return None

    def feasible(robust, raw):
        states = {}
        for i, layer in enumerate(layers):
            nxt = {}
            for j, interval in enumerate(layer):
                value = ratios[i][j]
                if value < raw:
                    continue
                high = value >= robust
                if i == 0:
                    nxt[j, (high,)] = (j,)
                    continue
                for (prev, hist), path in states.items():
                    if not overlap(layers[i - 1][prev], interval):
                        continue
                    triple = (*hist, high)
                    if len(triple) >= 3 and sum(triple[-3:]) < 2:
                        continue
                    key = (j, triple[-2:])
                    candidate = (*path, j)
                    if key not in nxt or candidate < nxt[key]:
                        nxt[key] = candidate
            states = nxt
            if not states:
                return None
        return min(states.values()) if states else None

    def maximum(predicate):
        lo, hi, best = (0, len(candidates) - 1, None)
        while lo <= hi:
            mid = (lo + hi) // 2
            if predicate(candidates[mid]):
                best = candidates[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    robust = maximum(lambda t: feasible(t, 0) is not None)
    if robust is None:
        return None
    raw = maximum(lambda t: feasible(robust, t) is not None)
    return feasible(robust, raw)


def measure(payload):
    if payload.get("schema_version") != VERSION:
        raise ValueError("unsupported input schema")
    if payload.get("frame_kind") != "common_straight_strip_st":
        raise ValueError("only common straight strip frame supported")
    if payload.get("scope") not in ("full", "partial", "provisional", "none"):
        raise ValueError("invalid scope")
    mode = payload.get("evidence_mode", "supported_interval_fixture")
    if mode not in ("supported_interval_fixture", "surface_candidate_review"):
        raise ValueError("unsupported evidence mode")
    spacing = number(payload["station_spacing_m"])
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    rows = payload["sections"]
    result = dict(
        schema_version=VERSION,
        sample_id=payload["sample_id"],
        upstream_contract_version=payload["upstream_contract_version"],
        metric_definition="max_path_min_rolling_median3_ratio",
        measurement_scope=payload["scope"],
        result_authority="none",
        human_review_required=True,
        production_eligible=False,
        topology_status="unknown",
        evidence_mode=mode,
        clearance_volume_verified=False,
        measurement_confidence="low",
        minimum_passable_ratio=None,
        raw_minimum_passable_ratio=None,
        raw_minimum_width_m_on_selected_path=None,
        robust_width_m_on_selected_path=None,
        absolute_passable_width_m=None,
        selected_corridor_path=[],
        bottleneck=None,
        coverage=0.0,
        measurement_coverage=0.0,
        abstention_reasons=[],
    )
    reasons = []
    if payload["scope"] == "none":
        reasons.append("scope_none")
    if len(rows) < 5:
        reasons.append("fewer_than_5_sections_draft_policy")
    layers, denominators = ([], [])
    known = 0
    previous_s = None
    for row in rows:
        s = number(row["s_m"])
        if previous_s is not None and (
            not math.isclose(s - previous_s, spacing, rel_tol=1e-08, abs_tol=1e-08)
        ):
            raise ValueError("missing/irregular station; no interpolation allowed")
        previous_s = s
        span = row.get("structural_span_m")
        if span is None:
            reasons.append("full_width_unobserved")
            layers.append([])
            denominators.append(1.0)
            continue
        left, right = map(number, span)
        if right <= left:
            raise ValueError("denominator must be positive")
        supported = (
            row.get("candidate_evidence_supported") is True
            if mode == "surface_candidate_review"
            else row.get("occupancy_observed") is True
        )
        if row.get("boundary_supported") is not True or not supported:
            reasons.append("boundary_or_occupancy_unverified")
        free = row.get("free_intervals_m")
        if free is None:
            reasons.append("free_space_unknown")
            free = []
        elif row.get("boundary_supported") is True and supported:
            known += 1
        layer = []
        for a, b in free:
            a, b = (number(a), number(b))
            if not left <= a < b <= right:
                raise ValueError("free interval outside structural span or nonpositive")
            if layer and a <= layer[-1][1]:
                raise ValueError("free intervals must be sorted, disjoint and maximal")
            layer.append((a, b))
        layers.append(layer)
        denominators.append(right - left)
    result["coverage"] = known / len(rows) if rows else 0.0
    if reasons:
        result["abstention_reasons"] = sorted(set(reasons))
        return result
    result["result_authority"] = "provisional_review"
    result["measurement_coverage"] = 1.0
    path = select_path(layers, denominators)
    if path is None:
        result.update(
            topology_status="disconnected_in_sampled_strip",
            minimum_passable_ratio=0.0,
            raw_minimum_passable_ratio=0.0,
            raw_minimum_width_m_on_selected_path=0.0,
            robust_width_m_on_selected_path=0.0,
            bottleneck={"kind": "sampled_strip_no_full_run_path"},
        )
        return result
    selected = []
    for i, j in enumerate(path):
        a, b = layers[i][j]
        selected.append(
            dict(
                section_index=i,
                s_m=rows[i]["s_m"],
                interval_m=[a, b],
                full_width_m=denominators[i],
                free_width_m=b - a,
                passable_ratio=(b - a) / denominators[i],
            )
        )
    ratios = [x["passable_ratio"] for x in selected]
    widths = [x["free_width_m"] for x in selected]
    robust, raw = objective(ratios)
    band = min(range(len(rows) - 2), key=lambda i: (median(ratios[i : i + 3]), i))
    result.update(
        topology_status="connected_in_sampled_strip",
        minimum_passable_ratio=robust,
        raw_minimum_passable_ratio=raw,
        raw_minimum_width_m_on_selected_path=min(widths),
        robust_width_m_on_selected_path=objective(widths)[0],
        selected_corridor_path=selected,
        bottleneck=dict(
            kind="three_section_robust_ratio",
            section_indices=list(range(band, band + 3)),
            ratios=ratios[band : band + 3],
            raw_section_index=ratios.index(raw),
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            measure(json.loads(args.input.read_text(encoding="utf-8"))),
            indent=2,
            allow_nan=False,
        )
    )
