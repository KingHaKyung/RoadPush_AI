from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from roadpush_ai.phase7b.handoff import Phase7AAdapter, Phase7BHandoff
from roadpush_ai.phase7b.corridors import (
    ConnectedCorridorTopology,
    CorridorNode,
    track_connected_corridors,
)
from roadpush_ai.phase7b.occupancy_builder import (
    ExistingOccupancyEvidenceIndex,
    Phase7BOccupancyResult,
    build_measurement_occupancy,
)
from roadpush_ai.phase7b.rows import (
    Phase7BRowSelection,
    determine_eligible_rows,
)


@dataclass(frozen=True)
class Phase7BRelativeMetrics:
    schema_version: str
    sample_id: str
    measurement_scope: str
    measurement_confidence: str
    representative_relative_metrics_available: bool
    minimum_passable_ratio: float | None
    raw_minimum_passable_ratio: float | None
    maximum_obstructed_width_ratio: float | None
    obstacle_area_ratio: float | None
    selected_corridor_path: tuple[dict[str, Any], ...]
    bottleneck: dict[str, Any] | None
    maximum_obstruction: dict[str, Any] | None
    abstention_reasons: tuple[str, ...]
    flags: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _median3(values: Sequence[Fraction]) -> Fraction:
    if len(values) != 3:
        raise ValueError("three-row median requires exactly three values")
    return sorted(values)[1]


def _node_ratios(
    topology: ConnectedCorridorTopology,
    rows: Phase7BRowSelection,
) -> dict[str, Fraction]:
    ratios: dict[str, Fraction] = {}
    for node in topology.nodes:
        span = rows.structural_spans[node.y]
        span_width = span[1] - span[0] + 1
        ratios[node.node_id] = Fraction(node.width_px, span_width)
    return ratios


def _path_key(path: tuple[str, ...], nodes: dict[str, CorridorNode]) -> tuple:
    return tuple((nodes[node_id].interval, node_id) for node_id in path)


def _threshold_path(
    topology: ConnectedCorridorTopology,
    ratios: dict[str, Fraction],
    *,
    robust_threshold: Fraction,
    raw_threshold: Fraction,
    return_path: bool,
) -> tuple[str, ...] | None:
    """Exact reachability with the last two robust-threshold booleans as state."""

    nodes = {node.node_id: node for node in topology.nodes}
    layers: dict[int, list[CorridorNode]] = {}
    for node in topology.nodes:
        layers.setdefault(node.y, []).append(node)
    states: dict[tuple[str, tuple[bool, ...]], tuple[str, ...] | None] = {}
    for layer_index, y in enumerate(sorted(layers)):
        next_states: dict[tuple[str, tuple[bool, ...]], tuple[str, ...] | None] = {}
        for node in layers[y]:
            ratio = ratios[node.node_id]
            if ratio < raw_threshold:
                continue
            current_high = ratio >= robust_threshold
            if layer_index == 0:
                key = (node.node_id, (current_high,))
                next_states[key] = (node.node_id,) if return_path else None
                continue
            for (previous_id, history), previous_path in states.items():
                if previous_id not in node.predecessor_ids:
                    continue
                triple = (*history, current_high)
                if len(triple) >= 3 and sum(triple[-3:]) < 2:
                    continue
                next_history = tuple(triple[-2:])
                key = (node.node_id, next_history)
                candidate = (
                    (*previous_path, node.node_id)
                    if return_path and previous_path is not None
                    else None
                )
                if key not in next_states:
                    next_states[key] = candidate
                elif return_path and candidate is not None:
                    incumbent = next_states[key]
                    if incumbent is None or _path_key(candidate, nodes) < _path_key(
                        incumbent, nodes
                    ):
                        next_states[key] = candidate
        states = next_states
        if not states:
            return None

    terminal_states = [
        path
        for (node_id, _history), path in states.items()
        if node_id in topology.full_run_terminal_ids
    ]
    if not terminal_states:
        return None
    if not return_path:
        return ()
    paths = [path for path in terminal_states if path is not None]
    return min(paths, key=lambda path: _path_key(path, nodes)) if paths else None


def _maximum_feasible_threshold(
    candidates: Sequence[Fraction],
    feasible,
) -> Fraction:
    ordered = sorted(set(candidates))
    if not ordered:
        raise ValueError("cannot select a threshold without corridor nodes")
    low, high = 0, len(ordered) - 1
    best = ordered[0]
    while low <= high:
        middle = (low + high) // 2
        value = ordered[middle]
        if feasible(value):
            best = value
            low = middle + 1
        else:
            high = middle - 1
    return best


def _rolling_medians(
    values: Sequence[Fraction], ys: Sequence[int]
) -> list[tuple[tuple[int, int, int], Fraction]]:
    return [
        ((int(ys[index]), int(ys[index + 1]), int(ys[index + 2])), _median3(values[index : index + 3]))
        for index in range(len(values) - 2)
    ]


def _lateral_label(value: float) -> str:
    if value < 0.33:
        return "left"
    if value <= 0.67:
        return "center"
    return "right"


def _obstruction_metrics(
    handoff: Phase7BHandoff,
    occupancy: Phase7BOccupancyResult,
    rows: Phase7BRowSelection,
) -> tuple[float, dict[str, Any], float]:
    structural = np.asarray(handoff.structural_mask, dtype=bool)
    occupied = np.asarray(occupancy.occupancy_mask, dtype=bool) & structural
    run_ys = list(rows.measurement_run_rows)
    obstruction_values: list[Fraction] = []
    for y in run_ys:
        left, right = rows.structural_spans[y]
        span_width = right - left + 1
        obstruction_values.append(
            Fraction(int(occupied[y, left : right + 1].sum()), span_width)
        )
    medians = _rolling_medians(obstruction_values, run_ys)
    maximum_band, maximum_value = max(medians, key=lambda item: (item[1], -item[0][0]))
    center_index = run_ys.index(maximum_band[1])
    maximum = {
        "row_band": list(maximum_band),
        "center_row": maximum_band[1],
        "row_ratios": [
            float(value)
            for value in obstruction_values[center_index - 1 : center_index + 2]
        ],
        "three_row_median_ratio": float(maximum_value),
    }
    start, end = handoff.authorized_row_range or (0, structural.shape[0] - 1)
    authorized = np.zeros_like(structural)
    authorized[start : end + 1] = structural[start : end + 1]
    area_ratio = float((occupied & authorized).sum()) / max(1, int(authorized.sum()))
    return float(maximum_value), maximum, area_ratio


def _null_metrics(
    handoff: Phase7BHandoff,
    rows: Phase7BRowSelection,
) -> Phase7BRelativeMetrics:
    return Phase7BRelativeMetrics(
        schema_version="phase7_b_relative_metrics_v1",
        sample_id=handoff.sample_id,
        measurement_scope=handoff.measurement_scope,
        measurement_confidence="insufficient",
        representative_relative_metrics_available=False,
        minimum_passable_ratio=None,
        raw_minimum_passable_ratio=None,
        maximum_obstructed_width_ratio=None,
        obstacle_area_ratio=None,
        selected_corridor_path=(),
        bottleneck=None,
        maximum_obstruction=None,
        abstention_reasons=rows.abstention_reasons
        or ("corridor_measurement_not_authorized",),
        flags=("relative_metrics_null",),
    )


def calculate_relative_metrics(
    handoff: Phase7BHandoff,
    occupancy: Phase7BOccupancyResult,
    rows: Phase7BRowSelection,
    topology: ConnectedCorridorTopology,
) -> Phase7BRelativeMetrics:
    """Select the exact maximin path and compute Step-5 relative metrics."""

    if (
        not rows.representative_measurement_allowed
        or topology.topology_status == "abstained"
    ):
        return _null_metrics(handoff, rows)
    if handoff.structural_mask is None or occupancy.occupancy_mask is None:
        raise ValueError("authorized structural and occupancy masks are required")
    if not (
        handoff.sample_id == occupancy.sample_id == rows.sample_id == topology.sample_id
    ):
        raise ValueError("sample_id mismatch across Phase 7-B inputs")

    maximum_obstruction, maximum_obstruction_evidence, area_ratio = _obstruction_metrics(
        handoff, occupancy, rows
    )
    base_flags = list(topology.flags)
    if topology.topology_status == "closed":
        y = topology.first_disconnected_row
        bottleneck = {
            "kind": "reliable_geometric_closure",
            "row": y,
            "row_band": [y, y] if y is not None else None,
            "selected_interval": None,
            "structural_span": (
                list(rows.structural_spans[y]) if y in rows.structural_spans else None
            ),
            "lateral_position": None,
            "lateral_label": None,
            "cause": "no_adjacent_overlap_or_no_free_interval",
            "confidence": "reliable",
        }
        base_flags.append("reliable_geometric_closure")
        return Phase7BRelativeMetrics(
            schema_version="phase7_b_relative_metrics_v1",
            sample_id=handoff.sample_id,
            measurement_scope=handoff.measurement_scope,
            measurement_confidence="reliable",
            representative_relative_metrics_available=True,
            minimum_passable_ratio=0.0,
            raw_minimum_passable_ratio=0.0,
            maximum_obstructed_width_ratio=maximum_obstruction,
            obstacle_area_ratio=area_ratio,
            selected_corridor_path=(),
            bottleneck=bottleneck,
            maximum_obstruction=maximum_obstruction_evidence,
            abstention_reasons=(),
            flags=tuple(dict.fromkeys(base_flags)),
        )
    if topology.topology_status != "connected":
        raise ValueError(f"unsupported topology status: {topology.topology_status}")

    nodes = {node.node_id: node for node in topology.nodes}
    ratios = _node_ratios(topology, rows)
    candidates = tuple(ratios.values())
    robust = _maximum_feasible_threshold(
        candidates,
        lambda threshold: _threshold_path(
            topology,
            ratios,
            robust_threshold=threshold,
            raw_threshold=Fraction(0, 1),
            return_path=False,
        )
        is not None,
    )
    raw = _maximum_feasible_threshold(
        candidates,
        lambda threshold: _threshold_path(
            topology,
            ratios,
            robust_threshold=robust,
            raw_threshold=threshold,
            return_path=False,
        )
        is not None,
    )
    selected_ids = _threshold_path(
        topology,
        ratios,
        robust_threshold=robust,
        raw_threshold=raw,
        return_path=True,
    )
    if selected_ids is None:
        raise AssertionError("feasible exact thresholds did not reconstruct a path")
    selected_nodes = [nodes[node_id] for node_id in selected_ids]
    selected_ratios = [ratios[node_id] for node_id in selected_ids]
    selected_ys = [node.y for node in selected_nodes]
    robust_bands = _rolling_medians(selected_ratios, selected_ys)
    bottleneck_band, robust_check = min(
        robust_bands, key=lambda item: (item[1], item[0][0])
    )
    raw_check = min(selected_ratios)
    if robust_check != robust or raw_check != raw:
        raise AssertionError("exact path thresholds disagree with reconstructed metrics")
    center_y = bottleneck_band[1]
    center_node = next(node for node in selected_nodes if node.y == center_y)
    span = rows.structural_spans[center_y]
    center_x = (center_node.interval[0] + center_node.interval[1]) / 2.0
    lateral = float((center_x - span[0]) / max(1, span[1] - span[0]))
    structural = np.asarray(handoff.structural_mask, dtype=bool)
    occupied = np.asarray(occupancy.occupancy_mask, dtype=bool) & structural
    band_has_occupancy = any(
        bool(occupied[y, rows.structural_spans[y][0] : rows.structural_spans[y][1] + 1].any())
        for y in bottleneck_band
    )
    band_indices = [selected_ys.index(y) for y in bottleneck_band]
    bottleneck = {
        "kind": "three_row_robust_bottleneck",
        "row_band": list(bottleneck_band),
        "center_row": center_y,
        "selected_interval": list(center_node.interval),
        "selected_intervals": [
            list(selected_nodes[index].interval) for index in band_indices
        ],
        "structural_span": list(span),
        "row_passable_ratios": [
            float(selected_ratios[index]) for index in band_indices
        ],
        "three_row_median_ratio": float(robust),
        "raw_minimum_row": selected_ys[selected_ratios.index(raw_check)],
        "raw_minimum_ratio": float(raw),
        "lateral_position": lateral,
        "lateral_label": _lateral_label(lateral),
        "cause": "recognized_obstacle" if band_has_occupancy else "structural_or_unrecognized",
        "confidence": "reliable",
    }
    selected_path = tuple(
        {
            "node_id": node.node_id,
            "y": node.y,
            "interval": list(node.interval),
            "width_px": node.width_px,
            "structural_span": list(rows.structural_spans[node.y]),
            "passable_ratio": float(ratios[node.node_id]),
        }
        for node in selected_nodes
    )
    if tuple(selected_ids) != topology.diagnostic_canonical_path:
        base_flags.append("metric_path_differs_from_leftmost_topology_witness")
    if len(topology.full_run_terminal_ids) > 1 or topology.split_node_ids:
        base_flags.append("multiple_corridor_histories_evaluated")
    if raw < robust:
        base_flags.append("raw_single_row_minimum_suppressed_by_three_row_rule")
    return Phase7BRelativeMetrics(
        schema_version="phase7_b_relative_metrics_v1",
        sample_id=handoff.sample_id,
        measurement_scope=handoff.measurement_scope,
        measurement_confidence="reliable",
        representative_relative_metrics_available=True,
        minimum_passable_ratio=float(robust),
        raw_minimum_passable_ratio=float(raw),
        maximum_obstructed_width_ratio=maximum_obstruction,
        obstacle_area_ratio=area_ratio,
        selected_corridor_path=selected_path,
        bottleneck=bottleneck,
        maximum_obstruction=maximum_obstruction_evidence,
        abstention_reasons=(),
        flags=tuple(dict.fromkeys(base_flags)),
    )


def materialize_focused_metric_audit(output: Path, sample_ids: Sequence[str]) -> Path:
    """Write Step-5 relative metrics only; absolute widths/states remain deferred."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metric audit: {output}")
    output.mkdir(parents=True)
    adapter = Phase7AAdapter()
    evidence = ExistingOccupancyEvidenceIndex()
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        authoritative = adapter.authoritative_record(sample_id)
        handoff = adapter.reconstruct(sample_id)
        occupancy = build_measurement_occupancy(
            handoff,
            v6_final_disposition=authoritative["v6_final_disposition"],
            v6_subclassification=authoritative["v6_subclassification"],
            evidence_index=evidence,
        )
        rows = determine_eligible_rows(handoff, occupancy)
        topology = track_connected_corridors(handoff, occupancy, rows)
        metrics = calculate_relative_metrics(handoff, occupancy, rows, topology)
        record = metrics.to_record()
        record["phase7a_status"] = handoff.phase7a_status
        record["topology_status"] = topology.topology_status
        record["absolute_widths_computed"] = False
        record["passability_state_computed"] = False
        records.append(record)
    destination = output / "results.json"
    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    arguments = parser.parse_args()
    print(
        materialize_focused_metric_audit(
            arguments.output, tuple(dict.fromkeys(arguments.sample_id))
        ).resolve()
    )
