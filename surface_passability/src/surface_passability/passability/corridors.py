"""Build connected free-space corridor topology across eligible rows."""

from __future__ import annotations

import numpy as np

from surface_passability.contracts.passability import (
    ConnectedCorridorTopology,
    CorridorNode,
    EligibleRowSelection,
    MeasurementOccupancyResult,
    PassabilityBoundaryHandoff,
    TopologyStatus,
)
from surface_passability.passability.intervals import (
    intervals_overlap,
    true_intervals,
)


def _abstained_topology(
    handoff: PassabilityBoundaryHandoff,
    rows: EligibleRowSelection,
) -> ConnectedCorridorTopology:
    reasons = rows.abstention_reasons or ("row_measurement_not_authorized",)
    return ConnectedCorridorTopology(
        schema_version="phase7_b_corridor_topology_v1",
        sample_id=handoff.sample_id,
        measurement_scope=handoff.measurement_scope,
        measurement_run_range=None,
        measurement_run_row_count=0,
        topology_status="abstained",
        row_intervals={},
        nodes=(),
        edges=(),
        full_run_terminal_ids=(),
        diagnostic_canonical_path=(),
        split_node_ids=(),
        merge_node_ids=(),
        first_disconnected_row=None,
        abstention_reasons=reasons,
        flags=("corridor_topology_not_measured",),
    )


def track_connected_corridors(
    handoff: PassabilityBoundaryHandoff,
    occupancy: MeasurementOccupancyResult,
    rows: EligibleRowSelection,
) -> ConnectedCorridorTopology:
    """Build the exact adjacent-row overlap DAG for the approved row run."""

    if not rows.representative_measurement_allowed or not rows.measurement_run_rows:
        return _abstained_topology(handoff, rows)
    if handoff.structural_mask is None or occupancy.occupancy_mask is None:
        raise ValueError("authorized structural and occupancy masks are required")
    if handoff.sample_id != occupancy.sample_id or handoff.sample_id != rows.sample_id:
        raise ValueError("sample_id mismatch across Phase 7-B inputs")
    structural = np.asarray(handoff.structural_mask, dtype=bool)
    occupied = np.asarray(occupancy.occupancy_mask, dtype=bool)
    if structural.ndim != 2 or occupied.shape != structural.shape:
        raise ValueError("structural and occupancy masks must be same-shaped 2D arrays")
    expected_run = tuple(
        range(rows.measurement_run_rows[0], rows.measurement_run_rows[-1] + 1)
    )
    if rows.measurement_run_rows != expected_run:
        raise ValueError("measurement run must be consecutive")

    free = structural & ~occupied
    row_intervals: dict[int, tuple[tuple[int, int], ...]] = {}
    mutable_nodes: dict[str, CorridorNode] = {}
    edges: list[tuple[str, str]] = []
    canonical_histories: dict[str, tuple[str, ...]] = {}
    active_ids: tuple[str, ...] = ()
    first_disconnected_row: int | None = None

    for row_offset, y in enumerate(rows.measurement_run_rows):
        span = rows.structural_spans.get(y)
        if span is None:
            raise ValueError(f"row {y} is missing its structural span")
        left, right = span
        all_intervals = true_intervals(free[y, left : right + 1], x_offset=left)
        row_intervals[y] = all_intervals
        next_ids: list[str] = []
        for interval_index, interval in enumerate(all_intervals):
            node_id = f"y{y:06d}_i{interval_index:04d}"
            if row_offset == 0:
                predecessors: tuple[str, ...] = ()
                canonical = (node_id,)
            else:
                predecessors = tuple(
                    previous_id
                    for previous_id in active_ids
                    if intervals_overlap(mutable_nodes[previous_id].interval, interval)
                )
                if not predecessors:
                    continue
                predecessor = min(
                    predecessors,
                    key=lambda value: tuple(
                        mutable_nodes[path_id].interval
                        for path_id in canonical_histories[value]
                    ),
                )
                canonical = (*canonical_histories[predecessor], node_id)
                edges.extend(
                    (predecessor_id, node_id)
                    for predecessor_id in predecessors
                )
            mutable_nodes[node_id] = CorridorNode(
                node_id=node_id,
                y=y,
                interval=interval,
                width_px=interval[1] - interval[0] + 1,
                predecessor_ids=predecessors,
            )
            canonical_histories[node_id] = canonical
            next_ids.append(node_id)
        active_ids = tuple(next_ids)
        if not active_ids:
            first_disconnected_row = y
            break

    final_y = rows.measurement_run_rows[-1]
    terminal_ids = (
        tuple(active_ids)
        if first_disconnected_row is None
        and active_ids
        and all(mutable_nodes[node_id].y == final_y for node_id in active_ids)
        else ()
    )
    if terminal_ids:
        terminal = min(
            terminal_ids,
            key=lambda value: tuple(
                mutable_nodes[path_id].interval
                for path_id in canonical_histories[value]
            ),
        )
        canonical_path = canonical_histories[terminal]
        status: TopologyStatus = "connected"
    else:
        canonical_path = ()
        status = "closed"

    successor_count: dict[str, int] = {}
    for predecessor, _successor in edges:
        successor_count[predecessor] = successor_count.get(predecessor, 0) + 1
    nodes = tuple(mutable_nodes.values())
    split_ids = tuple(
        node.node_id
        for node in nodes
        if successor_count.get(node.node_id, 0) > 1
    )
    merge_ids = tuple(
        node.node_id for node in nodes if len(node.predecessor_ids) > 1
    )
    flags: list[str] = []
    if split_ids:
        flags.append("split_corridor_topology")
    if merge_ids:
        flags.append("merged_corridor_histories_preserved")
    if status == "closed":
        flags.append("no_full_run_connected_corridor")
    return ConnectedCorridorTopology(
        schema_version="phase7_b_corridor_topology_v1",
        sample_id=handoff.sample_id,
        measurement_scope=handoff.measurement_scope,
        measurement_run_range=rows.measurement_run_range,
        measurement_run_row_count=len(rows.measurement_run_rows),
        topology_status=status,
        row_intervals=row_intervals,
        nodes=nodes,
        edges=tuple(edges),
        full_run_terminal_ids=terminal_ids,
        diagnostic_canonical_path=canonical_path,
        split_node_ids=split_ids,
        merge_node_ids=merge_ids,
        first_disconnected_row=first_disconnected_row,
        abstention_reasons=(),
        flags=tuple(flags),
    )
