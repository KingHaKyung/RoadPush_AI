from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from roadpush_ai.phase7b.handoff import Phase7AAdapter, Phase7BHandoff
from roadpush_ai.phase7b.occupancy_builder import (
    ExistingOccupancyEvidenceIndex,
    Phase7BOccupancyResult,
    build_measurement_occupancy,
)
from roadpush_ai.phase7b.rows import (
    Phase7BRowSelection,
    determine_eligible_rows,
)


TopologyStatus = Literal["connected", "closed", "abstained"]


@dataclass(frozen=True)
class CorridorNode:
    node_id: str
    y: int
    interval: tuple[int, int]
    width_px: int
    predecessor_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConnectedCorridorTopology:
    schema_version: str
    sample_id: str
    measurement_scope: str
    measurement_run_range: tuple[int, int] | None
    measurement_run_row_count: int
    topology_status: TopologyStatus
    row_intervals: dict[int, tuple[tuple[int, int], ...]]
    nodes: tuple[CorridorNode, ...]
    edges: tuple[tuple[str, str], ...]
    full_run_terminal_ids: tuple[str, ...]
    diagnostic_canonical_path: tuple[str, ...]
    split_node_ids: tuple[str, ...]
    merge_node_ids: tuple[str, ...]
    first_disconnected_row: int | None
    abstention_reasons: tuple[str, ...]
    flags: tuple[str, ...]

    @property
    def full_run_corridor_exists(self) -> bool:
        return self.topology_status == "connected"

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        if value["measurement_run_range"] is not None:
            value["measurement_run_range"] = list(value["measurement_run_range"])
        value["row_intervals"] = {
            str(y): [list(interval) for interval in intervals]
            for y, intervals in self.row_intervals.items()
        }
        return value


def _intervals(row: np.ndarray, *, x_offset: int = 0) -> tuple[tuple[int, int], ...]:
    xs = np.flatnonzero(row)
    if not len(xs):
        return ()
    cuts = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(xs) - 1]
    return tuple(
        (int(xs[start]) + x_offset, int(xs[end]) + x_offset)
        for start, end in zip(starts, ends)
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _abstained_topology(
    handoff: Phase7BHandoff,
    rows: Phase7BRowSelection,
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
    handoff: Phase7BHandoff,
    occupancy: Phase7BOccupancyResult,
    rows: Phase7BRowSelection,
) -> ConnectedCorridorTopology:
    """Build an exact adjacent-row overlap DAG for the approved run.

    The DAG is the compact, lossless representation of split/merge histories.
    It avoids exponentially enumerating every path. Intervals that cannot be
    reached from the first run row are pruned, which is exact because such an
    interval can never become a full-run corridor.

    `diagnostic_canonical_path` is only a deterministic topology witness. It is
    the lexicographically leftmost full-run history and is not the Phase 7-B
    maximin path; Step 5 performs metric-based path selection.
    """

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
    expected_run = tuple(range(rows.measurement_run_rows[0], rows.measurement_run_rows[-1] + 1))
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
        all_intervals = _intervals(free[y, left : right + 1], x_offset=left)
        row_intervals[y] = all_intervals
        next_ids: list[str] = []
        for interval_index, interval in enumerate(all_intervals):
            node_id = f"y{y:06d}_i{interval_index:04d}"
            if row_offset == 0:
                predecessors: tuple[str, ...] = ()
                canonical = (node_id,)
            else:
                predecessors = tuple(
                    node_id_previous
                    for node_id_previous in active_ids
                    if _overlaps(mutable_nodes[node_id_previous].interval, interval)
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
                edges.extend((predecessor_id, node_id) for predecessor_id in predecessors)
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
        node.node_id for node in nodes if successor_count.get(node.node_id, 0) > 1
    )
    merge_ids = tuple(node.node_id for node in nodes if len(node.predecessor_ids) > 1)
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


def materialize_focused_corridor_audit(output: Path, sample_ids: Sequence[str]) -> Path:
    """Write Step-4 topology only; final robust ratios remain uncomputed."""

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corridor audit: {output}")
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
        record = topology.to_record()
        record["phase7a_status"] = handoff.phase7a_status
        record["row_confidence"] = rows.measurement_confidence
        record["final_robust_metrics_computed"] = False
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
        materialize_focused_corridor_audit(
            arguments.output, tuple(dict.fromkeys(arguments.sample_id))
        ).resolve()
    )
