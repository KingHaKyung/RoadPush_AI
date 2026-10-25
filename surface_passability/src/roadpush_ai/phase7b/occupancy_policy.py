from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import cv2
import numpy as np


Confidence = Literal["reliable", "low", "insufficient"]


@dataclass(frozen=True)
class PassabilityPolicy:
    """Candidate Phase-7 policy. Freeze only after the fixed visual review."""

    policy_id: str = "phase7_passability_candidate_v1"
    minimum_structural_span_px: int = 12
    minimum_structural_span_fraction: float = 0.02
    vertical_majority_rows: int = 3
    bottleneck_band_rows: int = 3
    obstruction_band_rows: int = 3
    minimum_corridor_support_rows: int = 5
    minimum_reliable_row_fraction: float = 0.25
    maximum_excluded_row_fraction: float = 0.75
    low_ratio_flag_threshold: float = 0.20
    bottleneck_obstacle_row_tolerance: int = 2


@dataclass(frozen=True)
class PassabilityMeasurement:
    selected_surface: str | None
    selection_state: str
    minimum_passable_ratio: float | None
    maximum_obstructed_width_ratio: float | None
    obstacle_area_ratio: float | None
    bottleneck: dict[str, Any] | None
    obstacles: list[dict[str, Any]]
    surface_width_m: float | None
    absolute_passable_width_m: float | None
    agent: dict[str, Any]
    required_passable_ratio: float | None
    required_safe_width_m: float | None
    required_safe_ratio: float | None
    passability_state: str
    measurement: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label_lateral(value: float) -> str:
    if value < 0.33:
        return "left"
    if value <= 0.67:
        return "center"
    return "right"


def _label_longitudinal(value: float) -> str:
    if value < 0.33:
        return "far"
    if value < 0.67:
        return "middle"
    return "near"


def _intervals(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    cuts = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(xs) - 1]
    return [(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends)]


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), mask.astype(bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = labels == index
    return main, mask.astype(bool) & ~main


def _vertical_majority(mask: np.ndarray, rows: int) -> np.ndarray:
    if rows <= 1:
        return mask.astype(bool).copy()
    if rows % 2 != 1:
        raise ValueError("vertical majority window must be odd")
    radius = rows // 2
    padded = np.pad(mask.astype(np.uint8), ((radius, radius), (0, 0)))
    votes = np.zeros(mask.shape, dtype=np.uint16)
    for offset in range(rows):
        votes += padded[offset : offset + mask.shape[0]]
    return votes >= (radius + 1)


def _rolling_median(values: list[float], rows: int) -> tuple[list[float], list[int]]:
    if not values:
        return [], []
    radius = rows // 2
    medians: list[float] = []
    centers: list[int] = []
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        if hi - lo < min(rows, len(values)):
            continue
        medians.append(float(np.median(values[lo:hi])))
        centers.append(index)
    if not medians:
        return [float(np.median(values))], [len(values) // 2]
    return medians, centers


def _row_reliability_mask(
    height: int,
    structural: np.ndarray,
    row_reliability: list[dict[str, Any]] | dict[int, Any] | np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if isinstance(row_reliability, np.ndarray):
        if row_reliability.shape != (height,):
            raise ValueError("row reliability array must match image height")
        return row_reliability.astype(bool), "provided_dense"
    if isinstance(row_reliability, dict):
        dense = np.zeros(height, dtype=bool)
        for key, value in row_reliability.items():
            y = int(key)
            if 0 <= y < height:
                dense[y] = value is True or value == "reliable"
        return dense, "provided_mapping"
    if row_reliability:
        samples = sorted(
            (
                int(row["y"]),
                row.get("boundary_reliability") == "reliable"
                and row.get("branch_reliability") != "uncertain",
            )
            for row in row_reliability
            if "y" in row
        )
        if samples:
            sample_y = np.asarray([row[0] for row in samples])
            sample_ok = np.asarray([row[1] for row in samples], dtype=bool)
            dense = np.zeros(height, dtype=bool)
            for y in range(height):
                nearest = int(np.argmin(np.abs(sample_y - y)))
                dense[y] = bool(sample_ok[nearest])
            return dense, "provided_sampled_nearest_band"

    # Sidewalk's frozen JSON intentionally omits row records. Reconstruct only
    # the observable completeness condition from the saved main structural mask.
    dense = np.zeros(height, dtype=bool)
    width = structural.shape[1]
    for y in range(height):
        xs = np.flatnonzero(structural[y])
        dense[y] = bool(len(xs) and xs[0] > 0 and xs[-1] < width - 1)
    return dense, "derived_saved_mask_frame_completeness"


def _longest_true_run(mask: np.ndarray) -> list[int]:
    best: list[int] = []
    current: list[int] = []
    for y, value in enumerate(mask.tolist()):
        if value:
            current.append(y)
            if len(current) > len(best):
                best = current.copy()
        else:
            current = []
    return best


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _connected_corridor_paths(
    row_records: list[dict[str, Any]],
    policy: PassabilityPolicy,
) -> list[dict[str, Any]]:
    if not row_records or not row_records[0]["intervals"]:
        return []
    active = [
        {
            "path": [(row_records[0]["y"], interval)],
            "ratios": [
                (interval[1] - interval[0] + 1) / row_records[0]["span_width"]
            ],
        }
        for interval in row_records[0]["intervals"]
    ]
    for row in row_records[1:]:
        extended: list[dict[str, Any]] = []
        for interval in row["intervals"]:
            parents = [track for track in active if _overlap(track["path"][-1][1], interval)]
            for track in parents:
                extended.append(
                    {
                        "path": track["path"] + [(row["y"], interval)],
                        "ratios": track["ratios"]
                        + [(interval[1] - interval[0] + 1) / row["span_width"]],
                    }
                )
        if not extended:
            return []
        # Merges can otherwise grow exponentially. Preserve the best distinct
        # histories for each current interval; eight is ample for observed masks.
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for track in extended:
            grouped.setdefault(track["path"][-1][1], []).append(track)
        active = []
        for tracks in grouped.values():
            tracks.sort(key=lambda item: (min(item["ratios"]), np.mean(item["ratios"])), reverse=True)
            active.extend(tracks[:8])

    corridors: list[dict[str, Any]] = []
    for index, track in enumerate(active, 1):
        medians, centers = _rolling_median(track["ratios"], policy.bottleneck_band_rows)
        bottleneck = min(medians)
        center_index = centers[int(np.argmin(medians))]
        y, interval = track["path"][center_index]
        corridors.append(
            {
                "id": index,
                "support_rows": len(track["path"]),
                "bottleneck_ratio": float(bottleneck),
                "bottleneck_row": int(y),
                "bottleneck_interval": [int(interval[0]), int(interval[1])],
                "path": [
                    {"y": int(py), "interval": [int(pi[0]), int(pi[1])]}
                    for py, pi in track["path"]
                ],
            }
        )
    corridors.sort(key=lambda item: (item["bottleneck_ratio"], item["support_rows"]), reverse=True)
    for index, corridor in enumerate(corridors, 1):
        corridor["id"] = index
    return corridors


def _obstacle_records(
    blocker: np.ndarray,
    structural: np.ndarray,
    bottleneck_row: int | None,
    policy: PassabilityPolicy,
    obstacle_geometry: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        blocker.astype(np.uint8), 8
    )
    metadata = obstacle_geometry or []
    records: list[dict[str, Any]] = []
    height, width = blocker.shape
    for component_index in range(1, count):
        component = labels == component_index
        if not component.any():
            continue
        cx, cy = centroids[component_index]
        y = int(round(cy))
        xs = np.flatnonzero(structural[min(max(y, 0), height - 1)])
        if len(xs):
            left, right = int(xs[0]), int(xs[-1])
            lateral = float(np.clip((cx - left) / max(1, right - left), 0.0, 1.0))
        else:
            lateral = float(np.clip(cx / max(1, width - 1), 0.0, 1.0))
        ratios = []
        component_rows = []
        for row_y in np.flatnonzero(component.any(axis=1)):
            surface_xs = np.flatnonzero(structural[row_y])
            if not len(surface_xs):
                continue
            span_width = int(surface_xs[-1] - surface_xs[0] + 1)
            ratios.append(float(component[row_y].sum()) / max(1, span_width))
            component_rows.append(int(row_y))
        medians, _ = _rolling_median(ratios, policy.obstruction_band_rows)
        maximum_ratio = max(medians) if medians else 0.0
        source = metadata[component_index - 1] if component_index - 1 < len(metadata) else {}
        record_id = source.get("id", component_index)
        obstacle_type = (
            source.get("type")
            or source.get("class_name")
            or source.get("semantic_type")
            or "unknown_obstacle"
        )
        causes = bool(
            bottleneck_row is not None
            and component_rows
            and min(abs(row_y - bottleneck_row) for row_y in component_rows)
            <= policy.bottleneck_obstacle_row_tolerance
        )
        longitudinal = float(np.clip(cy / max(1, height - 1), 0.0, 1.0))
        records.append(
            {
                "id": record_id,
                "type": obstacle_type,
                "image_position": {
                    "x": float(np.clip(cx / max(1, width - 1), 0.0, 1.0)),
                    "y": longitudinal,
                },
                "lateral_position": lateral,
                "lateral_label": _label_lateral(lateral),
                "longitudinal_position": longitudinal,
                "longitudinal_label": _label_longitudinal(longitudinal),
                "maximum_obstructed_width_ratio": float(maximum_ratio),
                "causes_bottleneck": causes,
                "pixel_count": int(stats[component_index, cv2.CC_STAT_AREA]),
                "bbox_xywh": [int(value) for value in stats[component_index, :4]],
            }
        )
    return records


def _validate_absolute_inputs(
    surface_width_m: float | None,
    agent_width_m: float | None,
    safety_factor: float,
) -> None:
    if surface_width_m is not None and surface_width_m <= 0:
        raise ValueError("surface_width_m must be > 0")
    if agent_width_m is not None and agent_width_m <= 0:
        raise ValueError("agent_width_m must be > 0")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1.0")


def build_passability_occupancy_mask(
    structural_mask: np.ndarray,
    surface_blocker_mask: np.ndarray,
    passability_instances: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Convert frozen surface blockers into instance-aware occupancy envelopes.

    The accepted surface blocker remains untouched. For measurement only, all
    selected ground-contact pieces belonging to one owned instance are bridged
    across their common lateral extent over the observed contact-row band.
    """

    structural = np.asarray(structural_mask, dtype=bool)
    blocker = np.asarray(surface_blocker_mask, dtype=bool) & structural
    occupancy = np.zeros_like(structural)
    assigned = np.zeros_like(structural)
    records: list[dict[str, Any]] = []
    for instance_id, instance in enumerate(passability_instances, 1):
        selected = np.asarray(instance["selected_mask"], dtype=bool) & structural
        if not selected.any():
            continue
        ys, xs = np.where(selected)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        envelope = np.zeros_like(structural)
        for y in range(y0, y1 + 1):
            surface_xs = np.flatnonzero(structural[y])
            if not len(surface_xs):
                continue
            left = max(x0, int(surface_xs[0]))
            right = min(x1, int(surface_xs[-1]))
            if left <= right:
                envelope[y, left : right + 1] = True
        occupancy |= envelope
        assigned |= selected
        records.append(
            {
                "id": instance_id,
                "type": instance.get("class_name") or "unknown_obstacle",
                "kind": instance.get("kind"),
                "selected_pixels": int(selected.sum()),
                "occupancy_pixels": int(envelope.sum()),
                "contact_bbox_xyxy": [x0, y0, x1, y1],
            }
        )
    # Natural-footprint and other legacy blocker pixels without an exposed
    # instance remain occupied exactly as before rather than disappearing.
    occupancy |= blocker & ~assigned
    occupancy &= structural
    return occupancy, records


def measure_passability(
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray | None,
    free_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    obstacle_geometry: list[dict[str, Any]] | None = None,
    row_reliability: list[dict[str, Any]] | dict[int, Any] | np.ndarray | None = None,
    *,
    surface_width_m: float | None = None,
    agent_width_m: float | None = None,
    safety_factor: float = 1.0,
    agent_type: str | None = None,
    policy: PassabilityPolicy = PassabilityPolicy(),
) -> dict[str, Any]:
    """Measure a maximin connected passage on exactly one selected surface."""

    _validate_absolute_inputs(surface_width_m, agent_width_m, safety_factor)
    uncertain_selection = selected_surface not in {"sidewalk", "road"} or "uncertain" in selection_state
    agent = {
        "type": agent_type,
        "width_m": agent_width_m,
        "safety_factor": float(safety_factor),
    }
    base_measurement = {
        "basis": "observed_pixel_cross_section",
        "physical_width_estimated_from_image": False,
        "route_path_authoritative": False,
        "policy_id": policy.policy_id,
        "policy_status": "candidate_user_review_required",
    }
    if uncertain_selection:
        return PassabilityMeasurement(
            selected_surface=None,
            selection_state="uncertain",
            minimum_passable_ratio=None,
            maximum_obstructed_width_ratio=None,
            obstacle_area_ratio=None,
            bottleneck=None,
            obstacles=[],
            surface_width_m=surface_width_m,
            absolute_passable_width_m=None,
            agent=agent,
            required_passable_ratio=None,
            required_safe_width_m=None,
            required_safe_ratio=None,
            passability_state="uncertain",
            measurement=base_measurement,
            diagnostics={
                "measurement_confidence": "insufficient",
                "uncertain_reason": selection_state,
                "representative_surface_forced": False,
            },
        ).to_dict()

    if structural_mask is None or free_mask is None or blocker_mask is None:
        raise ValueError("selected surface masks are required")
    structural = np.asarray(structural_mask, dtype=bool)
    free = np.asarray(free_mask, dtype=bool)
    blocker = np.asarray(blocker_mask, dtype=bool)
    if structural.ndim != 2 or free.shape != structural.shape or blocker.shape != structural.shape:
        raise ValueError("surface masks must be same-shaped 2D arrays")

    main, fragments = _largest_component(structural)
    free &= main
    blocker &= main
    unavailable = main & ~free
    smoothed_unavailable = _vertical_majority(unavailable, policy.vertical_majority_rows) & main
    measured_free = main & ~smoothed_unavailable
    smoothed_blocker = _vertical_majority(blocker, policy.vertical_majority_rows) & main
    reliable, reliability_source = _row_reliability_mask(
        main.shape[0], main, row_reliability
    )
    minimum_span = max(
        policy.minimum_structural_span_px,
        int(round(main.shape[1] * policy.minimum_structural_span_fraction)),
    )
    present = main.any(axis=1)
    span_ok = np.zeros(main.shape[0], dtype=bool)
    for y in np.flatnonzero(present):
        xs = np.flatnonzero(main[y])
        span_ok[y] = int(xs[-1] - xs[0] + 1) >= minimum_span
    eligible = present & reliable & span_ok
    measurement_rows = _longest_true_run(eligible)

    row_records: list[dict[str, Any]] = []
    for y in measurement_rows:
        xs = np.flatnonzero(main[y])
        left, right = int(xs[0]), int(xs[-1])
        row_records.append(
            {
                "y": int(y),
                "span": [left, right],
                "span_width": right - left + 1,
                "intervals": _intervals(measured_free[y, left : right + 1]),
            }
        )
        row_records[-1]["intervals"] = [
            (a + left, b + left) for a, b in row_records[-1]["intervals"]
        ]

    corridors = _connected_corridor_paths(row_records, policy)
    selected_corridor = corridors[0] if corridors else None
    eligible_count = int(eligible.sum())
    present_count = int(present.sum())
    excluded_count = max(0, present_count - eligible_count)
    reliable_fraction = eligible_count / max(1, present_count)
    excluded_fraction = excluded_count / max(1, present_count)
    support_rows = len(measurement_rows)

    closure_rows = []
    for record in row_records:
        if not record["intervals"]:
            closure_rows.append(record["y"])
    confidence: Confidence
    if support_rows < policy.minimum_corridor_support_rows:
        confidence = "insufficient"
    elif (
        reliable_fraction < policy.minimum_reliable_row_fraction
        or excluded_fraction > policy.maximum_excluded_row_fraction
    ):
        confidence = "low"
    else:
        confidence = "reliable"

    minimum_ratio = (
        float(selected_corridor["bottleneck_ratio"])
        if selected_corridor is not None
        else 0.0 if confidence == "reliable" else None
    )
    bottleneck_row = (
        int(selected_corridor["bottleneck_row"])
        if selected_corridor is not None
        else int(closure_rows[len(closure_rows) // 2]) if closure_rows else None
    )

    obstruction_values: list[float] = []
    obstruction_rows: list[int] = []
    for record in row_records:
        left, right = record["span"]
        obstruction_values.append(
            float(smoothed_blocker[record["y"], left : right + 1].sum())
            / record["span_width"]
        )
        obstruction_rows.append(record["y"])
    obstruction_medians, obstruction_centers = _rolling_median(
        obstruction_values, policy.obstruction_band_rows
    )
    if obstruction_medians:
        obstruction_index = int(np.argmax(obstruction_medians))
        maximum_obstruction = float(obstruction_medians[obstruction_index])
        maximum_obstruction_row = int(
            obstruction_rows[obstruction_centers[obstruction_index]]
        )
    else:
        maximum_obstruction = None
        maximum_obstruction_row = None

    area_ratio = float(blocker.sum()) / max(1, int(main.sum()))
    obstacles = _obstacle_records(
        blocker, main, bottleneck_row, policy, obstacle_geometry
    )
    if bottleneck_row is not None:
        span_xs = np.flatnonzero(main[bottleneck_row])
        span = [int(span_xs[0]), int(span_xs[-1])] if len(span_xs) else None
    else:
        span = None
    if selected_corridor is not None and span is not None:
        interval = selected_corridor["bottleneck_interval"]
        center_x = (interval[0] + interval[1]) / 2.0
        lateral = float((center_x - span[0]) / max(1, span[1] - span[0]))
    elif span is not None:
        interval = None
        lateral = 0.5
    else:
        interval = None
        lateral = 0.5
    longitudinal = (
        float(bottleneck_row / max(1, main.shape[0] - 1))
        if bottleneck_row is not None
        else None
    )
    causes = [row["id"] for row in obstacles if row["causes_bottleneck"]]
    bottleneck = None
    if bottleneck_row is not None:
        bottleneck = {
            "row": bottleneck_row,
            "row_band": [
                max(0, bottleneck_row - policy.bottleneck_band_rows // 2),
                min(main.shape[0] - 1, bottleneck_row + policy.bottleneck_band_rows // 2),
            ],
            "structural_span": span,
            "passable_interval": interval,
            "lateral_position": lateral,
            "lateral_label": _label_lateral(lateral),
            "longitudinal_position": longitudinal,
            "longitudinal_label": _label_longitudinal(longitudinal),
            "support_rows": support_rows,
            "cause": "recognized_obstacle" if causes else "structural_or_unrecognized",
            "obstacle_ids": causes,
            "confidence": confidence,
        }

    absolute = (
        float(surface_width_m * minimum_ratio)
        if surface_width_m is not None and minimum_ratio is not None
        else None
    )
    required_ratio = (
        float(agent_width_m / surface_width_m)
        if surface_width_m is not None and agent_width_m is not None
        else None
    )
    required_safe_width = (
        float(agent_width_m * safety_factor) if agent_width_m is not None else None
    )
    required_safe_ratio = (
        float(required_safe_width / surface_width_m)
        if surface_width_m is not None and required_safe_width is not None
        else None
    )
    if confidence != "reliable":
        state = "uncertain"
    elif selected_corridor is None or minimum_ratio == 0.0:
        state = "blocked"
    elif absolute is None or agent_width_m is None:
        state = "relative_only"
    elif absolute < agent_width_m:
        state = "blocked"
    elif absolute < required_safe_width:
        state = "narrow"
    else:
        state = "passable"

    flags = []
    if confidence != "reliable":
        flags.append(f"measurement_{confidence}_confidence")
    if minimum_ratio is not None and minimum_ratio <= policy.low_ratio_flag_threshold:
        flags.append("extremely_low_minimum_passable_ratio")
    if selected_corridor is None:
        flags.append("no_connected_corridor")
    if any(len(record["intervals"]) > 1 for record in row_records):
        flags.append("split_corridor")
    if excluded_fraction > policy.maximum_excluded_row_fraction:
        flags.append("excessive_row_exclusion")
    if (
        bottleneck_row is not None
        and maximum_obstruction_row is not None
        and abs(bottleneck_row - maximum_obstruction_row) > max(10, main.shape[0] * 0.15)
    ):
        flags.append("bottleneck_obstruction_rows_differ")

    return PassabilityMeasurement(
        selected_surface=selected_surface,
        selection_state="selected",
        minimum_passable_ratio=minimum_ratio,
        maximum_obstructed_width_ratio=maximum_obstruction,
        obstacle_area_ratio=area_ratio,
        bottleneck=bottleneck,
        obstacles=obstacles,
        surface_width_m=surface_width_m,
        absolute_passable_width_m=absolute,
        agent=agent,
        required_passable_ratio=required_ratio,
        required_safe_width_m=required_safe_width,
        required_safe_ratio=required_safe_ratio,
        passability_state=state,
        measurement=base_measurement,
        diagnostics={
            "measurement_confidence": confidence,
            "row_reliability_source": reliability_source,
            "main_structural_pixels": int(main.sum()),
            "excluded_fragment_pixels": int(fragments.sum()),
            "structural_row_count": present_count,
            "eligible_row_count": eligible_count,
            "measurement_run_row_count": support_rows,
            "measurement_row_range": (
                [int(measurement_rows[0]), int(measurement_rows[-1])]
                if measurement_rows
                else None
            ),
            "excluded_row_count": excluded_count,
            "excluded_row_fraction": float(excluded_fraction),
            "minimum_structural_span_px": minimum_span,
            "vertical_smoothing_rows": policy.vertical_majority_rows,
            "corridors": corridors,
            "selected_corridor_id": selected_corridor["id"] if selected_corridor else None,
            "maximum_obstruction_row": maximum_obstruction_row,
            "closure_rows": closure_rows,
            "geometrically_blocked": bool(
                confidence == "reliable" and selected_corridor is None
            ),
            "flags": flags,
        },
    ).to_dict()

