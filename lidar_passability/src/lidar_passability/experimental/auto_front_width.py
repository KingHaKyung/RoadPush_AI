"""Separate v4-boundary / v5 conditional-width experiment, not anchor width."""

import numpy as np

POLICY = dict(
    station_step_m=0.5,
    lateral_cell_max_m=0.2,
    boundary_support_distance_m=0.75,
    boundary_residual_m=0.18,
    minimum_candidate_width_m=0.6,
    minimum_inner_inlier_fraction=0.3,
    floor_band_m=0.10,
    obstacle_height_min_m=0.15,
    obstacle_height_max_m=2.0,
    minimum_floor_points=5,
    minimum_supported_run_m=1.5,
)


def intervals(mask, edges):
    result = []
    start = None
    for i, value in enumerate([*mask, False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            result.append([float(edges[start]), float(edges[i])])
            start = None
    return result


def widest_path(layers):
    """Maximize the minimum interval width with positive-overlap connectivity."""
    widths = sorted({b - a for layer in layers for a, b in layer})

    def feasible(threshold):
        states = {}
        for i, layer in enumerate(layers):
            nxt = {}
            for j, (a, b) in enumerate(layer):
                if b - a + 1e-10 < threshold:
                    continue
                if i == 0:
                    nxt[j] = (j,)
                else:
                    paths = [
                        (*p, j)
                        for k, p in states.items()
                        if min(b, layers[i - 1][k][1]) - max(a, layers[i - 1][k][0])
                        > 1e-9
                    ]
                    if paths:
                        nxt[j] = min(paths)
            states = nxt
        return min(states.values()) if states else None

    lo, hi, best = 0, len(widths) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        path = feasible(widths[mid])
        if path is None:
            hi = mid - 1
        else:
            best = path
            lo = mid + 1
    return best


def path_metric(rows, key, full_width):
    layers = [r[key] for r in rows]
    path = widest_path(layers)
    if path is None:
        return dict(
            width_m=0.0,
            ratio=0.0,
            path=[],
            interpretation="no_connected_path_in_this_evidence_mask",
        )
    selected = [
        dict(s_m=r["s_m"], interval_m=layer[j], width_m=layer[j][1] - layer[j][0])
        for r, layer, j in zip(rows, layers, path)
    ]
    narrow = min(selected, key=lambda r: r["width_m"])
    return dict(
        width_m=narrow["width_m"],
        ratio=narrow["width_m"] / full_width,
        same_section_full_width_m=full_width,
        bottleneck_s_m=narrow["s_m"],
        path=selected,
    )


def measure_candidate(xyz, labels, raw_ids, frame, candidate, policy=POLICY):
    axis = np.asarray(frame["direction_xy"])
    normal = np.asarray(frame["normal_xy"])
    plane = np.asarray(frame["floor_plane_z_eq_ax_by_c"])
    s = xyz[:, :2] @ axis
    n = xyz[:, :2] @ normal
    h = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
    left, right = sorted([candidate["road_offset_m"], candidate["inner_offset_m"]])
    width = right - left
    result = dict(
        id=candidate["id"],
        kind="sidewalk_candidate",
        full_width_candidate_m=width,
        lateral_span_m=[left, right],
        sections=[],
        runs=[],
        rejected_reasons=[],
        relative_width=None,
        authority="conditional_geometry_review",
        physical_boundary_accuracy_verified=False,
    )
    if width < policy["minimum_candidate_width_m"]:
        result["rejected_reasons"].append("thin_model_strip_candidate")
    if (
        candidate["inner_fit"]["inlier_fraction"]
        < policy["minimum_inner_inlier_fraction"]
    ):
        result["rejected_reasons"].append("weak_inner_boundary_support")
    source = frame["source_points"][candidate["id"]]
    support_s = {}
    for name, offset in [
        ("road", candidate["road_offset_m"]),
        ("inner", candidate["inner_offset_m"]),
    ]:
        p = np.asarray(source[name]).reshape(-1, 2)
        p = p[np.abs(p @ normal - offset) <= policy["boundary_residual_m"]]
        support_s[name] = p @ axis
    if result["rejected_reasons"]:
        return result
    edges = np.linspace(
        left, right, int(np.ceil(width / policy["lateral_cell_max_m"])) + 1
    )
    step = policy["station_step_m"]
    fov_lo, fov_hi = candidate["common_fov_visible_s_interval_m"]
    # Entire station slab must be in the common FOV interval, not just its centre.
    stations = np.arange(
        np.ceil((fov_lo + step / 2) / step) * step, fov_hi - step / 2 + 1e-9, step
    )
    for station in stations:
        supported = all(
            len(v)
            and np.min(np.abs(v - station)) <= policy["boundary_support_distance_m"]
            for v in support_s.values()
        )
        row = dict(
            s_m=float(station),
            both_boundary_support=bool(supported),
            status="boundary_not_supported",
            observed_floor_intervals_m=[],
            not_detected_blocked_intervals_m=[],
            obstacle_intervals_m=[],
            unknown_intervals_m=[],
        )
        result["sections"].append(row)
        if not supported:
            continue
        ids = np.flatnonzero((abs(s - station) < step / 2) & (n >= left) & (n < right))
        floor_ids = ids[(labels[ids] == 48) & (h[ids] >= -0.25) & (h[ids] <= 0.5)]
        if len(floor_ids) < policy["minimum_floor_points"]:
            row["status"] = "insufficient_local_floor"
            continue
        # Adapt the road-plane reference to the sidewalk's local elevation.
        floor = float(np.quantile(h[floor_ids], 0.2))
        dz = h[ids] - floor
        bins = np.clip(
            np.searchsorted(edges, n[ids], side="right") - 1, 0, len(edges) - 2
        )
        ground = np.zeros(len(edges) - 1, bool)
        blocked = ground.copy()
        ground_point = (labels[ids] == 48) & (abs(dz) <= policy["floor_band_m"])
        obstacle_point = (dz >= policy["obstacle_height_min_m"]) & (
            dz <= policy["obstacle_height_max_m"]
        )
        ground[bins[ground_point]] = True
        blocked[bins[obstacle_point]] = True
        # Overlapping floor and elevated return stays blocked, regardless of label.
        observed = ground & ~blocked
        unknown = ~ground & ~blocked
        row.update(
            status="conditional_observed_section",
            local_floor_above_road_m=floor,
            observed_floor_intervals_m=intervals(observed, edges),
            not_detected_blocked_intervals_m=intervals(~blocked, edges),
            obstacle_intervals_m=intervals(blocked, edges),
            unknown_intervals_m=intervals(unknown, edges),
            unknown_width_m=float(np.diff(edges)[unknown].sum()),
            observed_floor_width_m=float(np.diff(edges)[observed].sum()),
            obstacle_raw_ids=raw_ids[ids[obstacle_point]].tolist(),
            floor_raw_ids=raw_ids[ids[ground_point]].tolist(),
        )
    chunks = []
    current = []
    for row in result["sections"]:
        if row["status"] == "conditional_observed_section":
            current.append(row)
        else:
            if current:
                chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    for chunk in chunks:
        length = len(chunk) * step
        if length + 1e-9 < policy["minimum_supported_run_m"]:
            continue
        observed = path_metric(chunk, "observed_floor_intervals_m", width)
        permissive = path_metric(chunk, "not_detected_blocked_intervals_m", width)
        unknown = any(r["unknown_width_m"] > 1e-9 for r in chunk)
        identifiable = abs(observed["width_m"] - permissive["width_m"]) <= 1e-9
        run = dict(
            s_interval_m=[chunk[0]["s_m"] - step / 2, chunk[-1]["s_m"] + step / 2],
            length_m=length,
            sections=len(chunk),
            unknown_space_present=unknown,
            observed_floor_path=observed,
            unknown_permitted_path=permissive,
            conditional_relative_width=observed["ratio"] if identifiable else None,
            conditional_bottleneck_width_m=(
                observed["width_m"] if identifiable else None
            ),
            evidence_masks_agree=identifiable,
            parameter_robustness_verified=False,
            physical_clearance_verified=False,
            interpretation="evidence-mask sensitivity; not physical lower/upper bounds",
            no_path_is_not_physical_blockage=True,
        )
        result["runs"].append(run)
    if not result["runs"]:
        result["rejected_reasons"].append("no_contiguous_supported_measurement_run")
    return result


def postprocess(xyz, labels, config, policy=POLICY):
    """End-to-end automatic CPU entry: original PCD coordinates + saved labels.

    No old boundary file or reviewer field is read by this entry point.
    Labels follow the existing 40=road / 48=sidewalk class contract.
    """
    from .auto_front_boundaries import detect
    from .fixed_front_roi import apply_fixed_front_roi

    xyz = np.asarray(xyz, dtype=float)
    labels = np.asarray(labels)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or labels.shape != (len(xyz),):
        raise ValueError("expected Nx3 xyz and N class IDs")
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite coordinate")
    try:
        detected = detect(xyz, labels, config)
    except ValueError as error:
        return dict(
            schema="roadpush.auto_front_width.v5",
            status="automatic_abstention",
            candidates=[],
            relative_width=None,
            reasons=[str(error)],
            whole_road_width_branch="unresolved_structural_road_boundaries",
        )
    frame = dict(
        direction_xy=detected["axis"].tolist(),
        normal_xy=detected["normal"].tolist(),
        floor_plane_z_eq_ax_by_c=detected["plane"].tolist(),
        source_points=detected["sources"],
    )
    mask, _ = apply_fixed_front_roi(xyz, config)
    mask &= labels >= 0
    raw_ids = np.flatnonzero(mask)
    candidates = [
        measure_candidate(xyz[mask], labels[mask], raw_ids, frame, c, policy)
        for c in detected["candidates"]
    ]
    return dict(
        schema="roadpush.auto_front_width.v5",
        status=(
            "conditional_evidence"
            if any(c["runs"] for c in candidates)
            else "automatic_abstention"
        ),
        geometry=frame,
        automatic_boundary_candidates=detected["candidates"],
        candidates=candidates,
        relative_width=None,
        full_60m_relative_width=None,
        policy=policy,
        whole_road_width_branch="unresolved_structural_road_boundaries",
        fully_automatic_execution=True,
        user_annotations_used=False,
        rgb_pixels_used=False,
    )
