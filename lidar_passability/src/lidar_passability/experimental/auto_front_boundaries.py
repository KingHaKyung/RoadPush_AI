"""Experimental v4 boundary detector; direction_override supports anchor review."""

import numpy as np
from scipy.spatial import cKDTree
from .fixed_front_roi import apply_fixed_front_roi
from .bev_parallel_boundaries import (
    hough_offset,
    joint_parallel_fit,
    rectangle,
    rasterize,
    SETTINGS,
)


def automatic_floor(xyz, labels, front):
    ids = np.flatnonzero(front & (labels == 40))
    p = xyz[ids]
    if len(p) < 30:
        raise ValueError("insufficient_front_road_for_floor")
    grid = np.floor(p[:, :2] / 0.5).astype("int32")
    _, inv = np.unique(grid, axis=0, return_inverse=True)
    z = np.full(inv.max() + 1, np.inf)
    np.minimum.at(z, inv, p[:, 2])
    keep = abs(p[:, 2] - z[inv]) < 0.03
    p = p[keep]
    lo, hi = np.quantile(p[:, 2], [0.01, 0.9])
    p = p[(p[:, 2] >= lo - 0.1) & (p[:, 2] <= hi)]
    A = np.c_[p[:, :2], np.ones(len(p))]
    rng = np.random.default_rng(20261028)
    best = None
    score = 0
    for _ in range(160):
        sample = rng.choice(len(p), 3, replace=False)
        try:
            c = np.linalg.solve(A[sample], p[sample, 2])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(c[:2]) > 0.3:
            continue
        inside = abs(A @ c - p[:, 2]) < 0.15
        if inside.sum() > score:
            score = int(inside.sum())
            best = inside
    if best is None or score < 15:
        raise ValueError("no_supported_front_floor")
    c = np.linalg.lstsq(A[best], p[best, 2], rcond=None)[0]
    return c, dict(model_road_points=len(ids), lower_points=len(p), inliers=score)


def detect(xyz, labels, config, direction_override=None):
    mask, projection = apply_fixed_front_roi(xyz, config)
    mask &= labels >= 0
    forward = np.array(config["forward_axis_lidar_xy"])
    normal = np.array([-forward[1], forward[0]])
    plane, floor_info = automatic_floor(xyz, labels, mask)
    height = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
    ground = abs(height) <= 0.25
    road = np.flatnonzero(mask & ground & (labels == 40))
    walk = np.flatnonzero(mask & ground & (labels == 48))
    if len(road) < 10 or len(walk) < 10:
        raise ValueError("no_front_ground_road_sidewalk_pair")
    distance, index = cKDTree(xyz[road, :2]).query(xyz[walk, :2], k=1)
    rids = road[index]
    delta = xyz[walk, :2] - xyz[rids, :2]
    cross = delta @ normal
    along = delta @ forward
    good = (
        (distance > 0.025)
        & (distance <= 0.65)
        & (abs(cross) > 0.025)
        & (abs(along) <= np.maximum(abs(cross) * 1.5, 0.15))
    )
    pairs = np.c_[walk[good], rids[good]]
    if len(pairs) < 6:
        raise ValueError("no_front_interface_support")
    mid = (xyz[pairs[:, 0], :2] + xyz[pairs[:, 1], :2]) / 2
    grid, unique = np.unique(
        np.floor(mid / 0.05).astype("int32"), axis=0, return_index=True
    )
    points = (grid + 0.5) * 0.05
    pairs = pairs[unique]
    if direction_override is None:
        (_, axis, _, angle), _ = joint_parallel_fit(
            {"road": points}, np.zeros(2), forward
        )
    else:
        axis = np.asarray(direction_override, dtype=float)
        axis = axis / np.linalg.norm(axis)
        angle = float(
            np.degrees(
                np.arctan2(axis[1], axis[0]) - np.arctan2(forward[1], forward[0])
            )
        )
    naxis = np.array([-axis[1], axis[0]])
    s = xyz[:, :2] @ axis
    n = xyz[:, :2] @ naxis
    pair_side = np.sign((xyz[pairs[:, 0], :2] - xyz[pairs[:, 1], :2]) @ naxis)
    rho = points @ naxis
    candidates = []
    source_groups = {}
    skipped = []
    for side in [-1, 1]:
        remaining = pair_side == side
        for candidate_no in range(3):
            p = points[remaining]
            if len(p) < 6:
                break
            fit = hough_offset(p, axis, np.zeros(2))
            if fit is None or fit["inlier_span_m"] < 3:
                break
            edge = fit["offset_m"]
            fit.pop("inlier_mask", None)
            eligible = (
                mask
                & (labels == 48)
                & (height >= -0.25)
                & (height <= 2.5)
                & ((n - edge) * side >= -0.2)
                & ((n - edge) * side <= 8)
            )
            inner_points = []
            clipped = 0
            for station in np.arange(1.0, 60.0, 1.0):
                ids = np.flatnonzero(eligible & (abs(s - station) <= 0.5))
                if len(ids) < 6:
                    continue
                t = (n[ids] - edge) * side
                bins = np.unique(np.floor(t / 0.1).astype(int))
                cuts = np.flatnonzero(np.diff(bins) > 5) + 1
                groups = np.split(bins, cuts)
                connected = [g for g in groups if len(g) >= 4 and g.min() * 0.1 <= 1.0]
                if not connected:
                    continue
                run = min(connected, key=lambda g: g.min())
                selected = ids[(t >= run.min() * 0.1) & (t < (run.max() + 1) * 0.1)]
                limit = np.quantile((n[selected] - edge) * side, 0.97)
                i = int(selected[np.argmin(abs((n[selected] - edge) * side - limit))])
                w, h = config["fixed_view_size_px"]
                pixel_margin = min(
                    projection["u"][i],
                    w - projection["u"][i],
                    projection["v"][i],
                    h - projection["v"][i],
                )
                if pixel_margin < 15:
                    clipped += 1
                    continue
                inner_points.append(station * axis + (edge + limit * side) * naxis)
            inner = (
                hough_offset(np.array(inner_points), axis, np.zeros(2))
                if len(inner_points) >= 4
                else None
            )
            if inner is None or inner["inlier_span_m"] < 3:
                skipped.append(
                    dict(
                        side=side,
                        road_offset_m=edge,
                        reason="inner_boundary_unsupported_or_fov_truncated",
                        fov_truncated_stations=clipped,
                    )
                )
                remaining &= abs(rho - edge) > 0.5
                continue
            inner.pop("inlier_mask", None)
            inside = inner["offset_m"]
            width = (inside - edge) * side
            if width < 0.4 or width > 8:
                remaining &= abs(rho - edge) > 0.5
                continue
            # User requests 60m nominal length. Only the shared frustum is reviewable.
            ends = [0.0, 60.0]
            corners = rectangle(np.zeros(2), axis, ends, edge, inside, plane)
            dense_s = np.arange(0, 60.001, 0.25)
            boundary = np.array(
                [ss * axis + nn * naxis for ss in dense_s for nn in [edge, inside]]
            )
            boundary3 = np.c_[boundary, boundary @ plane[:2] + plane[2]]
            visible, _ = apply_fixed_front_roi(boundary3, config)
            both = visible.reshape(-1, 2).all(1)
            valid_s = dense_s[both]
            if len(valid_s) < 2:
                remaining &= abs(rho - edge) > 0.5
                continue
            visible_ends = [float(valid_s.min()), float(valid_s.max())]
            visible_corners = rectangle(
                np.zeros(2), axis, visible_ends, edge, inside, plane
            )
            key = "side_" + str(side) + "_" + str(candidate_no)
            candidates.append(
                dict(
                    id=key,
                    side=side,
                    road_offset_m=edge,
                    inner_offset_m=inside,
                    horizontal_candidate_width_m=width,
                    nominal_length_m=60.0,
                    nominal_corners_xyz_m=corners.tolist(),
                    common_fov_visible_s_interval_m=visible_ends,
                    visible_rectangle_xyz_m=visible_corners.tolist(),
                    road_fit=fit,
                    inner_fit=inner,
                    inner_source_stations=len(inner_points),
                    fov_truncated_stations_excluded=clipped,
                    scope="automatic model/geometric sidewalk candidate; not color-defined outer paving boundary",
                    observed_clear_walkable_area_confirmed=False,
                )
            )
            source_groups[key] = dict(
                road=p[abs(p @ naxis - edge) <= 0.18].tolist(),
                inner=np.asarray(inner_points).tolist(),
            )
            remaining &= abs(rho - edge) > 0.5
    retained = []
    suppressed = []
    for c in candidates:
        interval = sorted([c["road_offset_m"], c["inner_offset_m"]])
        duplicate = False
        for old in retained:
            previous = sorted([old["road_offset_m"], old["inner_offset_m"]])
            overlap = max(
                0, min(interval[1], previous[1]) - max(interval[0], previous[0])
            )
            if (
                c["side"] == old["side"]
                and overlap
                / min(
                    c["horizontal_candidate_width_m"],
                    old["horizontal_candidate_width_m"],
                )
                > 0.75
            ):
                duplicate = True
                break
        if duplicate:
            suppressed.append(
                dict(id=c["id"], reason="nested_duplicate_lateral_interval")
            )
        else:
            retained.append(c)
    return dict(
        front_mask=mask,
        projection=projection,
        axis=axis,
        normal=naxis,
        angle=angle,
        plane=plane,
        floor=floor_info,
        pairs=pairs,
        candidates=retained,
        sources=source_groups,
        skipped=skipped + suppressed,
    )
