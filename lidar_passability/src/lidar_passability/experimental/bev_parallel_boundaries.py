"""Metric BEV helpers; no seed-dependent research runner."""

import numpy as np

SETTINGS = dict(
    bev_cell_m=0.05,
    angle_search_camera_half_deg=30.0,
    angle_step_deg=0.25,
    hough_offset_bin_m=0.05,
    hough_inlier_m=0.18,
    minimum_line_support=4,
    minimum_line_length_m=1.0,
    ground_band_m=0.18,
    projected_upper_height_m=2.5,
    display_extra_length_m=6.0,
    display_extra_lateral_m=5.0,
)


def rasterize(
    xyz, labels, origin, direction, core, length_extra=6.0, lateral_limit=7.0, cell=0.05
):
    normal = np.array([-direction[1], direction[0]])
    s = (xyz[:, :2] - origin) @ direction
    n = (xyz[:, :2] - origin) @ normal
    valid = (
        np.isfinite(xyz).all(1)
        & (np.linalg.norm(xyz, axis=1) > 0.3)
        & (s >= core[0] - length_extra)
        & (s <= core[1] + length_extra)
        & (np.abs(n) <= lateral_limit)
    )
    ids = np.flatnonzero(valid)
    indices = np.floor(np.c_[s[ids], n[ids]] / cell).astype("int32")
    grid, inverse, counts = np.unique(
        indices, axis=0, return_inverse=True, return_counts=True
    )
    class_ids = np.array([-1, 1, 2, 3, 8, 10, 12, 40, 48])
    votes = np.zeros((len(grid), len(class_ids)), dtype="int32")
    for j, c in enumerate(class_ids):
        np.add.at(votes[:, j], inverse, (labels[ids] == c).astype("int32"))
    dominant = class_ids[votes.argmax(1)]
    first = np.full(len(grid), len(xyz), dtype="int32")
    np.minimum.at(first, inverse, ids)
    zmin = np.full(len(grid), np.inf)
    zmax = np.full(len(grid), -np.inf)
    np.minimum.at(zmin, inverse, xyz[ids, 2])
    np.maximum.at(zmax, inverse, xyz[ids, 2])
    assert int(counts.sum()) == len(ids)
    return dict(
        grid=grid,
        counts=counts,
        dominant=dominant,
        representative=first,
        zmin=zmin,
        zmax=zmax,
        raw_ids=ids,
        inverse=inverse,
        origin=origin,
        direction=direction,
        normal=normal,
    )


def hough_offset(points, direction, origin):
    """Mode in Hesse-normal rho; dense duplicates cannot dominate repeated cells."""
    normal = np.array([-direction[1], direction[0]])
    rho = (points - origin) @ normal
    bins = np.floor(rho / SETTINGS["hough_offset_bin_m"]).astype("int32")
    lo = int(bins.min())
    hist = np.bincount(bins - lo)
    radius = int(round(SETTINGS["hough_inlier_m"] / SETTINGS["hough_offset_bin_m"]))
    smoothed = np.convolve(hist, np.ones(radius * 2 + 1, dtype=int), mode="full")[
        radius : radius + len(hist)
    ]
    peak = int(np.argmax(smoothed)) + lo
    center = (peak + 0.5) * SETTINGS["hough_offset_bin_m"]
    inside = np.abs(rho - center) <= SETTINGS["hough_inlier_m"] + 0.025
    if inside.sum() < SETTINGS["minimum_line_support"]:
        return None
    offset = float(np.median(rho[inside]))
    residual = np.abs(rho - offset)
    inside = residual <= SETTINGS["hough_inlier_m"]
    along = (points - origin) @ direction
    span = float(np.ptp(along[inside])) if inside.any() else 0.0
    # Group-normalized score; same theta is required for every accepted boundary.
    cost = float(np.mean(np.minimum((residual / SETTINGS["hough_inlier_m"]) ** 2, 1)))
    return dict(
        offset_m=offset,
        source_points=len(points),
        inlier_points=int(inside.sum()),
        inlier_fraction=float(inside.mean()),
        inlier_span_m=span,
        median_inlier_residual_m=(
            float(np.median(residual[inside])) if inside.any() else None
        ),
        cost=cost,
        inlier_mask=inside,
    )


def joint_parallel_fit(groups, origin, forward):
    base = float(np.arctan2(forward[1], forward[0]))
    best = None
    search = []
    for angle in np.arange(
        -SETTINGS["angle_search_camera_half_deg"],
        SETTINGS["angle_search_camera_half_deg"] + 0.001,
        SETTINGS["angle_step_deg"],
    ):
        theta = base + np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        fits = {
            name: hough_offset(points, direction, origin)
            for name, points in groups.items()
        }
        if any(
            f is None or f["inlier_span_m"] < SETTINGS["minimum_line_length_m"]
            for f in fits.values()
        ):
            continue
        cost = float(np.mean([f["cost"] for f in fits.values()]))
        search.append(dict(relative_angle_deg=float(angle), cost=cost))
        if best is None or cost < best[0]:
            best = (cost, direction, fits, float(angle))
    if best is None:
        raise ValueError("No jointly supported parallel boundary fit")
    return best, search


def rectangle(origin, direction, ends, road, inner, plane):
    normal = np.array([-direction[1], direction[0]])
    xy = np.array(
        [
            origin + ends[0] * direction + road * normal,
            origin + ends[1] * direction + road * normal,
            origin + ends[1] * direction + inner * normal,
            origin + ends[0] * direction + inner * normal,
        ]
    )
    xyz = np.c_[xy, xy @ plane[:2] + plane[2]]
    return xyz
