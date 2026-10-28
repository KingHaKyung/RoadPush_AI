"""Shared installed-rig predicate. Configuration must be supplied explicitly."""

import numpy as np


def apply_fixed_front_roi(xyz, config):
    """Same frozen numerical predicate for every raw point of every scene."""
    e = np.array(config["lidar_to_fixed_view"])
    p = xyz @ e[:3, :3].T + e[:3, 3]
    depth = p[:, 2]
    a = np.divide(p[:, 0], depth, out=np.zeros(len(xyz)), where=depth > 0.001)
    b = np.divide(p[:, 1], depth, out=np.zeros(len(xyz)), where=depth > 0.001)
    d = config["fixed_view_distortion"]
    k = np.array(config["fixed_view_intrinsic"])
    r2 = a * a + b * b
    rad = 1 + d[0] * r2 + d[1] * r2 * r2 + (d[4] if len(d) > 4 else 0) * r2 * r2 * r2
    u = k[0, 0] * (a * rad + 2 * d[2] * a * b + d[3] * (r2 + 2 * a * a)) + k[0, 2]
    v = k[1, 1] * (b * rad + d[2] * (r2 + 2 * b * b) + 2 * d[3] * a * b) + k[1, 2]
    forward = (xyz[:, :2] - np.array(config["origin_lidar_xy_m"])) @ np.array(
        config["forward_axis_lidar_xy"]
    )
    w, h = config["fixed_view_size_px"]
    mask = (
        np.isfinite(xyz).all(1)
        & (np.linalg.norm(xyz, axis=1) > 0.3)
        & (forward > 0)
        & (forward <= config["forward_limit_m"])
        & (depth > 0)
        & (abs(a) < 1.2)
        & (abs(b) < 1.2)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )
    return mask, dict(forward_m=forward, u=u, v=v, view_depth_m=depth)
