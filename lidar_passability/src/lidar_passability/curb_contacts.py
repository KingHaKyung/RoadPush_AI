"""Raw semantic contacts plus local height/plane support, inside the fixed v3 ROI.

No unknown-to-road conversion and no widening of model/postprocessing scope.
"""

import numpy as np
from scipy.spatial import cKDTree
from .boundary_recovery import robust_fit, Policy as FitPolicy


def contacts(record, p, labels, scores, floor):
    x0, x1, t0, t1 = record["config"]["bounds_xy"]
    res = record["config"]["resolution_m"]
    finite = np.isfinite(p).all(1)
    inside = (
        finite & (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= t0) & (p[:, 1] < t1)
    )
    height = p[:, 2] - np.c_[p[:, :2], np.ones(len(p))] @ floor
    side_ids = np.flatnonzero(
        inside & (labels == 48) & (scores >= 0.7) & (abs(height) <= 0.06)
    )
    road_ids = np.flatnonzero(
        inside & (labels == 40) & (height >= -0.35) & (height <= 0.06)
    )
    anchors = {"left": [], "right": []}
    diagnostic = dict(
        sidewalk_floor_points=len(side_ids),
        road_context_points=len(road_ids),
        rejected_no_local_road_plane=0,
        rejected_no_height_or_semantic_gap=0,
        raw_ids_removed=0,
    )
    if not len(side_ids) or not len(road_ids):
        return dict(
            models={"left": None, "right": None}, anchors=anchors, diagnostic=diagnostic
        )
    road_tree = cKDTree(p[road_ids, :2])
    side_tree = cKDTree(p[side_ids, :2])
    for i in range(record["grid"]["shape"][0]):
        s = x0 + (i + 0.5) * res
        row = side_ids[abs(p[side_ids, 0] - s) <= 0.15]
        for sign, name in [(-1, "left"), (1, "right")]:
            proposals = []
            for raw_id in row:
                ids = road_ids[road_tree.query_ball_point(p[raw_id, :2], 0.5)]
                ids = ids[(p[ids, 1] - p[raw_id, 1]) * sign > 0.03]
                if not len(ids):
                    continue
                nearest = ids[
                    np.argmin(np.linalg.norm(p[ids, :2] - p[raw_id, :2], axis=1))
                ]
                gap = float(abs(p[nearest, 1] - p[raw_id, 1]))
                local = road_ids[road_tree.query_ball_point(p[nearest, :2], 0.35)]
                if len(local) < 4:
                    diagnostic["rejected_no_local_road_plane"] += 1
                    continue
                q = p[local]
                X = np.c_[q[:, :2], np.ones(len(q))]
                if np.linalg.matrix_rank(X) < 3:
                    continue
                plane = np.linalg.lstsq(X, q[:, 2], rcond=None)[0]
                residual = abs(X @ plane - q[:, 2])
                if (
                    np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(25))
                    or np.quantile(residual, 0.8) > 0.03
                ):
                    diagnostic["rejected_no_local_road_plane"] += 1
                    continue
                dz = float(p[raw_id, 2] - np.r_[p[raw_id, :2], 1] @ plane)
                if not 0.05 <= dz <= 0.35:
                    diagnostic["rejected_no_height_or_semantic_gap"] += 1
                    continue
                t = float((p[nearest, 1] + p[raw_id, 1]) / 2)
                if not t0 + 0.05 < t < t1 - 0.05:
                    continue
                local_side = side_ids[side_tree.query_ball_point(p[raw_id, :2], 0.25)]
                if len(local_side) < 3:
                    continue
                proposals.append(
                    dict(
                        station_index=i,
                        s_m=s,
                        t_m=t,
                        uncertainty_m=max(0.05, gap / 2),
                        side=name,
                        kind="raw_semantic_contact_with_supported_height_step",
                        height_difference_m=dz,
                        semantic_gap_m=gap,
                        support_point_ids=list(map(int, np.r_[local, local_side])),
                        road_ids=list(map(int, local)),
                        sidewalk_ids=list(map(int, local_side)),
                        physical_boundary_confirmed=False,
                    )
                )
            if proposals:
                anchors[name].append(
                    min(proposals, key=lambda a: (a["uncertainty_m"], a["t_m"]))
                )
    models = {name: robust_fit(values, FitPolicy()) for name, values in anchors.items()}
    return dict(
        models=models,
        anchors=anchors,
        diagnostic=diagnostic,
        policy=dict(
            max_contact_distance_m=0.5,
            min_step_m=0.05,
            max_step_m=0.35,
            road_local_radius_m=0.35,
            road_local_residual_p80_m=0.03,
            source="development hypothesis; not adopted operational thresholds",
        ),
    )
