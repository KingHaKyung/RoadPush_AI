"""Boundary-led local measurement proposals from raw 3D semantic contacts.

No new inference, no changed argmax/confidence, no ROI outside existing +/-20m.
Candidate scope is explicit and is never selected by computed width.
"""

import numpy as np
from scipy.spatial import cKDTree


def propose(raw, labels, scores, matrix, geometry_context=False):
    p = raw.astype(float).copy()
    p[:, :2] = raw[:, :2] @ matrix.T
    finite = np.isfinite(p).all(1) & (abs(raw[:, 0]) <= 20) & (abs(raw[:, 1]) <= 20)
    sid = np.flatnonzero(
        finite
        & (labels == 48)
        & (scores >= 0.7 if not geometry_context else np.isfinite(scores))
    )
    rid = np.flatnonzero(finite & (labels == 40))
    if len(sid) < 8 or len(rid) < 8:
        return ([], dict(reason="semantic_ground_support_missing"))
    rt = cKDTree(p[rid] if geometry_context else p[rid, :2])
    st = cKDTree(p[sid] if geometry_context else p[sid, :2])
    _, idx = np.unique(
        np.floor(p[sid] / [0.25, 0.25, 0.15]).astype(int), axis=0, return_index=True
    )
    queries = sid[idx]
    if len(queries) > 15000:
        raise ValueError("bounded15000 contact probes")
    anchors = []
    rejections = {}
    for a in queries:
        query = p[a] if geometry_context else p[a, :2]
        near = rid[rt.query_ball_point(query, 0.5)]
        if not len(near):
            continue
        b = near[np.argmin(np.linalg.norm(p[near, :2] - p[a, :2], axis=1))]
        rd = rid[rt.query_ball_point(p[b] if geometry_context else p[b, :2], 0.35)]
        sd = sid[st.query_ball_point(query, 0.3)]
        if len(rd) < 4 or len(sd) < 4:
            continue

        def plane(ids):
            q = p[ids]
            X = np.c_[q[:, :2], np.ones(len(q))]
            if np.linalg.matrix_rank(X) < 3:
                return None
            coeff = np.linalg.lstsq(X, q[:, 2], rcond=None)[0]
            if (
                np.linalg.norm(coeff[:2]) > np.tan(np.deg2rad(25))
                or np.quantile(abs(X @ coeff - q[:, 2]), 0.8) > 0.03
            ):
                return None
            return coeff

        rp = plane(rd)
        sp = plane(sd)
        if rp is None or sp is None:
            continue
        dz = float(np.r_[p[a, :2], 1] @ (sp - rp))
        if not 0.05 <= dz <= 0.35:
            continue
        if abs(p[a, 1] - p[b, 1]) < 0.03:
            continue
        anchors.append(
            dict(
                raw_sidewalk_id=int(a),
                raw_road_id=int(b),
                s_m=float((p[a, 0] + p[b, 0]) / 2),
                t_m=float((p[a, 1] + p[b, 1]) / 2),
                side="left" if p[b, 1] < p[a, 1] else "right",
                height_step_m=dz,
                support_point_ids=list(map(int, np.r_[rd, sd])),
                gap_m=float(np.linalg.norm(p[a, :2] - p[b, :2])),
                confidence=float(scores[a]),
                score_used_as_class_acceptance=False,
                family=(
                    "3D_geometry_semantic_context"
                    if geometry_context
                    else "high_score_semantic_contact"
                ),
            )
        )
    groups = []
    for side in ("left", "right"):
        selected = [a for a in anchors if a["side"] == side]
        if not selected:
            continue
        xy = np.array([[a["s_m"], a["t_m"]] for a in selected])
        tree = cKDTree(xy)
        remaining = set(range(len(xy)))
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            queue = [seed]
            comp = []
            while queue:
                a = queue.pop()
                comp.append(a)
                for b in tree.query_ball_point(xy[a], 0.65):
                    if b in remaining:
                        remaining.remove(b)
                        queue.append(b)
            values = [selected[i] for i in comp]
            stations = {int(np.floor(a["s_m"] / 0.1)) for a in values}
            if len(stations) < 6 or np.ptp([a["s_m"] for a in values]) < 0.5:
                continue
            center_st = np.median([[a["s_m"], a["t_m"]] for a in values], axis=0)
            center_st[1] += 1.0 if side == "left" else -1.0
            center_xy = center_st @ matrix
            groups.append(
                dict(
                    side=side,
                    anchors=values,
                    stations=len(stations),
                    center_xy_m=center_xy.tolist(),
                    center_st_m=center_st.tolist(),
                    scope="curb-led alternative partialROI; distinct from previous component-centred region",
                    rank_key=[
                        -len(stations),
                        -len(values),
                        float(center_st[0]),
                        float(center_st[1]),
                    ],
                    source="raw road/sidewalk local-plane contact; not ground truth",
                    auto_adopted=False,
                )
            )
    groups.sort(key=lambda a: a["rank_key"])
    return (
        groups[:2],
        dict(
            queries=len(queries),
            raw_contacts=len(anchors),
            eligible_groups=len(groups),
            proposed_groups=min(2, len(groups)),
            width_used_for_ranking=False,
        ),
    )
