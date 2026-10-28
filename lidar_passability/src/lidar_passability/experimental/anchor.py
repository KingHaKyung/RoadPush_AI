"""Direction selection from stored front points; no import-time IO."""

import numpy as np
from scipy.spatial import cKDTree


def select_direction(xyz, labels, raw_ids, plane):
    raw = raw_ids
    height = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])

    def pca_axis(p):
        p = p - p.mean(0)
        _, _, v = np.linalg.svd(p, full_matrices=False)
        d = v[0]
        return d if d[0] > 0 else -d

    def line_candidates(ids, kind, min_span):
        if len(ids) < 20:
            return []
        cells, inv = np.unique(
            np.floor(xyz[ids, :2] / 0.12).astype(int), axis=0, return_inverse=True
        )
        # Equal spatial-cell votes prevent repeated vertical returns dominating.
        pp = np.column_stack(
            [np.bincount(inv, weights=xyz[ids, j]) / np.bincount(inv) for j in range(2)]
        )
        candidates = []
        for deg in np.arange(-35, 35.001, 0.5):
            d = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
            n = np.array([-d[1], d[0]])
            rho = pp @ n
            bins = np.floor(rho / 0.12).astype(int)
            lo = bins.min()
            hist = np.bincount(bins - lo)
            score = np.convolve(hist, np.ones(3), mode="same")
            for peak in np.argsort(score)[-5:]:
                center = (peak + lo + 0.5) * 0.12
                mask = abs(rho - center) <= 0.19
                p = pp[mask]
                if len(p) < 12:
                    continue
                axis = pca_axis(p)
                angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
                if abs(angle) > 35:
                    continue
                normal = np.array([-axis[1], axis[0]])
                offset = float(np.median(p @ normal))
                inside = abs(pp @ normal - offset) <= 0.16
                p = pp[inside]
                s = p @ axis
                # Ignore a small number of distant accidental alignments.
                ends = np.quantile(s, [0.025, 0.975])
                span = float(ends[1] - ends[0])
                bins_s = np.unique(
                    np.floor(s[(s >= ends[0]) & (s <= ends[1])]).astype(int)
                )
                coverage = len(bins_s) / (np.floor(ends[1]) - np.floor(ends[0]) + 1)
                if span < min_span or coverage < 0.65:
                    continue
                rawmask = abs(xyz[ids, :2] @ normal - offset) <= 0.16
                source = ids[
                    rawmask
                    & (xyz[ids, :2] @ axis >= ends[0])
                    & (xyz[ids, :2] @ axis <= ends[1])
                ]
                residual = float(np.median(abs(p @ normal - offset)))
                halves = [p[s <= np.median(s)], p[s > np.median(s)]]
                half_angles = [
                    float(np.degrees(np.arctan2(*(pca_axis(h)[::-1]))))
                    for h in halves
                    if len(h) >= 6
                ]
                disagreement = (
                    abs(half_angles[0] - half_angles[1])
                    if len(half_angles) == 2
                    else 90
                )
                quality = span * coverage / (1 + residual / 0.1 + disagreement / 3)
                candidates.append(
                    dict(
                        kind=kind,
                        angle_deg=angle,
                        direction_xy=axis.tolist(),
                        offset_m=offset,
                        span_m=span,
                        station_interval_m=ends.tolist(),
                        occupied_1m_bins=len(bins_s),
                        coverage=coverage,
                        median_residual_m=residual,
                        half_angle_difference_deg=disagreement,
                        score=quality,
                        raw_ids=raw[source].tolist(),
                        source_local_ids=source.tolist(),
                    )
                )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        unique = []
        for c in candidates:
            if any(
                abs(c["angle_deg"] - x["angle_deg"]) < 2
                and abs(c["offset_m"] - x["offset_m"]) < 0.6
                for x in unique
            ):
                continue
            unique.append(c)
        return unique[:8]

    # Local line-like XY geometry, together with actual vertical extent, identifies
    # facade/low-wall surfaces; a sensing cut edge does not create vertical returns.
    upper = np.flatnonzero(
        (height > 0.35) & (height < 3.5) & ~np.isin(labels, [1, 2, 3, 8, 10, 12])
    )
    tree = cKDTree(xyz[upper, :2])
    good = []
    for j, neighbors in enumerate(tree.query_ball_point(xyz[upper, :2], 0.55)):
        if len(neighbors) < 5:
            continue
        p = xyz[upper[neighbors]]
        vals = np.linalg.eigvalsh(np.cov(p[:, :2].T))
        if vals[1] > 5 * max(vals[0], 1e-6) and np.ptp(p[:, 2]) > 0.45:
            good.append(upper[j])
    static = line_candidates(np.array(good, int), "long_vertical_surface", 8)
    vehicle_ids = np.flatnonzero(
        np.isin(labels, [3, 8]) & (height > 0.2) & (height < 2.5)
    )
    # Separate objects before fitting: a row of parked cars must not be treated as
    # the side face of one vehicle.
    vehicle_tree = cKDTree(xyz[vehicle_ids, :2])
    adj = vehicle_tree.query_ball_point(xyz[vehicle_ids, :2], 0.7)
    seen = set()
    vehicles = []
    for seed in range(len(vehicle_ids)):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        cluster = []
        while stack:
            j = stack.pop()
            cluster.append(j)
            for k in adj[j]:
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        ids = vehicle_ids[cluster]
        if len(ids) < 20 or np.ptp(xyz[ids, 0]) > 8:
            continue
        vehicles.extend(line_candidates(ids, "connected_vehicle_class_surface", 2))
    vehicles.sort(key=lambda c: c["score"], reverse=True)
    vehicles = vehicles[:5]

    chosen = static[0] if static else (vehicles[0] if vehicles else None)
    return chosen, static + vehicles
