"""Geometry admission for semantic ground seeds; raw predictions remain intact."""

from pathlib import Path
import sys, math
import numpy as np
from scipy.spatial import cKDTree
from .spatial_grounded import Config, build_spatial
from .direction import direction


def horizontal_seed_support(raw, labels, scores):
    raw = np.asarray(raw, dtype=float)
    finite = np.isfinite(raw).all(1)
    candidates = np.flatnonzero(finite & np.isin(labels, [40, 48]) & (scores >= 0.7))
    admitted = np.zeros(len(raw), bool)
    if not len(candidates):
        return (
            admitted,
            dict(candidate_points=0, representatives=0, horizontal_patches=0),
        )
    _, indices = np.unique(
        np.floor(raw[candidates] / 0.25).astype(np.int64), axis=0, return_index=True
    )
    representatives = candidates[indices]
    if len(representatives) > 15000:
        raise ValueError("bounded 15000 normal queries")
    finite_ids = np.flatnonzero(finite)
    tree = cKDTree(raw[finite])
    neighborhoods = tree.query_ball_point(raw[representatives], 0.4, workers=2)
    horizontal = 0
    patches = []
    for representative, local in zip(representatives, neighborhoods):
        if len(local) < 8:
            continue
        ids = finite_ids[local]
        p = raw[ids]
        center = p.mean(0)
        eigen, vectors = np.linalg.eigh((p - center).T @ (p - center) / len(p))
        if eigen[1] <= 1e-10:
            continue
        normal = vectors[:, 0]
        if abs(normal[2]) < math.cos(math.radians(25)):
            continue
        residual = abs((p - center) @ normal)
        inliers = residual <= 0.03
        if inliers.sum() < 8 or inliers.mean() < 0.65:
            continue
        if np.sqrt(np.mean(residual[inliers] ** 2)) > 0.025:
            continue
        selected = ids[inliers]
        good = np.isin(labels[selected], [40, 48]) & (scores[selected] >= 0.7)
        admitted[selected[good]] = True
        horizontal += 1
        patches.append(
            dict(
                representative_raw_id=int(representative),
                support_raw_ids=list(map(int, selected)),
                normal=normal.tolist(),
            )
        )
    return (
        admitted,
        dict(
            candidate_points=len(candidates),
            representatives=len(representatives),
            horizontal_patches=horizontal,
            admitted_seed_points=int(admitted.sum()),
            sidewalk_seed_points=int((admitted & (labels == 48)).sum()),
            raw_points_removed=0,
            predictions_changed=False,
            method="3D local PCA horizontal plane admission before semantic-cell direction",
            radius_m=0.4,
            query_voxel_m=0.25,
            min_points=8,
            residual_m=0.03,
            max_rms_m=0.025,
            max_slope_deg=25,
            patches=patches,
        ),
    )


def build_grounded(old, raw, labels, scores):
    mask, evidence = horizontal_seed_support(raw, labels, scores)
    center, matrix, direction_evidence = direction(
        raw, np.where(mask, labels, -1), scores
    )
    rotated = raw.copy()
    rotated[:, :2] = (raw[:, :2] - center) @ matrix.T
    extent = rotated[mask & (labels == 48), :2]
    half_s = min(4.0, max(1.0, float(np.quantile(abs(extent[:, 0]), 0.6))))
    half_t = min(3.0, max(1.0, float(np.quantile(abs(extent[:, 1]), 0.8)) + 0.5))
    res = 0.1
    half_s = math.ceil(half_s / res) * res
    half_t = math.ceil(half_t / res) * res
    cfg = Config(**{**old["config"], "bounds_xy": (-half_s, half_s, -half_t, half_t)})
    meta = dict(old["metadata"])
    meta.update(
        raw_to_processing_xy_transform=direction_evidence,
        ground_seed_admission="horizontal_PCA_before_direction_v3",
        source_geometry_scope_changed=True,
        scope="8x6m maximum geometry-selected partial ROI; not same location as prior ROI",
    )
    record = build_spatial(
        rotated,
        np.arange(len(raw)),
        labels,
        scores,
        class_ids=dict(road=40, sidewalk=48),
        metadata=meta,
        config=cfg,
        ground_seed_mask=mask,
    )
    evidence["direction"] = direction_evidence
    return (record, evidence)
