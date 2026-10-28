"""P2 draft: local RANSAC/PCA surface evidence, raw-return occupancy candidates.

Indexed successor, bounded to 150000 raw rows / 10000 grid cells.
NumPy only. No model execution, gap fill, ray clearing, width or passability claim.
All operational thresholds must be provided; fixture settings are not adopted policy.
"""

from dataclasses import dataclass, asdict
from collections import deque
import math
import numpy as np

VERSION = "p2-spatial-draft-0.2"
LOW_OBJECT_NONBLOCKING_MAX_M = 0.18


@dataclass(frozen=True)
class Config:
    bounds_xy: tuple
    resolution_m: float
    support_radius_m: float
    min_support: int
    score_threshold: float
    plane_residual_m: float
    max_slope_deg: float
    max_rms_m: float
    min_inlier_fraction: float
    obstacle_height_min_m: float
    obstacle_height_max_m: float
    component_step_m: float
    ransac_trials: int

    def validate(self):
        values = [v for k, v in asdict(self).items() if k != "bounds_xy"]
        if not np.isfinite(values).all() or not np.isfinite(self.bounds_xy).all():
            raise ValueError("nonfinite config")
        x0, x1, y0, y1 = self.bounds_xy
        if not x0 < x1 or not y0 < y1:
            raise ValueError("invalid bounds")
        if (
            min(
                self.resolution_m,
                self.support_radius_m,
                self.plane_residual_m,
                self.max_rms_m,
                self.component_step_m,
            )
            <= 0
        ):
            raise ValueError("positive distances required")
        if (
            not isinstance(self.min_support, (int, np.integer))
            or not isinstance(self.ransac_trials, (int, np.integer))
            or self.min_support < 3
            or (self.ransac_trials < 1)
        ):
            raise ValueError("invalid counts")
        if not 0 <= self.score_threshold <= 1 or not 0 < self.min_inlier_fraction <= 1:
            raise ValueError("invalid fractions")
        if (
            not 0 < self.max_slope_deg < 90
            or not 0 <= self.obstacle_height_min_m < self.obstacle_height_max_m
        ):
            raise ValueError("invalid slope/height")
        shape = (
            math.ceil((x1 - x0) / self.resolution_m),
            math.ceil((y1 - y0) / self.resolution_m),
        )
        if shape[0] * shape[1] > 10000:
            raise ValueError(
                "draft limited to 10000 cells; no unapproved large processing"
            )
        return shape


def _components(keys, compatible):
    result = {}
    for key in sorted(keys):
        if key in result:
            continue
        cid = max(result.values(), default=0) + 1
        result[key] = cid
        queue = deque([key])
        while queue:
            a = queue.popleft()
            for b in (
                (a[0] - 1, a[1]),
                (a[0] + 1, a[1]),
                (a[0], a[1] - 1),
                (a[0], a[1] + 1),
            ):
                if b in keys and b not in result and compatible(a, b):
                    result[b] = cid
                    queue.append(b)
    return result


def _pca(p):
    center = p.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh((p - center).T @ (p - center) / len(p))
    if eigenvalues[1] <= 1e-10:
        return None
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    return (normal, -float(normal @ center))


def _surrounds(xy, q):
    delta = xy - q
    delta = delta[np.linalg.norm(delta, axis=1) > 1e-09]
    if len(delta) < 3:
        return False
    angles = np.sort(np.arctan2(delta[:, 1], delta[:, 0]))
    return np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() < np.pi - 1e-06


def _plane(p, q, cfg, rng):
    best = None
    best_count = 0
    for _ in range(cfg.ransac_trials):
        sample = p[rng.choice(len(p), 3, replace=False)]
        model = _pca(sample)
        if model is None:
            continue
        n, d = model
        if n[2] < math.cos(math.radians(cfg.max_slope_deg)):
            continue
        mask = np.abs(p @ n + d) <= cfg.plane_residual_m
        if mask.sum() > best_count:
            best, best_count = (mask, int(mask.sum()))
    if (
        best is None
        or best_count < cfg.min_support
        or best_count / len(p) < cfg.min_inlier_fraction
    ):
        return None
    fit = _pca(p[best])
    if fit is None:
        return None
    n, d = fit
    mask = np.abs(p @ n + d) <= cfg.plane_residual_m
    rms = (
        float(np.sqrt(np.mean((p[mask] @ n + d) ** 2))) if mask.any() else float("inf")
    )
    if (
        n[2] < math.cos(math.radians(cfg.max_slope_deg))
        or rms > cfg.max_rms_m
        or mask.sum() < cfg.min_support
        or (mask.mean() < cfg.min_inlier_fraction)
        or (not _surrounds(p[mask, :2], q))
    ):
        return None
    return (n, d, mask, rms)


def build_spatial(
    xyz,
    original_point_id,
    predicted_class,
    score,
    *,
    class_ids,
    metadata,
    config,
    ground_seed_mask=None,
):
    """Input rows are raw-resolution geometry with predictions joined by raw ID.

    Unpredicted raw points use class=-1, score=NaN. Ground seeds require scored
    road/sidewalk; ALL finite raw returns remain available for geometry.
    """
    cfg = config
    shape = cfg.validate()
    p = np.asarray(xyz, dtype=float)
    if len(p) > 150000:
        raise ValueError(
            "bounded indexed draft limited to 150000 raw points; no silent truncation"
        )
    ids = np.asarray(original_point_id)
    labels = np.asarray(predicted_class)
    scores = np.asarray(score, dtype=float)
    ground_seed_mask = (
        np.ones(len(p), dtype=bool)
        if ground_seed_mask is None
        else np.asarray(ground_seed_mask, dtype=bool)
    )
    if ground_seed_mask.shape != (len(p),):
        raise ValueError("ground seed mask row mismatch")
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or any((a.shape != (len(p),) for a in (ids, labels, scores)))
    ):
        raise ValueError("row shapes disagree")
    if (
        not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
        or (ids < 0).any()
    ):
        raise ValueError("unique nonnegative integer raw IDs required")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("integer class IDs required")
    if (
        set(class_ids) != {"road", "sidewalk"}
        or class_ids["road"] == class_ids["sidewalk"]
        or any(
            (not isinstance(v, (int, np.integer)) or v < 0 for v in class_ids.values())
        )
    ):
        raise ValueError("explicit distinct road and sidewalk IDs required")
    for key in (
        "input_sha256",
        "frame_id",
        "coordinate_frame",
        "units",
        "z_up_verified",
        "prediction_provenance",
        "raw_geometry_scope",
    ):
        if key not in metadata:
            raise ValueError("missing metadata: " + key)
    if metadata["units"] != "m" or metadata["z_up_verified"] is not True:
        raise ValueError("requires verified metric z-up coordinates")
    if (
        np.any(np.isfinite(scores) & ((scores < 0) | (scores > 1)))
        or np.isinf(scores).any()
    ):
        raise ValueError("scores must be NaN unknown or finite [0,1]")
    x0, x1, y0, y1 = cfg.bounds_xy
    finite = np.isfinite(p).all(axis=1)
    inside = (
        finite & (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    )
    origin = np.array([x0, y0])
    cell_index = np.zeros((len(p), 2), dtype=int)
    cell_index[inside] = np.floor((p[inside, :2] - origin) / cfg.resolution_m).astype(
        int
    )
    raw_by_cell = {}
    seeds = {}
    for row in np.flatnonzero(inside):
        key = tuple(map(int, cell_index[row]))
        raw_by_cell.setdefault(key, []).append(int(row))
        if (
            ground_seed_mask[row]
            and labels[row] in class_ids.values()
            and np.isfinite(scores[row])
            and (scores[row] >= cfg.score_threshold)
        ):
            seeds.setdefault(key, []).append(int(row))
    seed_class = {
        k: int(labels[v[0]]) for k, v in seeds.items() if len(np.unique(labels[v])) == 1
    }
    seed_components = _components(
        seed_class, lambda a, b: seed_class[a] == seed_class[b]
    )
    seed_rank = {key: i for i, key in enumerate(seeds)}
    bucket_radius = math.ceil(cfg.support_radius_m / cfg.resolution_m) + 1

    def neighbors(key, collection):
        i, j = key
        return [
            k
            for a in range(i - bucket_radius, i + bucket_radius + 1)
            for b in range(j - bucket_radius, j + bucket_radius + 1)
            if (k := (a, b)) in collection
        ]

    rng = np.random.default_rng(20261026)
    cells = {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            key = (i, j)
            q = origin + (np.array(key) + 0.5) * cfg.resolution_m
            rows = raw_by_cell.get(key, [])
            cell = dict(
                index=list(key),
                center_xy_m=q.tolist(),
                surface="unknown",
                ground_z_m=None,
                plane=None,
                surface_component_id=None,
                support_point_ids=[],
                raw_point_ids=[int(ids[r]) for r in rows],
                occupancy_candidate_point_ids=[],
                floor_reference_extrapolated=False,
                low_nonblocking_point_ids=[],
                height_unresolved_point_ids=[],
                out_of_height_band_point_ids=[],
                clearance_volume="unknown",
                observed_surface_only=True,
            )
            cells[key] = cell
            if key not in seed_class:
                if key in seeds:
                    cell["surface_reason"] = "mixed_road_sidewalk_cell"
                continue
            nearby_keys = sorted(neighbors(key, seeds), key=seed_rank.__getitem__)
            candidates = [
                r
                for k in nearby_keys
                if seed_components.get(k) == seed_components[key]
                for r in seeds[k]
            ]
            near = np.asarray(
                [
                    r
                    for r in candidates
                    if np.linalg.norm(p[r, :2] - q) <= cfg.support_radius_m
                ],
                dtype=int,
            )
            if len(near) < cfg.min_support:
                continue
            fit = _plane(p[near], q, cfg, rng)
            if fit is None:
                continue
            n, d, mask, rms = fit
            cell.update(
                surface=(
                    "sidewalk" if seed_class[key] == class_ids["sidewalk"] else "road"
                ),
                ground_z_m=float(-(n[:2] @ q + d) / n[2]),
                plane={
                    "normal": n.tolist(),
                    "d": d,
                    "rms_m": rms,
                    "candidate_count": len(near),
                },
                support_point_ids=ids[near[mask]].astype(int).tolist(),
            )
    supported = {k for k, c in cells.items() if c["plane"] is not None}

    def compatible(a, b):
        return (
            cells[a]["surface"] == cells[b]["surface"]
            and abs(cells[a]["ground_z_m"] - cells[b]["ground_z_m"])
            <= cfg.component_step_m
        )

    surface_components = _components(supported, compatible)
    for k, cid in surface_components.items():
        cells[k]["surface_component_id"] = cid
    supported_order = list(supported)
    supported_rank = {k: i for i, k in enumerate(supported_order)}
    for key, rows in raw_by_cell.items():
        c = cells[key]
        reference_keys = set(neighbors(key, supported))
        references = (
            [key]
            if key in supported
            else [
                k
                for k in sorted(reference_keys, key=supported_rank.__getitem__)
                if np.linalg.norm(
                    np.asarray(cells[k]["center_xy_m"]) - c["center_xy_m"]
                )
                <= cfg.support_radius_m
            ]
        )
        for r in rows:
            if not references:
                c["height_unresolved_point_ids"].append(int(ids[r]))
                continue
            heights = []
            for k in references:
                plane = cells[k]["plane"]
                n = np.array(plane["normal"])
                heights.append(float(-(n[:2] @ p[r, :2] + plane["d"]) / n[2]))
            if max(heights) - min(heights) > cfg.component_step_m:
                c["height_unresolved_point_ids"].append(int(ids[r]))
                continue
            h = float(p[r, 2] - np.median(heights))
            if 0 < h <= LOW_OBJECT_NONBLOCKING_MAX_M + 1e-09:
                if h > cfg.plane_residual_m:
                    c["low_nonblocking_point_ids"].append(int(ids[r]))
                    c["out_of_height_band_point_ids"].append(int(ids[r]))
            elif cfg.obstacle_height_min_m < h <= cfg.obstacle_height_max_m:
                c["occupancy_candidate_point_ids"].append(int(ids[r]))
                c["floor_reference_extrapolated"] |= key not in supported
            elif h > cfg.plane_residual_m or h < -cfg.plane_residual_m:
                c["out_of_height_band_point_ids"].append(int(ids[r]))
    edges = []
    for key, c in cells.items():
        if c["surface"] != "sidewalk":
            continue
        for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (key[0] + delta[0], key[1] + delta[1])
            other = cells.get(nb)
            kind = None
            if other is None or other["surface"] == "unknown":
                kind = "observation_frontier"
            elif other["surface"] == "road":
                kind = "semantic_transition_candidate"
            elif abs(other["ground_z_m"] - c["ground_z_m"]) > cfg.component_step_m:
                kind = "height_step_candidate"
            if kind:
                edges.append(
                    {
                        "cell": list(key),
                        "neighbor": list(nb),
                        "kind": kind,
                        "physical_boundary_confirmed": False,
                    }
                )
    for c in cells.values():
        c["occupancy"] = (
            "occupied_candidate" if c["occupancy_candidate_point_ids"] else "unknown"
        )
        c["surface_only_corridor_candidate"] = c["surface"] == "sidewalk" and (
            not c["occupancy_candidate_point_ids"]
        )
    return dict(
        schema_version=VERSION,
        metadata=dict(metadata),
        config=asdict(cfg),
        grid={
            "origin_xy_m": origin.tolist(),
            "shape": list(shape),
            "axis_order": "x,y",
            "bounds": "half-open; final cells may be clipped by ROI",
        },
        cells=list(cells.values()),
        boundary_candidates=edges,
        excluded_nonfinite_point_ids=ids[~finite].astype(int).tolist(),
        excluded_roi_point_ids=ids[finite & ~inside].astype(int).tolist(),
        free_space_inference="unavailable_no_validated_rays_or_volume_coverage",
        full_sidewalk_boundary_confirmed=False,
        total_width_m=None,
        minimum_passable_ratio=None,
        ground_filter_applied="custom_local_ransac_pca_draft",
        width_policy={
            "low_objects": "excluded_from_width_blocking_by_user_assumption",
            "max_height_m": LOW_OBJECT_NONBLOCKING_MAX_M,
            "decision_date": "2026-10-26",
            "safety_validated": False,
        },
        reference_labels_used=False,
    )
