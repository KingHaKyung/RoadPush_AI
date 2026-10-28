"""Unchanged development direction proposal."""

import numpy as np


def direction(xyz, labels, scores):
    """Surface-cell PCA proposes a straight strip; it is not a vehicle heading."""
    keep = (labels == 48) & (scores >= 0.7) & np.isfinite(xyz).all(1)
    if keep.sum() < 30:
        raise ValueError("Insufficient scored B sidewalk points for direction proposal")
    keys = np.unique(np.floor(xyz[keep, :2] / 0.25).astype(np.int64), axis=0)
    remaining = {tuple(map(int, k)) for k in keys}
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        todo = [seed]
        comp = []
        while todo:
            k = todo.pop()
            comp.append(k)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (k[0] + dx, k[1] + dy)
                    if nb in remaining:
                        remaining.remove(nb)
                        todo.append(nb)
        components.append(comp)
    comp = max(components, key=lambda c: (len(c), tuple(sorted(c))))
    xy = (np.asarray(comp, dtype=float) + 0.5) * 0.25
    if len(xy) < 20:
        raise ValueError("No sufficiently supported connected sidewalk direction")
    center = xy.mean(0)
    values, vectors = np.linalg.eigh((xy - center).T @ (xy - center) / len(xy))
    if values[-1] / max(values[0], 1e-12) < 2:
        raise ValueError("Direction ambiguous: PCA eigenvalue ratio below draft 2")
    u = vectors[:, -1]
    if u[np.argmax(abs(u))] < 0:
        u = -u
    matrix = np.array([[u[0], u[1]], [-u[1], u[0]]])
    local = (xy - center) @ matrix.T
    return (
        center,
        matrix,
        dict(
            method="largest 8-neighbor sidewalk-cell component + 2D PCA; review proposal",
            cell_m=0.25,
            component_cells=len(comp),
            eigenvalues=values.tolist(),
            source="B predictions only; no GT / sensor-heading substitution",
            origin_xy_m=center.tolist(),
            source_xy_to_strip_matrix=matrix.tolist(),
            straight_strip_only=True,
            curve_or_junction_accuracy_verified=False,
            component_extent_st_m=[local.min(0).tolist(), local.max(0).tolist()],
        ),
    )
