"""Explicit CPU boundary entry for coordinates and row-aligned saved labels."""

from typing import Any

import numpy as np

from .fixed_front_roi import apply_fixed_front_roi
from .auto_front_boundaries import automatic_floor, detect
from .anchor import select_direction


def analyze_front_boundaries(
    xyz: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    raw_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return anchor-based proposals/abstention. Does not measure clearance.

    xyz: Nx3 metric LiDAR coordinates. labels: aligned original class IDs,
    not network indices (40 road, 48 sidewalk; -1 unavailable). raw_ids are
    original PCD row IDs, or arange(N) for an unfiltered full-frame input.
    config is the same fixed rig specification for every frame.
    """
    xyz = np.asarray(xyz, dtype=float)
    labels = np.asarray(labels)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or labels.shape != (len(xyz),):
        raise ValueError("expected Nx3 xyz and N aligned class IDs")
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite coordinate")
    if (
        labels.dtype.kind not in "iu"
        or not np.isin(labels, [-1, 1, 2, 3, 8, 10, 12, 40, 48]).all()
    ):
        raise ValueError("expected original AIHub8 class IDs, not network indices")
    raw_ids = np.arange(len(xyz)) if raw_ids is None else np.asarray(raw_ids)
    if (
        raw_ids.shape != (len(xyz),)
        or raw_ids.dtype.kind not in "iu"
        or (raw_ids < 0).any()
        or len(np.unique(raw_ids)) != len(raw_ids)
    ):
        raise ValueError("expected unique nonnegative integer original row IDs")
    mask, _ = apply_fixed_front_roi(xyz, config)
    mask &= labels >= 0
    points, classes, rows = xyz[mask], labels[mask], raw_ids[mask]
    result = dict(
        schema="roadpush.anchor_boundary_package.v1",
        status="direction_anchor_unavailable",
        candidates=[],
        source_points={},
        direction_anchor=None,
        direction_candidates=[],
        user_annotations_used=False,
        rgb_pixels_used=False,
        new_inference=False,
        relative_width_computed=False,
        confirmed_width_m=None,
        confirmed_ratio=None,
        production_eligible=False,
        whole_road_width_branch="not_implemented",
        direction_semantic_identity_confirmed=False,
        physical_boundary_accuracy_verified=False,
    )
    try:
        # Match the stored anchor batch: its initial v4 runner retains a
        # provisional -1.7m plane when baseline boundary extraction abstains.
        # Reproduce that calculation explicitly, without reading old results.
        try:
            baseline = detect(points, classes, config)
            plane = baseline["plane"]
        except ValueError:
            plane = np.array([0.0, 0.0, -1.7])
        result["floor_reference_provisional"] = True
        anchor, alternatives = select_direction(points, classes, rows, plane)
        result.update(
            direction_anchor=anchor,
            direction_candidates=alternatives,
            floor_plane_z_eq_ax_by_c=plane.tolist(),
        )
        if anchor is None:
            return result
        axis = anchor["direction_xy"]
        result.update(direction_xy=axis, normal_xy=[-axis[1], axis[0]])
        found = detect(points, classes, config, direction_override=axis)
        for candidate in found["candidates"]:
            candidate.update(
                width_status="proposal_only_not_accepted_measured_width",
                full_nominal_length_boundary_confirmed=False,
                geometric_fov_is_not_observed_visibility=True,
            )
        result.update(
            status=(
                "automatic_candidates" if found["candidates"] else "boundary_abstention"
            ),
            candidates=found["candidates"],
            source_points=found["sources"],
            skipped=found["skipped"],
            floor_plane_z_eq_ax_by_c=found["plane"].tolist(),
        )
    except ValueError as error:
        result.update(status="boundary_abstention", skipped=[dict(reason=str(error))])
    return result
