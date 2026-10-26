"""Route-centered primitives for Road ownership analysis."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.road import (
    ActiveSurfaceOwnershipDiagnostics,
    IdentitySurfaceOwnershipDiagnostics,
    PersistentBoundaryTrack,
    RawBoundaryTracks,
    RoadContext,
    RoadOwnershipRouteQualityPolicy,
    RoadOwnershipTrackPolicy,
    RoadOwnershipPersistencePolicy,
    RouteQualityEvidence,
    SurfaceOwnershipProfile,
    SurfaceOwnershipResult,
    SurfaceOwnershipRowEvidence,
)
from surface_passability.contracts.surfaces import SurfaceProfile
from surface_passability.policies.road import (
    DEFAULT_ROAD_OWNERSHIP_ROUTE_QUALITY_POLICY,
    DEFAULT_ROAD_OWNERSHIP_TRACK_POLICY,
    DEFAULT_ROAD_OWNERSHIP_PERSISTENCE_POLICY,
    ROAD_CURB_SIDEWALK_ENVELOPE,
    SIDEWALK_IDENTITY_OWNERSHIP,
)
from surface_passability.policies.surfaces import SIDEWALK_SURFACE_PROFILE


def interpolate_route_x_by_row(
    shape: tuple[int, int],
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> np.ndarray:
    """Interpolate one route-center x coordinate for every image row."""

    height, width = shape
    if not route_path:
        return np.full(height, (width - 1) / 2.0, np.float32)
    points = np.asarray(route_path, np.float32)
    order = np.argsort(points[:, 0])
    ys = points[order, 0]
    xs = points[order, 1]
    unique_y, unique_index = np.unique(ys, return_index=True)
    unique_x = xs[unique_index]
    if len(unique_y) == 1:
        return np.full(height, float(unique_x[0]), np.float32)
    return np.interp(
        np.arange(height),
        unique_y,
        unique_x,
        left=unique_x[0],
        right=unique_x[-1],
    ).astype(np.float32)


def evaluate_route_quality(
    shape: tuple[int, int],
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    policy: RoadOwnershipRouteQualityPolicy = (
        DEFAULT_ROAD_OWNERSHIP_ROUTE_QUALITY_POLICY
    ),
) -> RouteQualityEvidence:
    """Evaluate whether a route is stable enough for ownership inference."""

    height, width = shape
    if len(route_path) < 2:
        return RouteQualityEvidence(
            accepted=False,
            reason="route_unavailable",
            vertical_progress=0.0,
            angle_degrees=None,
            line_residual_px=None,
        )
    points = np.asarray(route_path, np.float64)
    vertical_progress = float(
        (points[:, 0].max() - points[:, 0].min()) / height
    )
    dy = float(points[0, 0] - points[-1, 0])
    dx = float(points[-1, 1] - points[0, 1])
    angle = float(np.degrees(np.arctan2(dx, max(1.0, abs(dy)))))
    if float(np.ptp(points[:, 0])) < 1.0:
        residual = float("inf")
    else:
        fit = np.polyfit(points[:, 0], points[:, 1], 1)
        residual = float(
            np.percentile(
                np.abs(points[:, 1] - np.polyval(fit, points[:, 0])),
                90,
            )
        )
    reasons: list[str] = []
    if vertical_progress < policy.minimum_vertical_progress_fraction:
        reasons.append("insufficient_route_progress")
    if abs(angle) > policy.maximum_absolute_angle_degrees:
        reasons.append("route_not_forward_aligned")
    if residual > width * policy.maximum_line_residual_fraction:
        reasons.append("route_not_straight")
    return RouteQualityEvidence(
        accepted=not reasons,
        reason=None if not reasons else "+".join(reasons),
        vertical_progress=vertical_progress,
        angle_degrees=angle,
        line_residual_px=residual,
    )


def collect_raw_boundary_tracks(
    boundary_evidence: np.ndarray,
    curb_evidence: np.ndarray,
    route_x: np.ndarray,
    policy: RoadOwnershipTrackPolicy = DEFAULT_ROAD_OWNERSHIP_TRACK_POLICY,
) -> RawBoundaryTracks:
    """Collect nearest left/right evidence outside the route margin."""

    height, width = boundary_evidence.shape
    left = np.full(height, np.nan, np.float32)
    right = np.full(height, np.nan, np.float32)
    left_curb = np.zeros(height, bool)
    right_curb = np.zeros(height, bool)
    route_margin = max(2, round(width * policy.route_margin_fraction))
    for y in range(height):
        xs = np.flatnonzero(boundary_evidence[y])
        if not len(xs):
            continue
        center = route_x[y]
        left_x = xs[xs < center - route_margin]
        right_x = xs[xs > center + route_margin]
        if len(left_x):
            left[y] = float(left_x[-1])
            left_curb[y] = bool(curb_evidence[y, int(left_x[-1])])
        if len(right_x):
            right[y] = float(right_x[0])
            right_curb[y] = bool(curb_evidence[y, int(right_x[0])])
    return RawBoundaryTracks(
        left_x=left,
        right_x=right,
        left_curb_hit=left_curb,
        right_curb_hit=right_curb,
    )


def stabilize_boundary_track(
    raw: np.ndarray,
    curb_hit: np.ndarray,
    width: int,
    policy: RoadOwnershipPersistencePolicy = (
        DEFAULT_ROAD_OWNERSHIP_PERSISTENCE_POLICY
    ),
) -> PersistentBoundaryTrack:
    """Accept local boundary evidence and bridge short stable gaps in place."""

    height = len(raw)
    radius = max(3, round(height * policy.local_window_fraction / 2))
    max_gap = max(2, round(height * policy.maximum_gap_fraction))
    max_step = max(4.0, width * policy.maximum_step_fraction)
    frame_margin = max(2, round(width * policy.frame_margin_fraction))
    persistent = np.zeros(height, bool)
    interpolated = np.zeros(height, bool)
    reasons: list[str | None] = [None] * height

    for y in range(height):
        if not np.isfinite(raw[y]):
            reasons[y] = "boundary_absent"
            continue
        lo, hi = max(0, y - radius), min(height, y + radius + 1)
        nearby = raw[lo:hi]
        finite = np.isfinite(nearby)
        support = float(finite.mean())
        if int(raw[y]) <= frame_margin or int(raw[y]) >= width - 1 - frame_margin:
            reasons[y] = "frame_truncated_boundary"
            continue
        if support < policy.minimum_local_support_fraction:
            reasons[y] = "insufficient_adjacent_row_support"
            continue
        median = float(np.median(nearby[finite]))
        if abs(float(raw[y]) - median) > max_step:
            reasons[y] = "abrupt_boundary_step"
            continue
        persistent[y] = True

    accepted_rows = np.flatnonzero(persistent)
    for y0, y1 in zip(accepted_rows[:-1], accepted_rows[1:]):
        gap = int(y1 - y0 - 1)
        if gap <= 0 or gap > max_gap:
            continue
        if abs(float(raw[y1]) - float(raw[y0])) > max_step:
            continue
        for y in range(y0 + 1, y1):
            raw[y] = float(
                np.interp(y, [y0, y1], [raw[y0], raw[y1]])
            )
            persistent[y] = True
            interpolated[y] = True
            reasons[y] = "short_semantic_gap_interpolated"

    return PersistentBoundaryTrack(
        raw_x=raw,
        persistent_mask=persistent,
        interpolated_mask=interpolated,
        reasons=tuple(reasons),
    )


def compute_route_centered_ownership(
    source_mask: np.ndarray,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    curb_evidence: np.ndarray,
    sidewalk_evidence: np.ndarray,
    profile: SurfaceOwnershipProfile = ROAD_CURB_SIDEWALK_ENVELOPE,
) -> SurfaceOwnershipResult:
    """Compute positive surface ownership around an accepted route."""

    shape = source_mask.shape
    empty = np.zeros(shape, bool)
    if not profile.enabled:
        return SurfaceOwnershipResult(
            profile=profile,
            ownership_mask=np.ones(shape, bool),
            envelope_mask=np.ones(shape, bool),
            approved_removal_mask=empty.copy(),
            uncertain_mask=empty.copy(),
            veto_mask=empty.copy(),
            boundary_evidence_mask=empty.copy(),
            persistent_boundary_mask=empty.copy(),
            route_corridor_mask=source_mask.copy(),
            left_track_x=np.full(shape[0], np.nan, np.float32),
            right_track_x=np.full(shape[0], np.nan, np.float32),
            rows=(),
            diagnostics=IdentitySurfaceOwnershipDiagnostics(
                policy=profile.policy_name,
                status="identity_no_op",
                source_pixels=int(source_mask.sum()),
                approved_removal_pixels=0,
            ),
        )

    height, width = shape
    route_x = interpolate_route_x_by_row(shape, route_path)
    route_quality = evaluate_route_quality(
        shape,
        route_path,
        profile.route_quality_policy(),
    )
    sidewalk_u8 = sidewalk_evidence.astype(np.uint8)
    sidewalk_contour = sidewalk_evidence & ~cv2.erode(
        sidewalk_u8,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    boundary_evidence = curb_evidence | sidewalk_contour
    raw_tracks = collect_raw_boundary_tracks(
        boundary_evidence,
        curb_evidence,
        route_x,
        profile.track_policy(),
    )
    persistence_policy = profile.persistence_policy()
    left_track = stabilize_boundary_track(
        raw_tracks.left_x,
        raw_tracks.left_curb_hit,
        width,
        persistence_policy,
    )
    right_track = stabilize_boundary_track(
        raw_tracks.right_x,
        raw_tracks.right_curb_hit,
        width,
        persistence_policy,
    )

    left = left_track.raw_x
    right = right_track.raw_x
    left_ok = left_track.persistent_mask
    right_ok = right_track.persistent_mask
    left_interp = left_track.interpolated_mask
    right_interp = right_track.interpolated_mask
    left_reasons = left_track.reasons
    right_reasons = right_track.reasons

    envelope = np.ones(shape, bool)
    persistent_boundary = np.zeros(shape, bool)
    uncertain = np.zeros(shape, bool)
    veto = np.zeros(shape, bool)
    route_corridor = np.zeros(shape, bool)
    rows: list[SurfaceOwnershipRowEvidence] = []
    route_margin = max(2, round(width * profile.route_margin_fraction))
    interpolated_rows = int((left_interp | right_interp).sum())
    continuity_veto = (
        interpolated_rows
        > height * profile.maximum_total_interpolated_rows_fraction
    )
    scene_veto = not route_quality.accepted or continuity_veto
    scene_veto_reason = (
        route_quality.reason
        if not route_quality.accepted
        else (
            "excessive_boundary_gap_interpolation"
            if continuity_veto
            else None
        )
    )

    for y in range(height):
        center = int(round(route_x[y]))
        left_valid = bool(
            left_ok[y] and left[y] < center - route_margin
        )
        right_valid = bool(
            right_ok[y] and right[y] > center + route_margin
        )
        left_x = int(round(left[y])) if np.isfinite(left[y]) else None
        right_x = int(round(right[y])) if np.isfinite(right[y]) else None
        if left_valid:
            persistent_boundary[y, left_x] = True
        if right_valid:
            persistent_boundary[y, right_x] = True
        core_left = left_x if left_valid else center - route_margin
        core_right = right_x if right_valid else center + route_margin
        route_corridor[
            y,
            max(0, core_left) : min(width, core_right + 1),
        ] = True

        status = "owned"
        if scene_veto:
            veto[y] = source_mask[y]
            status = "veto"
        else:
            if left_valid:
                envelope[y, :left_x] = False
            elif left_x is not None:
                uncertain[y, : max(0, left_x)] |= source_mask[
                    y,
                    : max(0, left_x),
                ]
            if right_valid:
                envelope[y, right_x + 1 :] = False
            elif right_x is not None:
                uncertain[y, min(width, right_x + 1) :] |= source_mask[
                    y,
                    min(width, right_x + 1) :,
                ]
            if not left_valid and not right_valid:
                status = "uncertain_no_persistent_boundary"
            elif not left_valid or not right_valid:
                status = "one_sided_ownership"
            if left_interp[y] or right_interp[y]:
                status += "+limited_gap_continuity"
        rows.append(
            SurfaceOwnershipRowEvidence(
                y=y,
                route_x=float(route_x[y]),
                left_boundary_x=left_x,
                right_boundary_x=right_x,
                left_persistent=left_valid,
                right_persistent=right_valid,
                left_evidence_kind=(
                    "curb_or_curb_cut"
                    if left_valid and raw_tracks.left_curb_hit[y]
                    else (
                        "sidewalk_facing_edge" if left_valid else None
                    )
                ),
                right_evidence_kind=(
                    "curb_or_curb_cut"
                    if right_valid and raw_tracks.right_curb_hit[y]
                    else (
                        "sidewalk_facing_edge" if right_valid else None
                    )
                ),
                left_reason=left_reasons[y],
                right_reason=right_reasons[y],
                status=status,
            )
        )

    approved = source_mask & ~envelope & ~veto
    ownership_mask = ~approved
    neither = ~(left_ok | right_ok)
    gap_runs: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_gap in enumerate(neither.tolist() + [False]):
        if is_gap and start is None:
            start = y
        elif not is_gap and start is not None:
            gap_runs.append((start, y - 1))
            start = None
    longest_gap = max(
        (end - begin + 1 for begin, end in gap_runs),
        default=0,
    )
    diagnostics = ActiveSurfaceOwnershipDiagnostics(
        policy=profile.policy_name,
        status="scene_veto" if scene_veto else "active",
        scene_veto_reason=scene_veto_reason,
        route_quality=route_quality,
        source_pixels=int(source_mask.sum()),
        boundary_evidence_pixels=int(boundary_evidence.sum()),
        persistent_boundary_pixels=int(persistent_boundary.sum()),
        left_persistent_rows=int(left_ok.sum()),
        right_persistent_rows=int(right_ok.sum()),
        interpolated_gap_rows=interpolated_rows,
        rows_without_persistent_boundary=int(neither.sum()),
        longest_bilateral_boundary_gap_rows=int(longest_gap),
        bilateral_boundary_gap_runs=tuple(gap_runs),
        uncertain_pixels=int(uncertain.sum()),
        veto_pixels=int(veto.sum()),
        approved_removal_pixels=int(approved.sum()),
        route_corridor_removed_pixels=int(
            (approved & route_corridor).sum()
        ),
        obstacle_or_footprint_evidence_used=False,
    )
    return SurfaceOwnershipResult(
        profile=profile,
        ownership_mask=ownership_mask,
        envelope_mask=envelope,
        approved_removal_mask=approved,
        uncertain_mask=uncertain,
        veto_mask=veto,
        boundary_evidence_mask=boundary_evidence,
        persistent_boundary_mask=persistent_boundary,
        route_corridor_mask=route_corridor,
        left_track_x=left,
        right_track_x=right,
        rows=tuple(rows),
        diagnostics=diagnostics,
    )


def evaluate_surface_ownership(
    profile: SurfaceProfile,
    source_mask: np.ndarray,
    route_path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    curb_evidence: np.ndarray | None = None,
    sidewalk_evidence: np.ndarray | None = None,
) -> SurfaceOwnershipResult:
    """Evaluate ownership under the profile selected for one surface."""

    ownership_profile = (
        SIDEWALK_IDENTITY_OWNERSHIP
        if profile.name == SIDEWALK_SURFACE_PROFILE.name
        else ROAD_CURB_SIDEWALK_ENVELOPE
    )
    empty = np.zeros_like(source_mask)
    return compute_route_centered_ownership(
        source_mask,
        route_path,
        empty if curb_evidence is None else curb_evidence,
        empty if sidewalk_evidence is None else sidewalk_evidence,
        ownership_profile,
    )


def evaluate_road_ownership(
    class_map: np.ndarray,
    road: RoadContext,
) -> SurfaceOwnershipResult:
    """Assemble Road boundary evidence and evaluate positive ownership."""

    curb_evidence = np.isin(
        class_map,
        ROAD_CURB_SIDEWALK_ENVELOPE.curb_class_ids,
    )
    return evaluate_surface_ownership(
        road.surface.profile,
        road.raw_road_mask,
        road.surface.route_path,
        curb_evidence=curb_evidence,
        sidewalk_evidence=road.trusted_sidewalk_guard_mask,
    )
