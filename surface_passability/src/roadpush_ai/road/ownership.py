from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadpush_ai import legacy_policy as legacy
from roadpush_ai.road.context import (
    BIKE_LANE_ID,
    ROAD_ID,
    ROAD_PROFILE,
    ROAD_UNION_SEAM_RADIUS_PX,
    RoadContext,
    _repair_class_local,
)
from roadpush_ai.validation.frozen_suite import CYAN, GRAY, GREEN, MAGENTA, ORANGE, RED, YELLOW, contour, paint
from roadpush_ai.surface.context import SIDEWALK_PROFILE, SurfaceContext, SurfaceProfile


@dataclass(frozen=True)
class SurfaceOwnershipProfile:
    """Policy boundary around the reusable route/envelope computation engine."""

    surface_name: str
    policy_name: str
    enabled: bool
    owned_class_ids: tuple[int, ...]
    preserve_class_ids: tuple[int, ...]
    curb_class_ids: tuple[int, ...]
    sidewalk_class_ids: tuple[int, ...]
    local_window_fraction: float = 0.030
    minimum_local_support_fraction: float = 0.45
    maximum_gap_fraction: float = 0.035
    maximum_total_interpolated_rows_fraction: float = 0.075
    maximum_step_fraction: float = 0.045
    minimum_route_progress_fraction: float = 0.20
    maximum_route_angle_deg: float = 60.0
    maximum_line_residual_fraction: float = 0.090
    frame_margin_fraction: float = 0.012
    route_margin_fraction: float = 0.012


SIDEWALK_IDENTITY_OWNERSHIP = SurfaceOwnershipProfile(
    surface_name="sidewalk",
    policy_name="identity_no_op_frozen_sidewalk",
    enabled=False,
    owned_class_ids=SIDEWALK_PROFILE.source_class_ids,
    preserve_class_ids=(),
    curb_class_ids=(),
    sidewalk_class_ids=(),
)

ROAD_CURB_SIDEWALK_ENVELOPE = SurfaceOwnershipProfile(
    surface_name="road",
    policy_name="curb_sidewalk_envelope",
    enabled=True,
    owned_class_ids=(ROAD_ID,),
    # Service Lane and Parking deliberately do not inherit Raw Road's rule.
    preserve_class_ids=(14, 10),
    curb_class_ids=(2, 9),
    sidewalk_class_ids=(15,),
)


@dataclass
class SurfaceOwnershipResult:
    profile: SurfaceOwnershipProfile
    ownership_mask: np.ndarray
    envelope_mask: np.ndarray
    approved_removal: np.ndarray
    uncertain_mask: np.ndarray
    veto_mask: np.ndarray
    boundary_evidence: np.ndarray
    persistent_boundary: np.ndarray
    route_corridor: np.ndarray
    left_track: np.ndarray
    right_track: np.ndarray
    row_records: list[dict]
    diagnostics: dict


def _route_x_by_row(
    shape: tuple[int, int], route_path: list[tuple[int, int]]
) -> np.ndarray:
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
        np.arange(height), unique_y, unique_x, left=unique_x[0], right=unique_x[-1]
    ).astype(np.float32)


def _route_quality(
    shape: tuple[int, int], route_path: list[tuple[int, int]], profile: SurfaceOwnershipProfile
) -> dict:
    height, width = shape
    if len(route_path) < 2:
        return {
            "accepted": False,
            "reason": "route_unavailable",
            "vertical_progress": 0.0,
            "angle_deg": None,
            "line_residual_px": None,
        }
    points = np.asarray(route_path, np.float64)
    vertical_progress = float((points[:, 0].max() - points[:, 0].min()) / height)
    dy = float(points[0, 0] - points[-1, 0])
    dx = float(points[-1, 1] - points[0, 1])
    angle = float(np.degrees(np.arctan2(dx, max(1.0, abs(dy)))))
    if float(np.ptp(points[:, 0])) < 1.0:
        residual = float("inf")
    else:
        fit = np.polyfit(points[:, 0], points[:, 1], 1)
        residual = float(np.percentile(np.abs(points[:, 1] - np.polyval(fit, points[:, 0])), 90))
    reasons = []
    if vertical_progress < profile.minimum_route_progress_fraction:
        reasons.append("insufficient_route_progress")
    if abs(angle) > profile.maximum_route_angle_deg:
        reasons.append("route_not_forward_aligned")
    if residual > width * profile.maximum_line_residual_fraction:
        reasons.append("route_not_straight")
    return {
        "accepted": not reasons,
        "reason": None if not reasons else "+".join(reasons),
        "vertical_progress": vertical_progress,
        "angle_deg": angle,
        "line_residual_px": residual,
    }


def _raw_tracks(
    boundary_evidence: np.ndarray,
    curb_evidence: np.ndarray,
    route_x: np.ndarray,
    profile: SurfaceOwnershipProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = boundary_evidence.shape
    left = np.full(height, np.nan, np.float32)
    right = np.full(height, np.nan, np.float32)
    left_curb = np.zeros(height, bool)
    right_curb = np.zeros(height, bool)
    route_margin = max(2, round(width * profile.route_margin_fraction))
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
    return left, right, left_curb, right_curb


def _persistent_track(
    raw: np.ndarray,
    curb_hit: np.ndarray,
    width: int,
    profile: SurfaceOwnershipProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str | None]]:
    height = len(raw)
    radius = max(3, round(height * profile.local_window_fraction / 2))
    max_gap = max(2, round(height * profile.maximum_gap_fraction))
    max_step = max(4.0, width * profile.maximum_step_fraction)
    frame_margin = max(2, round(width * profile.frame_margin_fraction))
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
        if support < profile.minimum_local_support_fraction:
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
        # A short semantic gap is repaired only between two persistent anchors.
        for y in range(y0 + 1, y1):
            raw[y] = float(np.interp(y, [y0, y1], [raw[y0], raw[y1]]))
            persistent[y] = True
            interpolated[y] = True
            reasons[y] = "short_semantic_gap_interpolated"

    # A lone Sidewalk pixel is weaker than a curb hit, but a locally persistent
    # Sidewalk-facing edge is still valid evidence. The curb flags remain in the
    # row records so review can distinguish the two sources.
    return raw, persistent, interpolated, reasons


def compute_route_centered_ownership(
    source_mask: np.ndarray,
    route_path: list[tuple[int, int]],
    curb_evidence: np.ndarray,
    sidewalk_evidence: np.ndarray,
    profile: SurfaceOwnershipProfile,
) -> SurfaceOwnershipResult:
    """Reusable engine; evidence meaning and approval policy live in the profile."""

    shape = source_mask.shape
    empty = np.zeros(shape, bool)
    if not profile.enabled:
        return SurfaceOwnershipResult(
            profile=profile,
            ownership_mask=np.ones(shape, bool),
            envelope_mask=np.ones(shape, bool),
            approved_removal=empty.copy(),
            uncertain_mask=empty.copy(),
            veto_mask=empty.copy(),
            boundary_evidence=empty.copy(),
            persistent_boundary=empty.copy(),
            route_corridor=source_mask.copy(),
            left_track=np.full(shape[0], np.nan, np.float32),
            right_track=np.full(shape[0], np.nan, np.float32),
            row_records=[],
            diagnostics={
                "policy": profile.policy_name,
                "status": "identity_no_op",
                "source_pixels": int(source_mask.sum()),
                "approved_removal_pixels": 0,
            },
        )

    height, width = shape
    route_x = _route_x_by_row(shape, route_path)
    route_quality = _route_quality(shape, route_path, profile)
    # Only the road-facing contour of broad Sidewalk evidence is used. Curb and
    # Curb Cut remain direct evidence, never an inverse Road mask.
    sidewalk_u8 = sidewalk_evidence.astype(np.uint8)
    sidewalk_contour = sidewalk_evidence & ~cv2.erode(
        sidewalk_u8, np.ones((3, 3), np.uint8)
    ).astype(bool)
    boundary_evidence = curb_evidence | sidewalk_contour
    left, right, left_curb, right_curb = _raw_tracks(
        boundary_evidence, curb_evidence, route_x, profile
    )
    left, left_ok, left_interp, left_reasons = _persistent_track(
        left, left_curb, width, profile
    )
    right, right_ok, right_interp, right_reasons = _persistent_track(
        right, right_curb, width, profile
    )

    envelope = np.ones(shape, bool)
    persistent_boundary = np.zeros(shape, bool)
    uncertain = np.zeros(shape, bool)
    veto = np.zeros(shape, bool)
    route_corridor = np.zeros(shape, bool)
    records = []
    route_margin = max(2, round(width * profile.route_margin_fraction))
    interpolated_rows = int((left_interp | right_interp).sum())
    continuity_veto = (
        interpolated_rows
        > height * profile.maximum_total_interpolated_rows_fraction
    )
    scene_veto = not route_quality["accepted"] or continuity_veto
    scene_veto_reason = (
        route_quality["reason"]
        if not route_quality["accepted"]
        else ("excessive_boundary_gap_interpolation" if continuity_veto else None)
    )

    for y in range(height):
        center = int(round(route_x[y]))
        left_valid = bool(left_ok[y] and left[y] < center - route_margin)
        right_valid = bool(right_ok[y] and right[y] > center + route_margin)
        left_x = int(round(left[y])) if np.isfinite(left[y]) else None
        right_x = int(round(right[y])) if np.isfinite(right[y]) else None
        if left_valid:
            persistent_boundary[y, left_x] = True
        if right_valid:
            persistent_boundary[y, right_x] = True
        core_left = left_x if left_valid else center - route_margin
        core_right = right_x if right_valid else center + route_margin
        route_corridor[y, max(0, core_left) : min(width, core_right + 1)] = True

        status = "owned"
        if scene_veto:
            veto[y] = source_mask[y]
            status = "veto"
        else:
            if left_valid:
                envelope[y, :left_x] = False
            elif left_x is not None:
                uncertain[y, : max(0, left_x)] |= source_mask[y, : max(0, left_x)]
            if right_valid:
                envelope[y, right_x + 1 :] = False
            elif right_x is not None:
                uncertain[y, min(width, right_x + 1) :] |= source_mask[
                    y, min(width, right_x + 1) :
                ]
            if not left_valid and not right_valid:
                status = "uncertain_no_persistent_boundary"
            elif not left_valid or not right_valid:
                status = "one_sided_ownership"
            if left_interp[y] or right_interp[y]:
                status += "+limited_gap_continuity"
        records.append(
            {
                "y": y,
                "route_x": float(route_x[y]),
                "left_boundary_x": left_x,
                "right_boundary_x": right_x,
                "left_persistent": left_valid,
                "right_persistent": right_valid,
                "left_evidence_kind": "curb_or_curb_cut"
                if left_valid and left_curb[y]
                else ("sidewalk_facing_edge" if left_valid else None),
                "right_evidence_kind": "curb_or_curb_cut"
                if right_valid and right_curb[y]
                else ("sidewalk_facing_edge" if right_valid else None),
                "left_reason": left_reasons[y],
                "right_reason": right_reasons[y],
                "status": status,
            }
        )

    approved = source_mask & ~envelope & ~veto
    ownership_mask = ~approved
    neither = ~(left_ok | right_ok)
    gap_runs = []
    start = None
    for y, is_gap in enumerate(neither.tolist() + [False]):
        if is_gap and start is None:
            start = y
        elif not is_gap and start is not None:
            gap_runs.append([start, y - 1])
            start = None
    longest_gap = max((end - begin + 1 for begin, end in gap_runs), default=0)
    diagnostics = {
        "policy": profile.policy_name,
        "status": "scene_veto" if scene_veto else "active",
        "scene_veto_reason": scene_veto_reason,
        "route_quality": route_quality,
        "source_pixels": int(source_mask.sum()),
        "boundary_evidence_pixels": int(boundary_evidence.sum()),
        "persistent_boundary_pixels": int(persistent_boundary.sum()),
        "left_persistent_rows": int(left_ok.sum()),
        "right_persistent_rows": int(right_ok.sum()),
        "interpolated_gap_rows": interpolated_rows,
        "rows_without_persistent_boundary": int(neither.sum()),
        "longest_bilateral_boundary_gap_rows": int(longest_gap),
        "bilateral_boundary_gap_runs": gap_runs,
        "uncertain_pixels": int(uncertain.sum()),
        "veto_pixels": int(veto.sum()),
        "approved_removal_pixels": int(approved.sum()),
        "route_corridor_removed_pixels": int((approved & route_corridor).sum()),
        "obstacle_or_footprint_evidence_used": False,
    }
    return SurfaceOwnershipResult(
        profile=profile,
        ownership_mask=ownership_mask,
        envelope_mask=envelope,
        approved_removal=approved,
        uncertain_mask=uncertain,
        veto_mask=veto,
        boundary_evidence=boundary_evidence,
        persistent_boundary=persistent_boundary,
        route_corridor=route_corridor,
        left_track=left,
        right_track=right,
        row_records=records,
        diagnostics=diagnostics,
    )


def apply_surface_ownership(
    profile: SurfaceProfile,
    source_mask: np.ndarray,
    route_path: list[tuple[int, int]],
    *,
    curb_evidence: np.ndarray | None = None,
    sidewalk_evidence: np.ndarray | None = None,
) -> SurfaceOwnershipResult:
    """Common interface with a frozen Sidewalk identity policy."""

    ownership_profile = (
        SIDEWALK_IDENTITY_OWNERSHIP
        if profile.name == SIDEWALK_PROFILE.name
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


def apply_road_ownership(labels: np.ndarray, road: RoadContext) -> SurfaceOwnershipResult:
    curb = np.isin(labels, ROAD_CURB_SIDEWALK_ENVELOPE.curb_class_ids)
    return apply_surface_ownership(
        road.surface.profile,
        road.raw_road,
        road.surface.route_path,
        curb_evidence=curb,
        sidewalk_evidence=road.trusted_sidewalk_guard,
    )


def build_owned_road_context(
    labels: np.ndarray,
    base: RoadContext,
    ownership: SurfaceOwnershipResult,
) -> RoadContext:
    """Rebuild the Raw Road branch in positive-ownership -> repair order."""

    owned_raw_road = base.raw_road & ownership.ownership_mask
    repaired_road, add_road, records_road = _repair_class_local(
        labels, owned_raw_road, ROAD_ID, base.road_forbidden
    )
    # Repair cannot authorize Raw Road outside the positive envelope.
    repaired_road &= ownership.ownership_mask
    add_road &= ownership.ownership_mask

    # Service Lane and Parking are deliberately conservative pass-throughs.
    repaired_service = base.repaired_service_lane.copy()
    repaired_parking = base.repaired_parking.copy()
    preserve_family = repaired_service | repaired_parking
    union = repaired_road | preserve_family
    seam_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ROAD_UNION_SEAM_RADIUS_PX * 2 + 1,) * 2
    )
    closed = cv2.morphologyEx(
        union.astype(np.uint8), cv2.MORPH_CLOSE, seam_kernel
    ).astype(bool)
    union = ((closed & ownership.ownership_mask) | preserve_family) & ~base.surface_exclusion
    accepted_repair = (
        add_road
        | (base.repaired_service_lane & ~base.raw_service_lane)
        | (base.repaired_parking & ~base.raw_parking)
    ) & ~base.surface_exclusion

    structure, additions = legacy.base.structure_copy(union)
    structural_allow = ownership.ownership_mask | preserve_family
    structure &= structural_allow & ~base.surface_exclusion
    additions &= structural_allow & ~base.surface_exclusion
    route, route_path = legacy.outer.route_path(union)
    observed = legacy.outer.selected_component(structure, route_path)
    if not observed.any():
        observed = structure.copy()
    source_component = legacy.outer.selected_component(union, route_path)
    if not source_component.any():
        source_component = union.copy()
    surface = SurfaceContext(
        profile=ROAD_PROFILE,
        source_mask=source_component,
        repaired_mask=union,
        structure_copy=structure,
        forbidden_mask=base.surface_exclusion,
        observed_component=observed,
        route=route,
        route_path=route_path,
        structure_copy_additions=additions,
    )
    policy = {
        **base.policy,
        "positive_ownership_policy": ownership.profile.policy_name,
        "positive_ownership_before_repair": True,
        "raw_road_ownership_enabled": True,
        "service_lane_ownership_policy": "conservative_identity_insufficient_data",
        "parking_ownership_policy": "conservative_identity_curb_adjacent_allowed",
        "obstacle_or_footprint_used_for_ownership": False,
    }
    return replace(
        base,
        surface=surface,
        raw_road=owned_raw_road,
        repaired_road=repaired_road,
        repaired_service_lane=repaired_service,
        repaired_parking=repaired_parking,
        accepted_repair=accepted_repair,
        repair_records=records_road
        + [row for row in base.repair_records if row["source_class_id"] != ROAD_ID],
        policy=policy,
    )


def render_ownership_review(
    path: Path,
    sample_id: str,
    image: np.ndarray,
    base_road: RoadContext,
    ownership: SurfaceOwnershipResult,
    baseline_structural: np.ndarray,
    probe_structural: np.ndarray,
    probe_free: np.ndarray,
) -> None:
    raw = paint(image, base_road.raw_road, GREEN, 0.48)
    raw = paint(raw, base_road.raw_service_lane, CYAN, 0.65)
    raw = paint(raw, base_road.raw_parking, YELLOW, 0.72)
    existing = paint(image, baseline_structural, GREEN, 0.35)
    existing = contour(existing, baseline_structural, MAGENTA, 2)
    evidence = paint(image, ownership.boundary_evidence, ORANGE, 0.55)
    evidence = paint(evidence, ownership.persistent_boundary, RED, 0.95)
    envelope = paint(image, ownership.envelope_mask & base_road.raw_road, GREEN, 0.42)
    envelope = paint(envelope, ownership.route_corridor, CYAN, 0.18)
    approved = paint(image, ownership.approved_removal, RED, 0.88)
    uncertain = paint(image, ownership.uncertain_mask, ORANGE, 0.75)
    uncertain = paint(uncertain, ownership.veto_mask, GRAY, 0.82)
    final = paint(image, probe_structural, (80, 130, 220), 0.25)
    final = paint(final, probe_free, GREEN, 0.52)
    final = contour(final, probe_structural, MAGENTA, 2)
    panels = [image, raw, existing, evidence, envelope, approved, uncertain, final]
    titles = [
        "1 Original image",
        "2 Raw semantic: Road green / Service cyan / Parking yellow",
        "3 Existing Phase 3 structural Road",
        "4 Boundary evidence: orange=observed / red=persistent",
        "5 Route-centered ownership envelope (cyan=route corridor)",
        "6 Removal approved by Road ownership evidence",
        "7 Uncertain orange / sample or row veto gray",
        "8 Probe result: blue=structural / green=free",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5), constrained_layout=True)
    for axis, title, panel in zip(axes.flat, titles, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.suptitle(
        f"{sample_id} | curb_sidewalk_envelope | approved raw Road removal "
        f"{int(ownership.approved_removal.sum()):,} px | structural delta "
        f"{int((baseline_structural & ~probe_structural).sum()):,} removed / "
        f"{int((probe_structural & ~baseline_structural).sum()):,} added",
        fontsize=12,
    )
    fig.savefig(path, dpi=112, facecolor="white")
    plt.close(fig)
