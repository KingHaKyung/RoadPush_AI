from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from roadpush_ai.road.context import CURB_CUT_ID, CURB_ID, RoadContext
from roadpush_ai.surface.context import SurfaceContext


SelectorState = Literal[
    "sidewalk", "road", "uncertain_both", "uncertain_none"
]
InputProfile = Literal["image_upload", "cctv"]


@dataclass(frozen=True)
class SurfaceSelectorPolicy:
    """Conservative Phase-6 routing policy over already-computed contexts."""

    anchor_height_fraction: float = 0.18
    anchor_width_fraction: float = 0.36
    central_width_fraction: float = 0.60
    minimum_source_fraction: float = 0.001
    minimum_structural_fraction: float = 0.004
    minimum_anchor_structural_fraction: float = 0.06
    minimum_vertical_coverage: float = 0.20
    minimum_route_progress: float = 0.15
    image_upload_anchor_margin: float = 0.05
    bilateral_curb_warning_fraction: float = 0.18
    curb_search_width_fraction: float = 0.08
    sidewalk_fill_warning_fraction: float = 0.20
    dominant_sidewalk_source_warning_fraction: float = 0.55
    weak_opposing_road_source_fraction: float = 0.05
    cctv_minimum_sidewalk_component_fraction: float = 0.0025
    cctv_minimum_road_component_fraction: float = 0.0025
    cctv_minimum_component_vertical_span: float = 0.10
    parking_only_is_weak: bool = True
    uncertain_fallback: str = "explicit_mode_required_return_both"


@dataclass(frozen=True)
class SurfaceSelectorEvidence:
    surface: str
    source_pixels: int
    source_fraction: float
    structural_pixels: int
    structural_fraction: float
    anchor_source_fraction: float
    anchor_structural_fraction: float
    central_vertical_coverage: float
    route_progress: float
    route_point_count: int
    structural_source_support: float
    addition_fraction: float
    bilateral_curb_fraction: float
    source_quality: float
    viable: bool
    score: float
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceSelectorResult:
    input_profile: str
    state: SelectorState
    selected_surface: str | None
    fallback: str | None
    sidewalk: SurfaceSelectorEvidence
    road: SurfaceSelectorEvidence
    score_margin: float
    reasons: tuple[str, ...]
    profile_metrics: dict

    def to_dict(self) -> dict:
        return {
            "input_profile": self.input_profile,
            "state": self.state,
            "selected_surface": self.selected_surface,
            "fallback": self.fallback,
            "score_margin": self.score_margin,
            "reasons": list(self.reasons),
            "profile_metrics": self.profile_metrics,
            "sidewalk": self.sidewalk.to_dict(),
            "road": self.road.to_dict(),
        }


def _fraction(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def _anchor(mask: np.ndarray, policy: SurfaceSelectorPolicy) -> np.ndarray:
    height, width = mask.shape
    y0 = max(0, int(round(height * (1.0 - policy.anchor_height_fraction))))
    half = max(1, int(round(width * policy.anchor_width_fraction / 2.0)))
    center = width // 2
    return mask[y0:height, max(0, center - half) : min(width, center + half)]


def _central_vertical_coverage(
    mask: np.ndarray, policy: SurfaceSelectorPolicy
) -> float:
    _, width = mask.shape
    half = max(1, int(round(width * policy.central_width_fraction / 2.0)))
    center = width // 2
    central = mask[:, max(0, center - half) : min(width, center + half)]
    return float(np.any(central, axis=1).mean()) if len(central) else 0.0


def _route_progress(
    route_path: list[tuple[int, int]], height: int
) -> tuple[float, int]:
    if not route_path or height <= 1:
        return 0.0, 0
    # Frozen route paths use image-index order: ``(y, x)``.
    ys = np.asarray([point[0] for point in route_path], dtype=np.float32)
    return float((ys.max() - ys.min()) / (height - 1)), int(len(route_path))


def _bilateral_curb_fraction(
    structural: np.ndarray,
    labels: np.ndarray,
    policy: SurfaceSelectorPolicy,
) -> float:
    """Measure rows where a center surface span has curb evidence on both sides."""

    height, width = structural.shape
    curb = (labels == CURB_ID) | (labels == CURB_CUT_ID)
    search = max(3, int(round(width * policy.curb_search_width_fraction)))
    center = width // 2
    eligible = 0
    bilateral = 0
    for y in np.linspace(int(height * 0.42), int(height * 0.94), 28).astype(int):
        xs = np.flatnonzero(structural[y])
        if not len(xs):
            continue
        splits = np.flatnonzero(np.diff(xs) > 1)
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, len(xs) - 1]
        spans = [(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends)]
        containing = [span for span in spans if span[0] <= center <= span[1]]
        if containing:
            left, right = containing[0]
        else:
            left, right = min(
                spans,
                key=lambda span: min(abs(center - span[0]), abs(center - span[1])),
            )
        if right - left < max(5, int(width * 0.08)):
            continue
        eligible += 1
        left_curb = curb[y, max(0, left - search) : min(width, left + search + 1)].any()
        right_curb = curb[
            y, max(0, right - search) : min(width, right + search + 1)
        ].any()
        bilateral += bool(left_curb and right_curb)
    return bilateral / max(1, eligible)


def _surface_evidence(
    name: str,
    surface: SurfaceContext,
    structural: np.ndarray,
    labels: np.ndarray,
    source_quality: float,
    policy: SurfaceSelectorPolicy,
) -> SurfaceSelectorEvidence:
    source = surface.source_mask.astype(bool)
    structural = structural.astype(bool)
    source_fraction = _fraction(source)
    structural_fraction = _fraction(structural)
    anchor_source_fraction = _fraction(_anchor(source, policy))
    anchor_structural_fraction = _fraction(_anchor(structural, policy))
    vertical_coverage = _central_vertical_coverage(structural, policy)
    route_progress, route_points = _route_progress(surface.route_path, structural.shape[0])
    source_support = float((source & structural).sum()) / max(1, int(structural.sum()))
    additions = structural & ~source
    addition_fraction = float(additions.sum()) / max(1, int(structural.sum()))
    bilateral_curb = _bilateral_curb_fraction(structural, labels, policy)

    rejection_reasons = []
    if source_fraction < policy.minimum_source_fraction:
        rejection_reasons.append("insufficient_source")
    if structural_fraction < policy.minimum_structural_fraction:
        rejection_reasons.append("insufficient_structural_surface")
    if anchor_structural_fraction < policy.minimum_anchor_structural_fraction:
        rejection_reasons.append("no_camera_anchor_support")
    if vertical_coverage < policy.minimum_vertical_coverage:
        rejection_reasons.append("insufficient_vertical_coverage")
    if route_progress < policy.minimum_route_progress:
        rejection_reasons.append("insufficient_route_progress")

    warnings = []
    if addition_fraction > 0.80:
        warnings.append("repair_or_completion_dominated")
    if source_quality < 0.75:
        warnings.append("weak_source_family")
    if (
        name == "sidewalk"
        and structural_fraction >= policy.sidewalk_fill_warning_fraction
        and bilateral_curb >= policy.bilateral_curb_warning_fraction
    ):
        warnings.append("large_center_surface_bounded_by_bilateral_curb")

    viable = not rejection_reasons
    score = (
        0.27 * min(1.0, anchor_source_fraction / 0.55)
        + 0.23 * min(1.0, anchor_structural_fraction / 0.65)
        + 0.18 * min(1.0, vertical_coverage / 0.75)
        + 0.17 * min(1.0, route_progress / 0.65)
        + 0.15 * min(1.0, source_support / 0.75)
    ) * (0.75 + 0.25 * source_quality)
    if "repair_or_completion_dominated" in warnings:
        score *= 0.78
    if "large_center_surface_bounded_by_bilateral_curb" in warnings:
        score *= 0.70

    return SurfaceSelectorEvidence(
        surface=name,
        source_pixels=int(source.sum()),
        source_fraction=source_fraction,
        structural_pixels=int(structural.sum()),
        structural_fraction=structural_fraction,
        anchor_source_fraction=anchor_source_fraction,
        anchor_structural_fraction=anchor_structural_fraction,
        central_vertical_coverage=vertical_coverage,
        route_progress=route_progress,
        route_point_count=route_points,
        structural_source_support=source_support,
        addition_fraction=addition_fraction,
        bilateral_curb_fraction=bilateral_curb,
        source_quality=source_quality,
        viable=viable,
        score=float(score),
        warnings=tuple(warnings),
        rejection_reasons=tuple(rejection_reasons),
    )


def _largest_component_metrics(mask: np.ndarray) -> dict:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return {
            "pixels": 0,
            "fraction": 0.0,
            "vertical_span_fraction": 0.0,
        }
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pixels = int(stats[index, cv2.CC_STAT_AREA])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    return {
        "pixels": pixels,
        "fraction": pixels / max(1, mask.size),
        "vertical_span_fraction": height / max(1, mask.shape[0]),
    }


def _build_evidence(
    sidewalk: SurfaceContext,
    road: RoadContext,
    sidewalk_structural: np.ndarray,
    road_structural: np.ndarray,
    labels: np.ndarray,
    policy: SurfaceSelectorPolicy,
) -> tuple[SurfaceSelectorEvidence, SurfaceSelectorEvidence]:
    road_pixels = int(road.raw_road.sum())
    service_pixels = int(road.raw_service_lane.sum())
    parking_pixels = int(road.raw_parking.sum())
    road_family_pixels = road_pixels + service_pixels + parking_pixels
    if road_pixels + service_pixels:
        road_quality = 1.0
    elif parking_pixels and policy.parking_only_is_weak:
        road_quality = 0.45
    elif road_family_pixels:
        road_quality = 0.75
    else:
        road_quality = 0.0
    return (
        _surface_evidence(
            "sidewalk",
            sidewalk,
            sidewalk_structural,
            labels,
            1.0,
            policy,
        ),
        _surface_evidence(
            "road",
            road.surface,
            road_structural,
            labels,
            road_quality,
            policy,
        ),
    )


def select_surface_for_image_upload(
    sidewalk: SurfaceContext,
    road: RoadContext,
    sidewalk_structural: np.ndarray,
    road_structural: np.ndarray,
    labels: np.ndarray,
    policy: SurfaceSelectorPolicy = SurfaceSelectorPolicy(),
) -> SurfaceSelectorResult:
    """Choose the camera-front surface from a bottom-center structural ROI."""

    sidewalk_evidence, road_evidence = _build_evidence(
        sidewalk,
        road,
        sidewalk_structural,
        road_structural,
        labels,
        policy,
    )
    anchor_margin = abs(
        sidewalk_evidence.anchor_structural_fraction
        - road_evidence.anchor_structural_fraction
    )

    if sidewalk_evidence.viable and not road_evidence.viable:
        state: SelectorState = "sidewalk"
        selected = "sidewalk"
        reasons = ("image_upload_only_sidewalk_camera_anchor_viable",)
    elif road_evidence.viable and not sidewalk_evidence.viable:
        state = "road"
        selected = "road"
        reasons = ("image_upload_only_road_camera_anchor_viable",)
    elif sidewalk_evidence.viable and road_evidence.viable:
        if anchor_margin < policy.image_upload_anchor_margin:
            state = "uncertain_both"
            selected = None
            reasons = ("image_upload_anchor_majority_without_safe_margin",)
        elif (
            sidewalk_evidence.anchor_structural_fraction
            > road_evidence.anchor_structural_fraction
        ):
            state = "sidewalk"
            selected = "sidewalk"
            reasons = ("image_upload_sidewalk_anchor_majority",)
        else:
            state = "road"
            selected = "road"
            reasons = ("image_upload_road_anchor_majority",)
    else:
        state = "uncertain_none"
        selected = None
        reasons = ("image_upload_no_viable_camera_anchor_surface",)

    fallback = None if selected else policy.uncertain_fallback
    return SurfaceSelectorResult(
        input_profile="image_upload",
        state=state,
        selected_surface=selected,
        fallback=fallback,
        sidewalk=sidewalk_evidence,
        road=road_evidence,
        score_margin=float(anchor_margin),
        reasons=reasons,
        profile_metrics={
            "anchor_region": {
                "height_fraction": policy.anchor_height_fraction,
                "width_fraction": policy.anchor_width_fraction,
            },
            "sidewalk_anchor_structural_fraction": sidewalk_evidence.anchor_structural_fraction,
            "road_anchor_structural_fraction": road_evidence.anchor_structural_fraction,
            "minimum_majority_margin": policy.image_upload_anchor_margin,
        },
    )


def select_surface_for_cctv(
    sidewalk: SurfaceContext,
    road: RoadContext,
    sidewalk_structural: np.ndarray,
    road_structural: np.ndarray,
    labels: np.ndarray,
    policy: SurfaceSelectorPolicy = SurfaceSelectorPolicy(),
) -> SurfaceSelectorResult:
    """Prefer any persistent CCTV Sidewalk above the fixed 0.25% gate."""

    sidewalk_evidence, road_evidence = _build_evidence(
        sidewalk,
        road,
        sidewalk_structural,
        road_structural,
        labels,
        policy,
    )
    sidewalk_component = _largest_component_metrics(sidewalk.source_mask)
    road_component = _largest_component_metrics(road.surface.source_mask)
    sidewalk_present = bool(
        sidewalk_component["fraction"]
        >= policy.cctv_minimum_sidewalk_component_fraction
        and sidewalk_component["vertical_span_fraction"]
        >= policy.cctv_minimum_component_vertical_span
    )
    road_present = bool(
        road_component["fraction"] >= policy.cctv_minimum_road_component_fraction
        and road_component["vertical_span_fraction"]
        >= policy.cctv_minimum_component_vertical_span
    )

    if sidewalk_present:
        state: SelectorState = "sidewalk"
        selected = "sidewalk"
        reasons = (
            "cctv_sidewalk_priority_presence_gate",
            "cctv_both_present" if road_present else "cctv_sidewalk_only",
        )
    elif road_present:
        state = "road"
        selected = "road"
        reasons = ("cctv_road_only_after_sidewalk_presence_rejection",)
    else:
        state = "uncertain_none"
        selected = None
        reasons = ("cctv_no_persistent_surface_presence",)

    margin = abs(sidewalk_component["fraction"] - road_component["fraction"])
    fallback = None if selected else policy.uncertain_fallback
    return SurfaceSelectorResult(
        input_profile="cctv",
        state=state,
        selected_surface=selected,
        fallback=fallback,
        sidewalk=sidewalk_evidence,
        road=road_evidence,
        score_margin=float(margin),
        reasons=reasons
        + (
            f"sidewalk_component_fraction={sidewalk_component['fraction']:.8f}",
            "sidewalk_component_vertical_span="
            f"{sidewalk_component['vertical_span_fraction']:.8f}",
            f"road_component_fraction={road_component['fraction']:.8f}",
            "road_component_vertical_span="
            f"{road_component['vertical_span_fraction']:.8f}",
        ),
        profile_metrics={
            "sidewalk_component": sidewalk_component,
            "road_component": road_component,
            "sidewalk_present": sidewalk_present,
            "road_present": road_present,
            "minimum_component_fraction": policy.cctv_minimum_sidewalk_component_fraction,
            "minimum_component_vertical_span": policy.cctv_minimum_component_vertical_span,
            "sidewalk_priority": True,
        },
    )


def select_surface(
    sidewalk: SurfaceContext,
    road: RoadContext,
    sidewalk_structural: np.ndarray,
    road_structural: np.ndarray,
    labels: np.ndarray,
    policy: SurfaceSelectorPolicy = SurfaceSelectorPolicy(),
) -> SurfaceSelectorResult:
    """Backward-compatible name for the image-upload input profile."""

    return select_surface_for_image_upload(
        sidewalk,
        road,
        sidewalk_structural,
        road_structural,
        labels,
        policy,
    )


def select_surface_for_input_profile(
    input_profile: InputProfile,
    sidewalk: SurfaceContext,
    road: RoadContext,
    sidewalk_structural: np.ndarray,
    road_structural: np.ndarray,
    labels: np.ndarray,
    policy: SurfaceSelectorPolicy = SurfaceSelectorPolicy(),
) -> SurfaceSelectorResult:
    """Apply exactly one ingress-selected policy to shared per-image evidence."""

    if input_profile == "image_upload":
        selector = select_surface_for_image_upload
    elif input_profile == "cctv":
        selector = select_surface_for_cctv
    else:
        raise ValueError(f"unsupported input profile: {input_profile}")
    return selector(
        sidewalk,
        road,
        sidewalk_structural,
        road_structural,
        labels,
        policy,
    )

