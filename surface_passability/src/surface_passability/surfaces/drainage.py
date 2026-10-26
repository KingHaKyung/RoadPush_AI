"""Road drainage evidence primitives."""

from __future__ import annotations

import math

import cv2
import numpy as np

from surface_passability.contracts.drainage import (
    DrainComponentGeometry,
    DrainCorridorEvidence,
    DrainFrameContactEvidence,
    DrainRouteVectorEvidence,
    DrainSideSupportEvidence,
)
from surface_passability.policies.drainage import (
    DRAIN_CORRIDOR_BAND_DEPTH_FRACTION,
    DRAIN_CORRIDOR_MARGIN_PX,
)


def find_contiguous_row_segments(row: np.ndarray) -> list[tuple[int, int]]:
    """Return ordered inclusive spans for truthy positions in a flattened row."""

    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    starts = [int(xs[0])]
    ends: list[int] = []
    for left, right in zip(xs[:-1], xs[1:]):
        if right > left + 1:
            ends.append(int(left))
            starts.append(int(right))
    ends.append(int(xs[-1]))
    return list(zip(starts, ends))


def measure_drain_component_geometry(mask: np.ndarray) -> DrainComponentGeometry:
    """Measure PCA geometry for one candidate transverse-drain component."""

    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = (
        np.cov(centered, rowvar=False)
        if len(points) >= 2
        else np.zeros((2, 2), dtype=np.float64)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major = eigenvectors[:, order[0]]
    if major[0] < 0:
        major = -major
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    major_projection = centered @ major
    minor_projection = centered @ minor
    major_span = float(np.ptp(major_projection) + 1.0)
    minor_span = float(np.ptp(minor_projection) + 1.0)
    elongation = float(
        math.sqrt(float(eigenvalues[0]) / max(float(eigenvalues[1]), 1e-9))
    )
    return DrainComponentGeometry(
        center=center,
        major_axis=major,
        minor_axis=minor,
        major_span_px=major_span,
        minor_span_px=minor_span,
        pca_elongation=elongation,
        major_axis_angle_deg=float(math.degrees(math.atan2(major[1], major[0]))),
    )


def estimate_drain_route_vector(
    route_path: list[tuple[int, int]],
    center_y: float,
    component_height: int,
) -> tuple[np.ndarray, DrainRouteVectorEvidence]:
    """Estimate the local Road route direction beside one drain component."""

    if len(route_path) < 2:
        return np.array([0.0, 1.0]), DrainRouteVectorEvidence(
            status="vertical_fallback",
            point_count=len(route_path),
        )
    points = np.asarray([(x, y) for y, x in route_path], np.float64)
    radius = max(component_height * 3, 25)
    local = points[np.abs(points[:, 1] - center_y) <= radius]
    used = local if len(local) >= 2 else points
    y_values = used[:, 1]
    x_values = used[:, 0]
    if float(np.ptp(y_values)) < 1.0:
        vector = used[-1] - used[0]
    else:
        slope, _ = np.polyfit(y_values, x_values, 1)
        vector = np.array([float(slope), 1.0])
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        vector = np.array([0.0, 1.0])
        norm = 1.0
    return vector / norm, DrainRouteVectorEvidence(
        status="local_route_fit" if len(local) >= 2 else "full_route_fit",
        point_count=int(len(used)),
        local_radius_px=int(radius),
    )


def build_shifted_mask_union(
    mask: np.ndarray,
    direction: np.ndarray,
    depth: int,
) -> np.ndarray:
    """Union nearest-neighbor translations of a mask along one direction."""

    height, width = mask.shape
    out = np.zeros_like(mask)
    source = mask.astype(np.uint8)
    for distance in range(1, depth + 1):
        dx, dy = np.rint(direction * distance).astype(int)
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        out |= cv2.warpAffine(
            source,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    return out & ~mask


def measure_drain_side_support(
    component: np.ndarray,
    road_family: np.ndarray,
    minor_axis: np.ndarray,
    depth: int,
) -> tuple[list[DrainSideSupportEvidence], list[np.ndarray]]:
    """Measure Road-family support before and after a drain component."""

    records: list[DrainSideSupportEvidence] = []
    bands: list[np.ndarray] = []
    for label, sign in (("before", -1.0), ("after", 1.0)):
        band = build_shifted_mask_union(component, minor_axis * sign, depth)
        support = band & road_family
        band_pixels = int(band.sum())
        support_pixels = int(support.sum())
        records.append(
            DrainSideSupportEvidence(
                side=label,
                band_pixels=band_pixels,
                road_support_pixels=support_pixels,
                road_support_fraction=float(support_pixels / max(1, band_pixels)),
            )
        )
        bands.append(support)
    return records, bands


def collect_nearby_road_rows(
    road_family: np.ndarray,
    center_x: float,
    y0: int,
    y1: int,
    depth: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Collect qualifying Road spans immediately before and after a component."""

    height = road_family.shape[0]
    groups: list[list[tuple[int, int, int]]] = []
    for start, stop in (
        (max(0, y0 - depth), max(0, y0)),
        (min(height, y1), min(height, y1 + depth)),
    ):
        rows: list[tuple[int, int, int]] = []
        for y in range(start, stop):
            spans = find_contiguous_row_segments(road_family[y])
            if not spans:
                continue
            containing = [span for span in spans if span[0] <= center_x <= span[1]]
            span = (
                max(containing, key=lambda value: value[1] - value[0])
                if containing
                else min(
                    spans,
                    key=lambda value: abs((value[0] + value[1]) / 2 - center_x),
                )
            )
            if span[1] - span[0] + 1 >= 5:
                rows.append((y, span[0], span[1]))
        groups.append(rows)
    return groups[0], groups[1]


def fit_road_edge_line(
    rows: list[tuple[int, int, int]],
    index: int,
) -> tuple[float, float]:
    """Fit one Road-edge x coordinate as a linear function of row y."""

    y_values = np.asarray([row[0] for row in rows], np.float64)
    x_values = np.asarray([row[index] for row in rows], np.float64)
    if len(rows) == 1:
        return 0.0, float(x_values[0])
    slope, intercept = np.polyfit(y_values, x_values, 1)
    return float(slope), float(intercept)


def estimate_drain_corridor(
    shape: tuple[int, int],
    road_family: np.ndarray,
    component: np.ndarray,
    center_x: float,
) -> tuple[np.ndarray, DrainCorridorEvidence]:
    """Estimate the Road corridor crossing one candidate drain component."""

    ys, _ = np.nonzero(component)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    depth = max(10, round(shape[0] * DRAIN_CORRIDOR_BAND_DEPTH_FRACTION))
    before, after = collect_nearby_road_rows(
        road_family, center_x, y0, y1, depth
    )
    evidence = before + after
    corridor = np.zeros(shape, dtype=bool)
    if not before or not after or len(evidence) < 4:
        return corridor, DrainCorridorEvidence(
            status="insufficient_independent_road_rows",
            before_row_count=len(before),
            after_row_count=len(after),
            band_depth_px=depth,
        )
    left_fit = fit_road_edge_line(evidence, 1)
    right_fit = fit_road_edge_line(evidence, 2)
    for y in range(y0, y1):
        left = int(math.floor(left_fit[0] * y + left_fit[1]))
        left -= DRAIN_CORRIDOR_MARGIN_PX
        right = int(math.ceil(right_fit[0] * y + right_fit[1]))
        right += DRAIN_CORRIDOR_MARGIN_PX
        left, right = max(0, left), min(shape[1] - 1, right)
        if right >= left:
            corridor[y, left : right + 1] = True
    center_fit = (
        (left_fit[0] + right_fit[0]) / 2,
        (left_fit[1] + right_fit[1]) / 2,
    )
    route_vector = np.array([center_fit[0], 1.0], dtype=np.float64)
    route_vector /= max(float(np.linalg.norm(route_vector)), 1e-9)
    return corridor, DrainCorridorEvidence(
        status="estimated",
        before_row_count=len(before),
        after_row_count=len(after),
        band_depth_px=depth,
        left_fit_x_per_y=left_fit,
        right_fit_x_per_y=right_fit,
        center_fit_x_per_y=center_fit,
        route_vector_xy=tuple(route_vector.tolist()),
    )


def measure_acute_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned acute angle between two direction vectors."""

    cosine = abs(float(np.dot(first, second))) / max(
        float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-9
    )
    return float(math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0)))))


def measure_drain_frame_contact(mask: np.ndarray) -> DrainFrameContactEvidence:
    """Return the image-frame edges touched by a drain component mask."""

    return DrainFrameContactEvidence(
        left=bool(mask[:, 0].any()),
        right=bool(mask[:, -1].any()),
        top=bool(mask[0].any()),
        bottom=bool(mask[-1].any()),
    )
