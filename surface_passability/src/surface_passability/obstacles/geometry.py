"""Surface-independent source geometry for obstacle components."""

from __future__ import annotations

import numpy as np
import cv2

from surface_passability.contracts.obstacles import (
    GroundedSilhouetteEvidence,
    LowerAxisGeometry,
)
from surface_passability.policies.obstacles import (
    GROUNDED_SILHOUETTE_BOTTOM_TRUNCATION_MARGIN_PX,
    GROUNDED_SILHOUETTE_DILATION_KERNEL_PX,
    GROUNDED_SILHOUETTE_MINIMUM_SUPPORT_PIXELS,
    LOWER_SILHOUETTE_HEIGHT_FRACTION,
    LOWER_AXIS_HEIGHT_FRACTION,
    LOWER_AXIS_MINIMUM_POINTS,
    LOWER_AXIS_VERTICAL_EPSILON,
)


def extract_lower_silhouette(
    source_mask: np.ndarray,
    fraction: float = LOWER_SILHOUETTE_HEIGHT_FRACTION,
) -> np.ndarray:
    """Retain the accepted lower fraction of an obstacle source mask."""

    rows_y, _ = np.nonzero(source_mask)
    result = np.zeros_like(source_mask, dtype=bool)
    if not len(rows_y):
        return result
    top, bottom = int(rows_y.min()), int(rows_y.max())
    band = max(1, int(np.ceil((bottom - top + 1) * fraction)))
    band_top = max(top, bottom - band + 1)
    result[band_top:bottom + 1] = source_mask[band_top:bottom + 1]
    return result


def evaluate_grounded_silhouette(
    candidate_mask: np.ndarray,
    source_mask: np.ndarray,
    surface_mask: np.ndarray,
) -> GroundedSilhouetteEvidence:
    """Evaluate surface support around a lower-silhouette candidate."""

    if not candidate_mask.any():
        return GroundedSilhouetteEvidence(False, False, 0, 0.0)
    ring = cv2.dilate(
        candidate_mask.astype(np.uint8),
        np.ones(
            (
                GROUNDED_SILHOUETTE_DILATION_KERNEL_PX,
                GROUNDED_SILHOUETTE_DILATION_KERNEL_PX,
            ),
            np.uint8,
        ),
    ).astype(bool)
    support = int((ring & surface_mask).sum())
    fraction = support / max(1, int(ring.sum()))
    rows_y, _ = np.nonzero(source_mask)
    bottom_truncated = bool(
        len(rows_y)
        and int(rows_y.max())
        >= source_mask.shape[0]
        - GROUNDED_SILHOUETTE_BOTTOM_TRUNCATION_MARGIN_PX
    )
    accepted = (
        bottom_truncated
        or support >= GROUNDED_SILHOUETTE_MINIMUM_SUPPORT_PIXELS
    )
    return GroundedSilhouetteEvidence(
        candidate_present=True,
        accepted=accepted,
        ground_support_pixels=support,
        ground_support_fraction=float(fraction),
        source_bottom_truncated=bottom_truncated,
    )


def measure_lower_axis_geometry(source_mask: np.ndarray) -> LowerAxisGeometry:
    """Measure the historical lower-mask PCA cue for an obstacle component."""

    rows_y, columns_x = np.nonzero(source_mask)
    if not len(columns_x):
        return LowerAxisGeometry(90.0, (0.0, 0.0), (1.0, 0.0))
    x0, x1 = int(columns_x.min()), int(columns_x.max())
    y0, y1 = int(rows_y.min()), int(rows_y.max())
    height = y1 - y0 + 1
    lower = source_mask.copy()
    lower[: y0 + round(height * LOWER_AXIS_HEIGHT_FRACTION)] = False
    lower_rows, lower_columns = np.nonzero(lower)
    if len(lower_columns) < LOWER_AXIS_MINIMUM_POINTS:
        return LowerAxisGeometry(
            90.0,
            (float((x0 + x1) / 2), float((y0 + y1) / 2)),
            (1.0, 0.0),
        )
    points = np.column_stack((lower_columns, lower_rows)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[1] < 0:
        axis = -axis
    return LowerAxisGeometry(
        angle_from_vertical_degrees=float(
            np.degrees(
                np.arctan2(
                    abs(axis[0]),
                    abs(axis[1]) + LOWER_AXIS_VERTICAL_EPSILON,
                )
            )
        ),
        center=(float(center[0]), float(center[1])),
        vector=(float(axis[0]), float(axis[1])),
    )
