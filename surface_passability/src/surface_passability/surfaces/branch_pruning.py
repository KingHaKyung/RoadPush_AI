"""Branch-aware main-corridor boundary analysis."""

from __future__ import annotations

import numpy as np

from surface_passability.boundary.fitting import fit_side_boundary_continuation
from surface_passability.contracts.boundaries import (
    BoundaryRowObservation,
    BoundaryWidthState,
    MainCorridorAnalysis,
    MainCorridorRow,
)
from surface_passability.policies.boundaries import (
    BOUNDARY_BRANCH_BLOCK_OBSERVED_PADDING_PX,
    BOUNDARY_BRANCH_BLOCK_PREDICTED_PADDING_PX,
    BOUNDARY_BRANCH_BLOCK_VERTICAL_RADIUS_PX,
    BOUNDARY_BRANCH_MEDIAN_WIDTH_FRACTION,
    BOUNDARY_BRANCH_MINIMUM_DEVIATION_PX,
)


def analyze_main_corridor_boundaries(
    observations: tuple[BoundaryRowObservation, ...],
    image_width: int,
    blocked_mask: np.ndarray,
) -> MainCorridorAnalysis:
    """Reject side branches and derive the main-corridor width for each row."""

    left_fit = fit_side_boundary_continuation(observations, "left", image_width)
    right_fit = fit_side_boundary_continuation(observations, "right", image_width)
    widths = (
        np.asarray([observation.width for observation in observations], np.float64)
        if observations
        else np.zeros(0)
    )
    threshold = (
        max(
            BOUNDARY_BRANCH_MINIMUM_DEVIATION_PX,
            float(np.median(widths) * BOUNDARY_BRANCH_MEDIAN_WIDTH_FRACTION),
        )
        if len(widths)
        else BOUNDARY_BRANCH_MINIMUM_DEVIATION_PX
    )
    left_predicted = left_fit.predicted_columns
    right_predicted = right_fit.predicted_columns
    rows: list[MainCorridorRow] = []
    for index, observation in enumerate(observations):
        left_delta = (
            left_predicted[index] - observation.left
            if left_predicted
            else 0.0
        )
        right_delta = (
            observation.right - right_predicted[index]
            if right_predicted
            else 0.0
        )
        opposite_right_ok = (
            not right_predicted
            or observation.right_kind == "frame_truncated"
            or abs(observation.right - right_predicted[index]) <= threshold
        )
        opposite_left_ok = (
            not left_predicted
            or observation.left_kind == "frame_truncated"
            or abs(observation.left - left_predicted[index]) <= threshold
        )
        left_candidate = bool(
            left_predicted
            and left_delta > threshold
            and opposite_right_ok
        )
        right_candidate = bool(
            right_predicted
            and right_delta > threshold
            and opposite_left_ok
        )

        y0 = max(
            0,
            observation.y - BOUNDARY_BRANCH_BLOCK_VERTICAL_RADIUS_PX,
        )
        y1 = min(
            blocked_mask.shape[0],
            observation.y + BOUNDARY_BRANCH_BLOCK_VERTICAL_RADIUS_PX + 1,
        )
        if left_candidate:
            x0 = max(
                0,
                observation.left - BOUNDARY_BRANCH_BLOCK_OBSERVED_PADDING_PX,
            )
            x1 = min(
                blocked_mask.shape[1],
                int(round(left_predicted[index]))
                + BOUNDARY_BRANCH_BLOCK_PREDICTED_PADDING_PX,
            )
            if x1 > x0 and blocked_mask[y0:y1, x0:x1].any():
                left_candidate = False
        if right_candidate:
            x0 = max(
                0,
                int(round(right_predicted[index]))
                - BOUNDARY_BRANCH_BLOCK_OBSERVED_PADDING_PX,
            )
            x1 = min(
                blocked_mask.shape[1],
                observation.right
                + BOUNDARY_BRANCH_BLOCK_PREDICTED_PADDING_PX,
            )
            if x1 > x0 and blocked_mask[y0:y1, x0:x1].any():
                right_candidate = False

        main_left = (
            int(round(left_predicted[index]))
            if left_candidate
            else observation.left
        )
        main_right = (
            int(round(right_predicted[index]))
            if right_candidate
            else observation.right
        )
        main_left = max(
            observation.left,
            min(observation.right, main_left),
        )
        main_right = min(
            observation.right,
            max(observation.left, main_right),
        )
        left_complete = observation.left_kind not in (
            "frame_truncated",
            "obstacle_occlusion",
        )
        right_complete = observation.right_kind not in (
            "frame_truncated",
            "obstacle_occlusion",
        )
        width_state: BoundaryWidthState
        if left_complete and right_complete:
            width_state = "complete"
        elif (
            observation.left_kind == "frame_truncated"
            or observation.right_kind == "frame_truncated"
        ):
            width_state = "lower_bound"
        elif (
            observation.left_kind == "obstacle_occlusion"
            or observation.right_kind == "obstacle_occlusion"
        ):
            width_state = "occlusion_limited"
        else:
            width_state = "unavailable"
        rows.append(MainCorridorRow(
            observation=observation,
            left_branch_candidate=left_candidate,
            right_branch_candidate=right_candidate,
            main_left=main_left,
            main_right=main_right,
            width_state=width_state,
            main_width=max(0, main_right - main_left + 1),
        ))
    return MainCorridorAnalysis(
        rows=tuple(rows),
        left_fit=left_fit,
        right_fit=right_fit,
        branch_threshold_px=threshold,
    )
