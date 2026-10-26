"""Independent side-boundary continuation fitting."""

from __future__ import annotations

import math

import numpy as np

from surface_passability.contracts.boundaries import (
    BoundaryRowObservation,
    BoundarySide,
    SideBoundaryFit,
)
from surface_passability.policies.boundaries import (
    BOUNDARY_FIT_EXCLUDED_ENDPOINT_KINDS,
    BOUNDARY_FIT_FAR_OBSERVATION_FRACTION,
    BOUNDARY_FIT_MEDIAN_RESIDUAL_MULTIPLIER,
    BOUNDARY_FIT_MINIMUM_ELIGIBLE_COUNT,
    BOUNDARY_FIT_MINIMUM_FAR_OBSERVATION_COUNT,
    BOUNDARY_FIT_MINIMUM_OBSERVATION_COUNT,
    BOUNDARY_FIT_MINIMUM_RESIDUAL_PX,
    BOUNDARY_FIT_POLYNOMIAL_DEGREE,
    BOUNDARY_FIT_RESIDUAL_WIDTH_FRACTION,
    BOUNDARY_FIT_ROBUST_ITERATION_COUNT,
)


def fit_side_boundary_continuation(
    observations: tuple[BoundaryRowObservation, ...],
    side: BoundarySide,
    image_width: int,
) -> SideBoundaryFit:
    """Fit a conservative continuation from complete far-corridor observations."""

    if len(observations) < BOUNDARY_FIT_MINIMUM_OBSERVATION_COUNT:
        return SideBoundaryFit(None, (), 0)

    kind_attribute = f"{side}_kind"
    try:
        eligible = [
            index
            for index, observation in enumerate(observations)
            if getattr(observation, kind_attribute)
            not in BOUNDARY_FIT_EXCLUDED_ENDPOINT_KINDS
        ]
    except AttributeError as error:
        raise KeyError(kind_attribute) from error
    if len(eligible) < BOUNDARY_FIT_MINIMUM_ELIGIBLE_COUNT:
        return SideBoundaryFit(None, (), len(eligible))

    far_limit = max(
        BOUNDARY_FIT_MINIMUM_FAR_OBSERVATION_COUNT,
        math.ceil(len(observations) * BOUNDARY_FIT_FAR_OBSERVATION_FRACTION),
    )
    far = [
        index
        for index in eligible
        if index >= len(observations) - far_limit
    ]
    chosen = far if len(far) >= BOUNDARY_FIT_MINIMUM_ELIGIBLE_COUNT else eligible
    rows = np.asarray([observations[index].y for index in chosen], np.float64)
    try:
        columns = np.asarray(
            [getattr(observations[index], side) for index in chosen],
            np.float64,
        )
    except AttributeError as error:
        raise KeyError(side) from error

    keep = np.ones(len(chosen), bool)
    for _ in range(BOUNDARY_FIT_ROBUST_ITERATION_COUNT):
        coefficients = np.polyfit(
            rows[keep],
            columns[keep],
            BOUNDARY_FIT_POLYNOMIAL_DEGREE,
        )
        residual = np.abs(columns - np.polyval(coefficients, rows))
        threshold = max(
            BOUNDARY_FIT_MINIMUM_RESIDUAL_PX,
            image_width * BOUNDARY_FIT_RESIDUAL_WIDTH_FRACTION,
            float(
                np.median(residual[keep])
                * BOUNDARY_FIT_MEDIAN_RESIDUAL_MULTIPLIER
            ),
        )
        updated = residual <= threshold
        if (
            updated.sum() < BOUNDARY_FIT_MINIMUM_ELIGIBLE_COUNT
            or np.array_equal(updated, keep)
        ):
            break
        keep = updated

    coefficients = np.polyfit(
        rows[keep],
        columns[keep],
        BOUNDARY_FIT_POLYNOMIAL_DEGREE,
    )
    all_rows = np.asarray(
        [observation.y for observation in observations],
        np.float64,
    )
    predicted = np.polyval(coefficients, all_rows)
    coefficient_values = coefficients.tolist()
    predicted_values = predicted.tolist()
    return SideBoundaryFit(
        coefficients=(coefficient_values[0], coefficient_values[1]),
        predicted_columns=tuple(predicted_values),
        eligible_observation_count=len(eligible),
    )
