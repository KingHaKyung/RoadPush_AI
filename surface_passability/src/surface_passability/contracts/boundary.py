"""Typed result of the complete surface-boundary analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from surface_passability.contracts.boundary_extension import (
    BoundaryObservationSnapshot,
)


@dataclass(frozen=True, slots=True)
class BoundaryAnalysisResult:
    """Expose every accepted boundary stage without downstream recomputation."""

    snapshot: BoundaryObservationSnapshot
    straight_result: dict[str, Any]
    straight_masks: dict[str, np.ndarray]
    generation_diagnostic: dict[str, Any]
    generation_masks: dict[str, np.ndarray]
    occluder_mask: np.ndarray
    decision: dict[str, Any]
    selected_masks: dict[str, np.ndarray]
