"""Accepted policies for passability measurement."""

from __future__ import annotations

from dataclasses import dataclass
import math


FREE_ROW_MINIMUM_BAND_RADIUS_PX = 2
FREE_ROW_BAND_IMAGE_FRACTION = 0.006
FREE_ROW_MINIMUM_SUPPORT_FRACTION = 0.45


@dataclass(frozen=True)
class EligibleRowPolicy:
    """Require a sufficiently long, representative structural row run."""

    policy_id: str = "phase7b_rows_v1_longest_coverage50"
    minimum_structural_span_px: int = 12
    minimum_structural_span_fraction: float = 0.02
    minimum_consecutive_rows: int = 5
    minimum_measurement_run_coverage: float = 0.50

    def minimum_span(self, image_width: int) -> int:
        return max(
            self.minimum_structural_span_px,
            int(math.ceil(image_width * self.minimum_structural_span_fraction)),
        )
