"""Thresholds for continuous-core boundary support selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousCorePolicy:
    """Prioritize stable continuous support before disconnected evidence."""

    minimum_core_bands: int = 6
    eligible_run_classifications: tuple[str, ...] = ("consistent",)
