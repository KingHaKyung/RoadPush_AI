"""Thresholds for grounded vegetation at selected boundary caps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundedVegetationCapPolicy:
    """Reject sustained vertical vegetation grounded through a support cap."""

    vegetation_class_id: int = 30
    cap_band_count: int = 5
    minimum_sustained_contact_bands: int = 5
    endpoint_contact_radius_px: int = 6
    minimum_local_vegetation_fraction: float = 0.40
    minimum_component_vertical_aspect: float = 3.0
    minimum_component_image_height_fraction: float = 0.35
