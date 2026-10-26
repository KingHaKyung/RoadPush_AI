"""Connected semantic obstacle component extraction."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from surface_passability.contracts.obstacles import SemanticObstacleComponent
from surface_passability.policies.obstacles import (
    OBSTACLE_COMPONENT_MINIMUM_IMAGE_FRACTION,
    OBSTACLE_COMPONENT_MINIMUM_PIXELS,
)


def extract_semantic_obstacle_components(
    class_map: np.ndarray,
    class_ids: Iterable[int],
) -> tuple[SemanticObstacleComponent, ...]:
    """Extract accepted eight-connected components for the requested classes."""

    records: list[SemanticObstacleComponent] = []
    minimum_pixels = max(
        OBSTACLE_COMPONENT_MINIMUM_PIXELS,
        round(class_map.size * OBSTACLE_COMPONENT_MINIMUM_IMAGE_FRACTION),
    )
    for class_id in sorted(class_ids):
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            (class_map == class_id).astype(np.uint8),
            8,
        )
        for component_index in range(1, count):
            if (
                int(stats[component_index, cv2.CC_STAT_AREA])
                < minimum_pixels
            ):
                continue
            records.append(SemanticObstacleComponent(
                class_id=int(class_id),
                component_index=int(component_index),
                bbox=tuple(int(value) for value in stats[component_index, :4]),
                source_pixels=int(stats[component_index, cv2.CC_STAT_AREA]),
                mask=component_labels == component_index,
            ))
    return tuple(records)
