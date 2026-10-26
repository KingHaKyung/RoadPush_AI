"""Curb-context evidence for disconnected Sidewalk components."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.surfaces import (
    CurbComponentEvidence,
    CurbEvidence,
)
from surface_passability.obstacles.taxonomy import (
    BOUNDARY_CONTEXT_CLASS_IDS,
    SIDEWALK_CLASS_ID,
)
from surface_passability.policies.surfaces import (
    CURB_EVIDENCE_MINIMUM_BOUNDARY_CONTEXT_FRACTION,
    CURB_EVIDENCE_MINIMUM_COMPONENT_PIXELS,
    CURB_EVIDENCE_MINIMUM_RING_RADIUS_PX,
    CURB_EVIDENCE_REFERENCE_COMPONENT_PIXELS,
    CURB_EVIDENCE_REFERENCE_IMAGE_PIXELS,
    CURB_EVIDENCE_RING_RADIUS_IMAGE_DIVISOR,
)


def extract_curb_evidence(class_map: np.ndarray) -> CurbEvidence:
    """Identify non-anchor Sidewalk components surrounded by boundary context."""

    sidewalk = (class_map == SIDEWALK_CLASS_ID).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        sidewalk,
        8,
    )
    if count == 1:
        return CurbEvidence(
            component_labels=labels,
            suspect_mask=np.zeros(class_map.shape, dtype=bool),
            anchor_component_id=0,
            components=(),
        )

    anchor_component_id = max(
        range(1, count),
        key=lambda component_id: int(stats[component_id, cv2.CC_STAT_AREA]),
    )
    radius = max(
        CURB_EVIDENCE_MINIMUM_RING_RADIUS_PX,
        round(min(class_map.shape) / CURB_EVIDENCE_RING_RADIUS_IMAGE_DIVISOR),
    )
    minimum_pixels = max(
        CURB_EVIDENCE_MINIMUM_COMPONENT_PIXELS,
        round(
            class_map.size
            * CURB_EVIDENCE_REFERENCE_COMPONENT_PIXELS
            / CURB_EVIDENCE_REFERENCE_IMAGE_PIXELS
        ),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    suspect = np.zeros(class_map.shape, dtype=bool)
    components: list[CurbComponentEvidence] = []

    for component_id in range(1, count):
        component = (labels == component_id).astype(np.uint8)
        ring = (cv2.dilate(component, kernel, iterations=1) > 0) & (
            component == 0
        )
        ring_pixels = int(ring.sum())
        context_pixels = int(
            (np.isin(class_map, BOUNDARY_CONTEXT_CLASS_IDS) & ring).sum()
        )
        context_fraction = (
            context_pixels / ring_pixels if ring_pixels else 0.0
        )
        pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        flagged_suspect = (
            component_id != anchor_component_id
            and pixels >= minimum_pixels
            and context_fraction
            >= CURB_EVIDENCE_MINIMUM_BOUNDARY_CONTEXT_FRACTION
        )
        if flagged_suspect:
            suspect |= component.astype(bool)
        components.append(
            CurbComponentEvidence(
                component_id=component_id,
                pixels=pixels,
                bbox_x=int(stats[component_id, cv2.CC_STAT_LEFT]),
                bbox_y=int(stats[component_id, cv2.CC_STAT_TOP]),
                bbox_width=int(stats[component_id, cv2.CC_STAT_WIDTH]),
                bbox_height=int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                centroid_x=float(centroids[component_id, 0]),
                centroid_y=float(centroids[component_id, 1]),
                boundary_context_fraction=context_fraction,
                minimum_pixels_for_image=minimum_pixels,
                anchor=component_id == anchor_component_id,
                flagged_suspect=flagged_suspect,
            )
        )

    return CurbEvidence(
        component_labels=labels,
        suspect_mask=suspect,
        anchor_component_id=anchor_component_id,
        components=tuple(components),
    )

