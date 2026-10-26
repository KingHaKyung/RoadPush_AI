"""Natural-obstacle geometry used by surface passability."""

from __future__ import annotations

import cv2
import numpy as np

from surface_passability.contracts.obstacles import (
    HorizontalBracketEvidence,
    NaturalGroundFootprint,
    NaturalHoleEvidence,
)
from surface_passability.obstacles.taxonomy import (
    OCCUPANCY_CLASS_IDS,
    TRANSIENT_OCCLUSION_CLASS_IDS,
)
from surface_passability.policies.obstacles import (
    NATURAL_BRACKET_MINIMUM_FRACTION,
    NATURAL_BRACKET_MINIMUM_ROWS,
    NATURAL_BRACKET_SEARCH_MINIMUM_PX,
    NATURAL_BRACKET_SEARCH_WIDTH_FRACTION,
    NATURAL_FOOTPRINT_BOTTOM_EXTENSION_PX,
    NATURAL_FOOTPRINT_HEIGHT_FRACTION,
    NATURAL_FOOTPRINT_HORIZONTAL_IMAGE_FRACTION,
    NATURAL_FOOTPRINT_HORIZONTAL_MINIMUM_MARGIN_PX,
    NATURAL_FOOTPRINT_HORIZONTAL_WIDTH_FRACTION,
    NATURAL_FOOTPRINT_IMAGE_HEIGHT_FRACTION,
    NATURAL_FOOTPRINT_MINIMUM_DEPTH_PX,
    NATURAL_GROUND_RING_KERNEL_PX,
    NATURAL_GROUND_SUPPORT_MINIMUM_FRACTION,
    NATURAL_GROUND_SUPPORT_MINIMUM_PIXELS,
    NATURAL_GROUNDED_CLASS_IDS,
    NATURAL_GROUNDED_CLASS_NAMES,
    NATURAL_HOLE_BASELINE_COVERAGE_KERNEL_PX,
    NATURAL_HOLE_CLOSE_KERNEL_PX,
    NATURAL_HOLE_COVERAGE_FRACTION,
    NATURAL_HOLE_COVERAGE_MINIMUM_PIXELS,
    NATURAL_HOLE_EXISTING_POLICY_REJECTION_FRACTION,
    NATURAL_HOLE_MINIMUM_AREA_PX,
    NATURAL_HOLE_MINIMUM_HEIGHT_PX,
    NATURAL_HOLE_MINIMUM_IMAGE_AREA_FRACTION,
    NATURAL_HOLE_MINIMUM_IMAGE_FRACTION,
    NATURAL_HOLE_MINIMUM_IMAGE_HEIGHT_FRACTION,
    NATURAL_HOLE_MINIMUM_NATURAL_FRACTION,
    NATURAL_HOLE_MINIMUM_PIXELS,
    NATURAL_HOLE_TRANSIENT_REJECTION_FRACTION,
    NATURAL_INTRUSION_MINIMUM_IMAGE_FRACTION,
    NATURAL_INTRUSION_MINIMUM_PIXELS,
    NATURAL_OVERHEAD_RUN_HEIGHT_FRACTION,
    NATURAL_OVERHEAD_RUN_MINIMUM_PX,
    NATURAL_UPRIGHT_LEFT_PERCENTILE,
    NATURAL_UPRIGHT_RIGHT_PERCENTILE,
)
from surface_passability.surfaces.routes import (
    select_routed_surface_component,
    solve_surface_route,
)


def fill_outer_envelope(component_mask: np.ndarray) -> np.ndarray:
    """Fill every external contour of a natural-obstacle component."""

    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    envelope = np.zeros_like(component_mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(envelope, contours, -1, 1, cv2.FILLED)
    return envelope.astype(bool)


def fill_row_span_envelope(component_mask: np.ndarray) -> np.ndarray:
    """Fill between the outermost occupied pixels on each component row."""

    envelope = np.zeros_like(component_mask, dtype=bool)
    for row_y in np.flatnonzero(component_mask.any(axis=1)):
        columns_x = np.flatnonzero(component_mask[row_y])
        if len(columns_x):
            envelope[
                row_y,
                int(columns_x[0]):int(columns_x[-1]) + 1,
            ] = True
    return envelope


def identify_horizontal_bracket_rows(
    intrusion_mask: np.ndarray,
    source_walkway_mask: np.ndarray,
    search_radius: int,
) -> HorizontalBracketEvidence:
    """Find intrusion rows visibly bracketed by source walkway pixels."""

    _, width = intrusion_mask.shape
    rows: list[tuple[int, int, int]] = []
    restorable = np.zeros_like(intrusion_mask, dtype=bool)
    for row_y in np.flatnonzero(intrusion_mask.any(axis=1)):
        columns_x = np.flatnonzero(intrusion_mask[row_y])
        if not len(columns_x):
            continue
        left, right = int(columns_x[0]), int(columns_x[-1])
        left_support = source_walkway_mask[
            row_y,
            max(0, left - search_radius):left,
        ].any()
        right_support = source_walkway_mask[
            row_y,
            right + 1:min(width, right + 1 + search_radius),
        ].any()
        if left_support and right_support:
            rows.append((int(row_y), left, right))
            restorable[row_y, left:right + 1] = True
    return HorizontalBracketEvidence(tuple(rows), restorable)


def estimate_natural_ground_footprint(
    component_mask: np.ndarray,
    corridor_envelope_mask: np.ndarray,
    source_walkway_mask: np.ndarray,
    class_map: np.ndarray,
    class_id: int,
    component_index: int,
) -> NaturalGroundFootprint:
    """Estimate a grounded clearance footprint for one natural intrusion."""

    height, width = component_mask.shape
    inside = component_mask & corridor_envelope_mask
    rows_y, columns_x = np.nonzero(inside)
    class_id_value = int(class_id)
    class_name = NATURAL_GROUNDED_CLASS_NAMES.get(class_id, str(class_id))
    component_index_value = int(component_index)
    source_pixels = int(component_mask.sum())
    intrusion_pixels = int(inside.sum())
    empty = np.zeros_like(component_mask, dtype=bool)
    minimum = max(
        NATURAL_INTRUSION_MINIMUM_PIXELS,
        round(component_mask.size * NATURAL_INTRUSION_MINIMUM_IMAGE_FRACTION),
    )
    if len(columns_x) < minimum:
        return NaturalGroundFootprint(
            empty,
            empty,
            class_id_value,
            class_name,
            component_index_value,
            source_pixels,
            intrusion_pixels,
            False,
            "too_little_corridor_intrusion",
        )

    x0, x1 = int(columns_x.min()), int(columns_x.max())
    y0, y1 = int(rows_y.min()), int(rows_y.max())
    intrusion_bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    search_radius = max(
        NATURAL_BRACKET_SEARCH_MINIMUM_PX,
        round(width * NATURAL_BRACKET_SEARCH_WIDTH_FRACTION),
    )
    bracket_evidence = identify_horizontal_bracket_rows(
        inside,
        source_walkway_mask,
        search_radius,
    )
    bracketed = bracket_evidence.rows
    intrusion_rows = int(len(np.unique(rows_y)))
    bracket_fraction = len(bracketed) / max(1, intrusion_rows)
    if (
        len(bracketed) < NATURAL_BRACKET_MINIMUM_ROWS
        or bracket_fraction < NATURAL_BRACKET_MINIMUM_FRACTION
    ):
        return NaturalGroundFootprint(
            empty,
            empty,
            class_id_value,
            class_name,
            component_index_value,
            source_pixels,
            intrusion_pixels,
            False,
            "not_an_internal_walkway_intrusion",
            intrusion_bbox,
            len(bracketed),
            intrusion_rows,
            bracket_fraction,
        )

    bottom = max(row[0] for row in bracketed)
    top = min(row[0] for row in bracketed)
    natural = np.isin(
        class_map,
        np.asarray(NATURAL_GROUNDED_CLASS_IDS, dtype=np.uint8),
    )
    x0, x1 = int(columns_x.min()), int(columns_x.max())
    anchor_y = max(0, top - 1)
    minimum_vertical_run = max(
        NATURAL_OVERHEAD_RUN_MINIMUM_PX,
        round((bottom - top + 1) * NATURAL_OVERHEAD_RUN_HEIGHT_FRACTION),
    )
    overhead_columns: list[int] = []
    overhead_runs: dict[int, int] = {}
    for column_x in range(x0, x1 + 1):
        run = 0
        for row_y in range(anchor_y, -1, -1):
            if natural[row_y, column_x]:
                run += 1
            else:
                break
        if run >= minimum_vertical_run:
            overhead_columns.append(column_x)
            overhead_runs[column_x] = run

    source = np.zeros_like(component_mask, dtype=bool)
    if overhead_columns:
        groups: list[list[int]] = []
        for column_x in overhead_columns:
            if not groups or column_x > groups[-1][-1] + 1:
                groups.append([column_x])
            else:
                groups[-1].append(column_x)
        selected_columns = max(groups, key=len)
        left, right = selected_columns[0], selected_columns[-1]
        source_top = max(
            0,
            anchor_y - max(overhead_runs[x] for x in selected_columns) + 1,
        )
        source[source_top:bottom + 1, left:right + 1] = natural[
            source_top:bottom + 1,
            left:right + 1,
        ]
        strategy = "overhead_vertical_silhouette_projected_to_ground"
    else:
        upper_bottom = top + max(1, (bottom - top + 1) // 2)
        upper_rows = [row for row in bracketed if row[0] <= upper_bottom]
        left = int(round(np.percentile(
            [row[1] for row in upper_rows],
            NATURAL_UPRIGHT_LEFT_PERCENTILE,
        )))
        right = int(round(np.percentile(
            [row[2] for row in upper_rows],
            NATURAL_UPRIGHT_RIGHT_PERCENTILE,
        )))
        source = component_mask.copy()
        strategy = "upper_upright_silhouette_projected_to_ground"

    stable_width = float(right - left + 1)
    margin = max(
        NATURAL_FOOTPRINT_HORIZONTAL_MINIMUM_MARGIN_PX,
        round(width * NATURAL_FOOTPRINT_HORIZONTAL_IMAGE_FRACTION),
        round(
            (right - left + 1)
            * NATURAL_FOOTPRINT_HORIZONTAL_WIDTH_FRACTION
        ),
    )
    left, right = max(0, left - margin), min(width - 1, right + margin)
    depth = max(
        NATURAL_FOOTPRINT_MINIMUM_DEPTH_PX,
        min(
            round((bottom - top + 1) * NATURAL_FOOTPRINT_HEIGHT_FRACTION),
            round(height * NATURAL_FOOTPRINT_IMAGE_HEIGHT_FRACTION),
        ),
    )
    footprint_top = max(0, bottom - depth + 1)
    footprint_bottom = min(
        height - 1,
        bottom + NATURAL_FOOTPRINT_BOTTOM_EXTENSION_PX,
    )
    footprint = np.zeros_like(component_mask, dtype=bool)
    footprint[
        footprint_top:footprint_bottom + 1,
        left:right + 1,
    ] = True

    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (NATURAL_GROUND_RING_KERNEL_PX, NATURAL_GROUND_RING_KERNEL_PX),
    )
    ring = (
        cv2.dilate(footprint.astype(np.uint8), ring_kernel).astype(bool)
        & ~footprint
    )
    support_pixels = int((ring & source_walkway_mask).sum())
    support_fraction = support_pixels / max(1, int(ring.sum()))
    result_fields = {
        "intrusion_bbox": intrusion_bbox,
        "bracketed_rows": len(bracketed),
        "intrusion_rows": intrusion_rows,
        "bracket_fraction": bracket_fraction,
        "stable_lower_width_px": stable_width,
        "silhouette_strategy": strategy,
        "overhead_vertical_run_threshold": minimum_vertical_run,
        "overhead_supported_columns": len(overhead_columns),
        "source_silhouette_pixels": int(source.sum()),
        "footprint_bbox": (
            left,
            footprint_top,
            right - left + 1,
            footprint_bottom - footprint_top + 1,
        ),
        "footprint_pixels": int(footprint.sum()),
        "ground_ring_support_pixels": support_pixels,
        "ground_ring_support_fraction": support_fraction,
    }
    if (
        support_pixels < NATURAL_GROUND_SUPPORT_MINIMUM_PIXELS
        or support_fraction < NATURAL_GROUND_SUPPORT_MINIMUM_FRACTION
    ):
        return NaturalGroundFootprint(
            empty,
            empty,
            class_id_value,
            class_name,
            component_index_value,
            source_pixels,
            intrusion_pixels,
            False,
            "no_visible_walkway_ground_support",
            **result_fields,
        )

    footprint &= corridor_envelope_mask
    result_fields["footprint_pixels"] = int(footprint.sum())
    accepted = bool(footprint.any())
    return NaturalGroundFootprint(
        footprint,
        source,
        class_id_value,
        class_name,
        component_index_value,
        source_pixels,
        intrusion_pixels,
        accepted,
        None if accepted else "footprint_outside_corridor",
        **result_fields,
    )


def collect_natural_hole_evidence(
    class_map: np.ndarray,
    repaired_surface_mask: np.ndarray,
    structural_surface_mask: np.ndarray,
    source_walkway_mask: np.ndarray,
    baseline_footprint_mask: np.ndarray,
) -> NaturalHoleEvidence:
    """Recover grounded natural obstacles from holes inside the route corridor."""

    route = solve_surface_route(source_walkway_mask)
    corridor = select_routed_surface_component(
        structural_surface_mask,
        route.main_path,
    )
    if not corridor.any():
        corridor = structural_surface_mask
    envelope = fill_row_span_envelope(corridor)
    holes = envelope & ~repaired_surface_mask
    holes = cv2.morphologyEx(
        holes.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (NATURAL_HOLE_CLOSE_KERNEL_PX, NATURAL_HOLE_CLOSE_KERNEL_PX),
        ),
    ).astype(bool)
    footprints = np.zeros_like(class_map, dtype=bool)
    source_occluders = np.zeros_like(class_map, dtype=bool)
    accepted_sources = np.zeros_like(class_map, dtype=bool)
    rejected_sources = np.zeros_like(class_map, dtype=bool)
    records: list[dict[str, object]] = []
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8),
        8,
    )
    transient_ids = np.asarray(
        tuple(TRANSIENT_OCCLUSION_CLASS_IDS),
        dtype=np.uint8,
    )
    existing_policy_ids = np.asarray(
        tuple(OCCUPANCY_CLASS_IDS) + tuple(TRANSIENT_OCCLUSION_CLASS_IDS),
        dtype=np.uint8,
    )
    natural_ids = np.asarray(NATURAL_GROUNDED_CLASS_IDS, dtype=np.uint8)
    covered = cv2.dilate(
        baseline_footprint_mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                NATURAL_HOLE_BASELINE_COVERAGE_KERNEL_PX,
                NATURAL_HOLE_BASELINE_COVERAGE_KERNEL_PX,
            ),
        ),
    ).astype(bool)
    minimum = max(
        NATURAL_HOLE_MINIMUM_PIXELS,
        round(class_map.size * NATURAL_HOLE_MINIMUM_IMAGE_FRACTION),
    )
    for component_index in range(1, count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) < minimum:
            continue
        component = component_labels == component_index
        component_classes = class_map[component]
        class_ids, class_counts = np.unique(
            component_classes,
            return_counts=True,
        )
        dominant = int(class_ids[int(np.argmax(class_counts))])
        class_mix = {
            str(int(value)): int(value_count)
            for value, value_count in zip(class_ids, class_counts)
        }
        transient_fraction = float(
            np.isin(component_classes, transient_ids).mean()
        )
        existing_policy_fraction = float(
            np.isin(component_classes, existing_policy_ids).mean()
        )
        natural_fraction = float(
            np.isin(component_classes, natural_ids).mean()
        )
        component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[component_index, cv2.CC_STAT_AREA])
        record_base: dict[str, object] = {
            "source": "class_agnostic_corridor_hole",
            "component_index": component_index,
            "source_pixels": int(component.sum()),
            "dominant_class_id": dominant,
            "class_mix": class_mix,
            "transient_fraction": transient_fraction,
            "existing_policy_fraction": existing_policy_fraction,
            "natural_fraction": natural_fraction,
            "component_bbox": [
                int(stats[component_index, cv2.CC_STAT_LEFT]),
                int(stats[component_index, cv2.CC_STAT_TOP]),
                component_width,
                component_height,
            ],
            "accepted": False,
        }
        if (
            transient_fraction
            >= NATURAL_HOLE_TRANSIENT_REJECTION_FRACTION
        ):
            record_base["rejection_reason"] = "person_or_rider_occlusion"
            records.append(record_base)
            rejected_sources |= component
            continue
        if (
            existing_policy_fraction
            >= NATURAL_HOLE_EXISTING_POLICY_REJECTION_FRACTION
        ):
            record_base["rejection_reason"] = "owned_by_existing_object_policy"
            records.append(record_base)
            rejected_sources |= component
            continue
        overlap = int((component & covered).sum())
        if overlap >= max(
            NATURAL_HOLE_COVERAGE_MINIMUM_PIXELS,
            round(component.sum() * NATURAL_HOLE_COVERAGE_FRACTION),
        ):
            record_base["rejection_reason"] = (
                "already_covered_by_current_v2_footprint"
            )
            record_base["current_v2_coverage_pixels"] = overlap
            records.append(record_base)
            rejected_sources |= component
            continue
        minimum_height = max(
            NATURAL_HOLE_MINIMUM_HEIGHT_PX,
            round(
                class_map.shape[0]
                * NATURAL_HOLE_MINIMUM_IMAGE_HEIGHT_FRACTION
            ),
        )
        minimum_area = max(
            NATURAL_HOLE_MINIMUM_AREA_PX,
            round(
                class_map.size * NATURAL_HOLE_MINIMUM_IMAGE_AREA_FRACTION
            ),
        )
        if component_height < minimum_height or component_area < minimum_area:
            record_base["rejection_reason"] = "shallow_or_tiny_surface_fragment"
            records.append(record_base)
            rejected_sources |= component
            continue
        if natural_fraction < NATURAL_HOLE_MINIMUM_NATURAL_FRACTION:
            record_base["rejection_reason"] = (
                "insufficient_natural_occluder_support"
            )
            records.append(record_base)
            rejected_sources |= component
            continue
        footprint = estimate_natural_ground_footprint(
            component,
            envelope,
            source_walkway_mask,
            class_map,
            dominant,
            component_index,
        )
        metadata = footprint.to_legacy_metadata()
        metadata.update(record_base)
        metadata["accepted"] = bool(footprint.footprint_mask.any())
        metadata["rejection_reason"] = (
            None
            if footprint.footprint_mask.any()
            else metadata.get("rejection_reason")
        )
        records.append(metadata)
        if metadata["accepted"]:
            footprints |= footprint.footprint_mask
            source_occluders |= footprint.source_mask
            accepted_sources |= component
        else:
            rejected_sources |= component
    return NaturalHoleEvidence(
        footprints,
        source_occluders,
        accepted_sources,
        rejected_sources,
        tuple(records),
        envelope,
    )
