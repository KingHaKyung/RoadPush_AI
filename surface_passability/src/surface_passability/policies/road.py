"""Accepted Road construction thresholds."""

from surface_passability.contracts.road import (
    RoadContextPolicy,
    RoadDrainRestorationPolicy,
    RoadOwnershipPolicy,
    RoadOwnershipPersistencePolicy,
    RoadOwnershipRouteQualityPolicy,
    RoadOwnershipTrackPolicy,
    SurfaceOwnershipProfile,
)
from surface_passability.contracts.surfaces import SurfaceProfile


ROAD_CLASS_ID = 13
SERVICE_LANE_CLASS_ID = 14
PARKING_CLASS_ID = 10
BIKE_LANE_CLASS_ID = 7
CURB_CLASS_ID = 2
CURB_CUT_CLASS_ID = 9
SIDEWALK_CLASS_ID = 15

ROAD_REPAIR_CLASS_IDS = (8, 23, 24, 36, 41)
ROAD_REPAIR_MINIMUM_RADIUS_PX = 3
ROAD_REPAIR_RADIUS_FRACTION = 1 / 420
ROAD_REPAIR_MINIMUM_RING_FRACTION = 0.02
ROAD_REPAIR_MINIMUM_SUPPORT_PIXELS = 3

ROAD_REPAIR_CATCH_BASIN_CLASS_ID = 36
ROAD_REPAIR_CATCH_BASIN_MAXIMUM_IMAGE_FRACTION = 0.004
ROAD_UNION_SEAM_RADIUS_PX = 2

ROAD_SOURCE_CLASS_IDS = (
    ROAD_CLASS_ID,
    SERVICE_LANE_CLASS_ID,
    PARKING_CLASS_ID,
)
ROAD_FORBIDDEN_CLASS_IDS = (CURB_CLASS_ID, CURB_CUT_CLASS_ID)

ROAD_SURFACE_PROFILE = SurfaceProfile(
    name="road",
    source_class_ids=ROAD_SOURCE_CLASS_IDS,
    repair_class_ids=ROAD_REPAIR_CLASS_IDS,
    forbidden_class_ids=ROAD_FORBIDDEN_CLASS_IDS,
    side_pruning_enabled=True,
)

DEFAULT_ROAD_CONTEXT_POLICY = RoadContextPolicy(
    source_class_ids=ROAD_SOURCE_CLASS_IDS,
    repair_class_ids=ROAD_REPAIR_CLASS_IDS,
    repair_radius_fraction=ROAD_REPAIR_RADIUS_FRACTION,
    minimum_ring_fraction=ROAD_REPAIR_MINIMUM_RING_FRACTION,
    minimum_support_pixels=ROAD_REPAIR_MINIMUM_SUPPORT_PIXELS,
    union_seam_radius_px=ROAD_UNION_SEAM_RADIUS_PX,
    forbidden_class_ids=ROAD_FORBIDDEN_CLASS_IDS,
    trusted_sidewalk_uses_all_repaired_components=True,
    sidewalk_outer_row_span_used_as_guard=False,
    bike_lane_included=False,
)

DEFAULT_ROAD_OWNERSHIP_POLICY = RoadOwnershipPolicy(
    positive_ownership_policy="curb_sidewalk_envelope",
    positive_ownership_before_repair=True,
    raw_road_ownership_enabled=True,
    service_lane_ownership_policy="conservative_identity_insufficient_data",
    parking_ownership_policy="conservative_identity_curb_adjacent_allowed",
    obstacle_or_footprint_used_for_ownership=False,
)

ROAD_OWNERSHIP_MINIMUM_ROUTE_PROGRESS_FRACTION = 0.20
ROAD_OWNERSHIP_MAXIMUM_ROUTE_ANGLE_DEGREES = 60.0
ROAD_OWNERSHIP_MAXIMUM_LINE_RESIDUAL_FRACTION = 0.090

DEFAULT_ROAD_OWNERSHIP_ROUTE_QUALITY_POLICY = (
    RoadOwnershipRouteQualityPolicy(
        minimum_vertical_progress_fraction=(
            ROAD_OWNERSHIP_MINIMUM_ROUTE_PROGRESS_FRACTION
        ),
        maximum_absolute_angle_degrees=(
            ROAD_OWNERSHIP_MAXIMUM_ROUTE_ANGLE_DEGREES
        ),
        maximum_line_residual_fraction=(
            ROAD_OWNERSHIP_MAXIMUM_LINE_RESIDUAL_FRACTION
        ),
    )
)

ROAD_OWNERSHIP_ROUTE_MARGIN_FRACTION = 0.012
DEFAULT_ROAD_OWNERSHIP_TRACK_POLICY = RoadOwnershipTrackPolicy(
    route_margin_fraction=ROAD_OWNERSHIP_ROUTE_MARGIN_FRACTION,
)

ROAD_OWNERSHIP_LOCAL_WINDOW_FRACTION = 0.030
ROAD_OWNERSHIP_MINIMUM_LOCAL_SUPPORT_FRACTION = 0.45
ROAD_OWNERSHIP_MAXIMUM_GAP_FRACTION = 0.035
ROAD_OWNERSHIP_MAXIMUM_TOTAL_INTERPOLATED_ROWS_FRACTION = 0.075
ROAD_OWNERSHIP_MAXIMUM_STEP_FRACTION = 0.045
ROAD_OWNERSHIP_FRAME_MARGIN_FRACTION = 0.012

DEFAULT_ROAD_OWNERSHIP_PERSISTENCE_POLICY = (
    RoadOwnershipPersistencePolicy(
        local_window_fraction=ROAD_OWNERSHIP_LOCAL_WINDOW_FRACTION,
        minimum_local_support_fraction=(
            ROAD_OWNERSHIP_MINIMUM_LOCAL_SUPPORT_FRACTION
        ),
        maximum_gap_fraction=ROAD_OWNERSHIP_MAXIMUM_GAP_FRACTION,
        maximum_step_fraction=ROAD_OWNERSHIP_MAXIMUM_STEP_FRACTION,
        frame_margin_fraction=ROAD_OWNERSHIP_FRAME_MARGIN_FRACTION,
    )
)

SIDEWALK_IDENTITY_OWNERSHIP = SurfaceOwnershipProfile(
    surface_name="sidewalk",
    policy_name="identity_no_op_frozen_sidewalk",
    enabled=False,
    owned_class_ids=(SIDEWALK_CLASS_ID,),
    preserve_class_ids=(),
    curb_class_ids=(),
    sidewalk_class_ids=(),
)

ROAD_CURB_SIDEWALK_ENVELOPE = SurfaceOwnershipProfile(
    surface_name="road",
    policy_name="curb_sidewalk_envelope",
    enabled=True,
    owned_class_ids=(ROAD_CLASS_ID,),
    preserve_class_ids=(SERVICE_LANE_CLASS_ID, PARKING_CLASS_ID),
    curb_class_ids=(CURB_CLASS_ID, CURB_CUT_CLASS_ID),
    sidewalk_class_ids=(SIDEWALK_CLASS_ID,),
    local_window_fraction=ROAD_OWNERSHIP_LOCAL_WINDOW_FRACTION,
    minimum_local_support_fraction=(
        ROAD_OWNERSHIP_MINIMUM_LOCAL_SUPPORT_FRACTION
    ),
    maximum_gap_fraction=ROAD_OWNERSHIP_MAXIMUM_GAP_FRACTION,
    maximum_total_interpolated_rows_fraction=(
        ROAD_OWNERSHIP_MAXIMUM_TOTAL_INTERPOLATED_ROWS_FRACTION
    ),
    maximum_step_fraction=ROAD_OWNERSHIP_MAXIMUM_STEP_FRACTION,
    minimum_route_progress_fraction=(
        ROAD_OWNERSHIP_MINIMUM_ROUTE_PROGRESS_FRACTION
    ),
    maximum_route_angle_deg=ROAD_OWNERSHIP_MAXIMUM_ROUTE_ANGLE_DEGREES,
    maximum_line_residual_fraction=(
        ROAD_OWNERSHIP_MAXIMUM_LINE_RESIDUAL_FRACTION
    ),
    frame_margin_fraction=ROAD_OWNERSHIP_FRAME_MARGIN_FRACTION,
    route_margin_fraction=ROAD_OWNERSHIP_ROUTE_MARGIN_FRACTION,
)

ROAD_DRAIN_RESTORATION_POLICY_ID = (
    "phase3_road_context_v2_drain_ownership_v1"
)
ROAD_DRAIN_RESTORATION_STAGE_ORDER = (
    "base_road_and_linear_drain_evidence",
    "drain_route_informed_raw_road_positive_ownership",
    "approved_linear_drain_whole_object_restore",
    "existing_component_selection_and_row_span",
    "existing_structural_side_pruning",
    "existing_blocker_subtraction",
)

DEFAULT_ROAD_DRAIN_RESTORATION_POLICY = RoadDrainRestorationPolicy(
    policy_id=ROAD_DRAIN_RESTORATION_POLICY_ID,
    linear_transverse_drain_enabled=True,
    linear_drain_whole_object_restoration=True,
    road_positive_ownership_enabled=True,
    ordered_stages=ROAD_DRAIN_RESTORATION_STAGE_ORDER,
)
