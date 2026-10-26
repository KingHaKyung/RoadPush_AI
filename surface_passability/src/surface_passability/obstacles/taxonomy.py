"""Mapillary semantic class groups used by surface and obstacle analysis."""

from __future__ import annotations


SEMANTIC_CLASS_COUNT = 65

SIDEWALK_CLASS_ID = 15
CAR_CLASS_ID = 55
VEHICLE_INSTANCE_CLASS_NAMES = frozenset(("car", "truck", "bus"))
BANNER_CLASS_ID = 32
GROUND_ANIMAL_CLASS_ID = 1
TRANSIENT_SOURCE_CLASS_IDS = (
    GROUND_ANIMAL_CLASS_ID,
    19,
    20,
    21,
    22,
)
WHEELED_OBSTACLE_CLASS_IDS = (52, 54, 55, 56, 57, 58, 59, 60, 61, 62)
BANNER_SUPPORT_CLASS_IDS = (35, 45, 46, 47, 49, 50)
SURFACE_REPAIR_CLASS_IDS = (24, 36, 41)
CONNECTION_CLASS_IDS = (8, 9, 23)
BOUNDARY_CLASS_IDS = (2, 3, 4, 5, 6, 17)
BOUNDARY_CONTEXT_CLASS_IDS = (2, 9, 10, 13, 14)
REPAIR_ROAD_CONTEXT_CLASS_IDS = (10, 13, 14)

FIXED_OCCUPANCY_CLASS_IDS = (
    33,
    34,
    35,
    38,
    39,
    40,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
)
MOVING_OCCUPANCY_CLASS_IDS = (1, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62)
OCCUPANCY_CLASS_IDS = FIXED_OCCUPANCY_CLASS_IDS + MOVING_OCCUPANCY_CLASS_IDS
TRANSIENT_OCCLUSION_CLASS_IDS = (19, 20, 21, 22)
HAZARD_CLASS_IDS = (12, 26, 28, 31, 43)
NATURAL_NON_WALKABLE_CLASS_IDS = (25, 29, 30)
EXCLUDED_GROUND_CLASS_IDS = (7, 10, 11, 13, 14)
STRUCTURE_CONTEXT_CLASS_IDS = (16, 18)
CONTEXT_ONLY_CLASS_IDS = (0, 27, 32, 37, 53, 63, 64)

CONDITIONAL_GROUNDED_OCCUPANCY_CLASS_IDS = (32,)
WIDTH_IGNORED_TRANSIENT_CLASS_IDS = TRANSIENT_OCCLUSION_CLASS_IDS

# Exact semantic classes allowed to explain a same-side boundary containment
# gap. Ground, hazard, boundary, vegetation, and surface classes are absent.
BOUNDARY_CONTAINMENT_OCCLUDER_CLASS_IDS = (
    TRANSIENT_SOURCE_CLASS_IDS
    + FIXED_OCCUPANCY_CLASS_IDS
    + WHEELED_OBSTACLE_CLASS_IDS
)

# These groups are the accepted, mutually exclusive semantic policy partition.
# Derived context groups above intentionally overlap this partition.
SEMANTIC_POLICY_PARTITION = (
    ("basic_walkway", (SIDEWALK_CLASS_ID,)),
    ("conditional_surface_inherit", SURFACE_REPAIR_CLASS_IDS),
    ("connection_only", CONNECTION_CLASS_IDS),
    ("boundary", BOUNDARY_CLASS_IDS),
    ("structure_context", STRUCTURE_CONTEXT_CLASS_IDS),
    ("fixed_occupancy_candidate", FIXED_OCCUPANCY_CLASS_IDS),
    ("moving_occupancy_candidate", MOVING_OCCUPANCY_CLASS_IDS),
    ("dynamic_occlusion", TRANSIENT_OCCLUSION_CLASS_IDS),
    ("hazard_or_unknown", HAZARD_CLASS_IDS),
    ("natural_nonwalk", NATURAL_NON_WALKABLE_CLASS_IDS),
    ("excluded_ground_observation", EXCLUDED_GROUND_CLASS_IDS),
    ("context_only", CONTEXT_ONLY_CLASS_IDS),
)


def semantic_policy_partition_ids() -> tuple[int, ...]:
    """Return the accepted class partition in policy order."""

    return tuple(
        class_id
        for _, class_ids in SEMANTIC_POLICY_PARTITION
        for class_id in class_ids
    )

