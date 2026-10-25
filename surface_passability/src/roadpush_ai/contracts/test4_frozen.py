from __future__ import annotations

from roadpush_ai.config.settings import get_settings

from pathlib import Path


ROOT = get_settings().project_root

BASELINE_ID = "v3_test4_sidewalk_v1"
BASELINE_OUTPUT = (
    ROOT
    / "outputs"
    / "walkway_detection_v3"
    / "baseline_test4_sidewalk_40_v1"
)

# Frozen blocker policy inherited from the accepted Test4 candidate v3.
WHEELED_SILHOUETTE_ANGLE_MAX_DEG = 85.0
BANNER_SUPPORT_GROUP_MODE = "grounded_all"

# Frozen obstacle-connected component policy.
MINIMUM_COMPONENT_PIXELS = 20
MINIMUM_COMPONENT_ROWS = 3
MINIMUM_CONTACT_PIXELS_EACH_SIDE = 3
CONTACT_RADIUS_IMAGE_FRACTION = 0.010
MINIMUM_FORWARD_PROJECTION_PX = 2.0
MINIMUM_ROUTE_ALIGNMENT_COSINE = 0.50

# Execution contract.
EXECUTION_DEVICE = "cpu"
EXPECTED_SAMPLE_COUNT = 40

