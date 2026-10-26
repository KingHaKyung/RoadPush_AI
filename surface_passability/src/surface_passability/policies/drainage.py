"""Accepted linear-drain geometry thresholds."""

from surface_passability.contracts.drainage import LinearDrainPolicy


CATCH_BASIN_CLASS_ID = 36

LINEAR_DRAIN_MINIMUM_COMPONENT_PIXELS = 20
LINEAR_DRAIN_MINIMUM_PCA_ELONGATION = 6.0
LINEAR_DRAIN_MINIMUM_TRANSVERSE_ANGLE_DEGREES = 60.0
LINEAR_DRAIN_STRONG_MINIMUM_TRANSVERSE_ANGLE_DEGREES = 55.0
LINEAR_DRAIN_STRONG_MINIMUM_PCA_ELONGATION = 10.0
LINEAR_DRAIN_STRONG_MINIMUM_SIDE_SUPPORT_FRACTION = 0.75
LINEAR_DRAIN_STRONG_MINIMUM_COMPONENT_FRACTION = 0.70
LINEAR_DRAIN_MAXIMUM_MINOR_SPAN_FRACTION = 0.18
LINEAR_DRAIN_MAXIMUM_COMPONENT_FRACTION = 0.08
LINEAR_DRAIN_SUPPORT_DEPTH_FRACTION = 0.035
LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_PIXELS = 20
LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_FRACTION = 0.18
LINEAR_DRAIN_MINIMUM_CORRIDOR_APPROVED_PIXELS = 20
LINEAR_DRAIN_MINIMUM_CORRIDOR_COMPONENT_FRACTION = 0.20

DRAIN_CORRIDOR_BAND_DEPTH_FRACTION = 0.09
DRAIN_CORRIDOR_MARGIN_PX = 2


def build_linear_drain_policy(
    *,
    restore_whole_accepted_component: bool,
    production_policy: bool,
) -> LinearDrainPolicy:
    """Build the exact serializable policy for one aggregate evaluation."""

    return LinearDrainPolicy(
        probe_only=not production_policy,
        production_policy_modified=production_policy,
        catch_basin_class_id=CATCH_BASIN_CLASS_ID,
        minimum_component_pixels=LINEAR_DRAIN_MINIMUM_COMPONENT_PIXELS,
        minimum_pca_elongation=LINEAR_DRAIN_MINIMUM_PCA_ELONGATION,
        minimum_transverse_angle_deg=(
            LINEAR_DRAIN_MINIMUM_TRANSVERSE_ANGLE_DEGREES
        ),
        strong_minimum_transverse_angle_deg=(
            LINEAR_DRAIN_STRONG_MINIMUM_TRANSVERSE_ANGLE_DEGREES
        ),
        strong_minimum_pca_elongation=(
            LINEAR_DRAIN_STRONG_MINIMUM_PCA_ELONGATION
        ),
        strong_minimum_side_support_fraction_each=(
            LINEAR_DRAIN_STRONG_MINIMUM_SIDE_SUPPORT_FRACTION
        ),
        strong_minimum_corridor_component_fraction=(
            LINEAR_DRAIN_STRONG_MINIMUM_COMPONENT_FRACTION
        ),
        maximum_minor_span_fraction=(
            LINEAR_DRAIN_MAXIMUM_MINOR_SPAN_FRACTION
        ),
        maximum_component_fraction=(
            LINEAR_DRAIN_MAXIMUM_COMPONENT_FRACTION
        ),
        support_depth_fraction=LINEAR_DRAIN_SUPPORT_DEPTH_FRACTION,
        minimum_side_support_pixels=(
            LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_PIXELS
        ),
        minimum_side_support_fraction=(
            LINEAR_DRAIN_MINIMUM_SIDE_SUPPORT_FRACTION
        ),
        minimum_corridor_approved_pixels=(
            LINEAR_DRAIN_MINIMUM_CORRIDOR_APPROVED_PIXELS
        ),
        minimum_corridor_component_fraction=(
            LINEAR_DRAIN_MINIMUM_CORRIDOR_COMPONENT_FRACTION
        ),
        corridor_band_depth_fraction=DRAIN_CORRIDOR_BAND_DEPTH_FRACTION,
        corridor_margin_px=DRAIN_CORRIDOR_MARGIN_PX,
        approval_clipped_to_estimated_corridor=(
            not restore_whole_accepted_component
        ),
        whole_accepted_component_restoration=(
            restore_whole_accepted_component
        ),
    )
