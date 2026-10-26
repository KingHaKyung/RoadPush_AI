"""Complete production composition for surface-boundary analysis."""

from __future__ import annotations

import numpy as np

from surface_passability.boundary.decision import fit_straight_boundary
from surface_passability.boundary.extension import generate_boundary_extension
from surface_passability.boundary.grading import (
    classify_straight_boundary_disposition,
    grade_boundary_extension,
)
from surface_passability.boundary.occlusion import (
    build_containment_occluder_mask,
)
from surface_passability.boundary.safe_prefix import (
    evaluate_safe_extension_prefix,
)
from surface_passability.boundary.snapshot import freeze_boundary_observation
from surface_passability.contracts.boundary import BoundaryAnalysisResult
from surface_passability.obstacles.taxonomy import (
    BANNER_CLASS_ID,
    BOUNDARY_CONTAINMENT_OCCLUDER_CLASS_IDS,
)
from surface_passability.policies.boundary_analysis import (
    BoundaryAnalysisPolicy,
)


def analyze_surface_boundary(
    sample_id: str,
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray,
    blocker_mask: np.ndarray,
    semantic_labels: np.ndarray | None,
    *,
    obstacle_silhouette_mask: np.ndarray | None = None,
    raw_structural_mask: np.ndarray | None = None,
    policy: BoundaryAnalysisPolicy | None = None,
) -> BoundaryAnalysisResult:
    """Run the accepted straight fit, extension, validation, and grading once."""

    policy = policy or BoundaryAnalysisPolicy()
    straight_result, straight_masks = fit_straight_boundary(
        selected_surface,
        selection_state,
        structural_mask,
        blocker_mask,
        semantic_labels=semantic_labels,
        obstacle_silhouette_mask=obstacle_silhouette_mask,
        raw_structural_mask=raw_structural_mask,
        policy=policy.straight,
    )
    straight_result["final_disposition"] = (
        classify_straight_boundary_disposition(straight_result)
    )
    snapshot = freeze_boundary_observation(
        sample_id,
        straight_result,
        structural_mask.shape,
    )
    diagnostic, generation_masks = generate_boundary_extension(
        snapshot,
        structural_mask,
        blocker_mask,
        policy.extension,
    )
    occluder_mask = build_containment_occluder_mask(
        semantic_labels,
        structural_mask.shape,
        include_banner=True,
    )
    decision, selected_masks = evaluate_safe_extension_prefix(
        snapshot,
        diagnostic,
        generation_masks,
        occluder_mask,
        policy.containment,
        policy.safe_prefix,
    )
    decision["policy_id"] = f'{decision["policy_id"]}_banner32'
    decision["banner_occluder_enabled"] = True
    decision["occluder_class_ids"] = [
        *BOUNDARY_CONTAINMENT_OCCLUDER_CLASS_IDS,
        BANNER_CLASS_ID,
    ]
    decision, selected_masks = grade_boundary_extension(
        snapshot,
        decision,
        selected_masks,
        generation_masks,
        top_bottom_width_ratio=straight_result.get("envelope_fit", {}).get(
            "top_bottom_width_ratio"
        ),
        policy=policy.grading,
    )
    return BoundaryAnalysisResult(
        snapshot=snapshot,
        straight_result=straight_result,
        straight_masks=straight_masks,
        generation_diagnostic=diagnostic,
        generation_masks=generation_masks,
        occluder_mask=occluder_mask,
        decision=decision,
        selected_masks=selected_masks,
    )
