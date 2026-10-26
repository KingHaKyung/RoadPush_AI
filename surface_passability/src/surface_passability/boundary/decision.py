"""Final production composition for straight-boundary decisions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from surface_passability.boundary.core_fit import (
    fit_continuous_core_boundary,
)
from surface_passability.boundary.direction import (
    apply_runwise_direction_consistency,
)
from surface_passability.boundary.geometric_support import (
    fit_boundary_with_geometric_gaps,
)
from surface_passability.boundary.vegetation import (
    apply_grounded_vegetation_cap,
)
from surface_passability.policies.boundary_decision import (
    StraightBoundaryDecisionPolicy,
)


POLICY_ID = "phase7_trapezoid_boundary_retry_v6"


def fit_straight_boundary(
    selected_surface: str | None,
    selection_state: str,
    structural_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    *,
    semantic_labels: np.ndarray | None = None,
    obstacle_silhouette_mask: np.ndarray | None = None,
    raw_structural_mask: np.ndarray | None = None,
    policy: StraightBoundaryDecisionPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the accepted straight-boundary calculation and safety gates."""

    policy = policy or StraightBoundaryDecisionPolicy()
    fit_kwargs: dict[str, Any] = {
        "core_policy": policy.continuous_core,
        "direction_policy": policy.direction,
        "geometric_gap_policy": policy.geometric_gap,
    }
    if obstacle_silhouette_mask is not None:
        fit_kwargs["obstacle_silhouette_mask"] = obstacle_silhouette_mask
    if raw_structural_mask is not None:
        fit_kwargs["raw_structural_mask"] = raw_structural_mask

    if selected_surface is None or structural_mask is None:
        result, masks = fit_boundary_with_geometric_gaps(
            selected_surface,
            selection_state,
            structural_mask,
            blocker_mask,
            policy=policy.geometric_gap,
        )
        result, masks = apply_runwise_direction_consistency(
            result,
            masks,
            mode="candidate_gate",
            direction_policy=policy.direction,
        )
    else:
        result, masks = fit_continuous_core_boundary(
            selected_surface,
            selection_state,
            structural_mask,
            blocker_mask,
            **fit_kwargs,
        )

    if selected_surface is not None and structural_mask is not None:
        if semantic_labels is None:
            raise ValueError(
                "semantic_labels are required for a selected surface"
            )
        result, masks = apply_grounded_vegetation_cap(
            result,
            masks,
            semantic_labels,
            policy.vegetation_cap,
        )

    result["policy_id"] = POLICY_ID
    result["v6_contract"] = {
        "baseline": "retry_v5_promoted_from_v4_probe_a",
        "component_order": [
            "v5_probe_1_geometric_gap",
            "refit",
            "v5_probe_2_runwise_direction",
            "v5_probe_3_continuous_core",
            "v5_probe_4_grounded_vegetation_cap",
        ],
        "historical_diagnostic_field_map": {
            "v5_probe_1": "probe_g_geometric_gap",
            "v5_probe_2": "probe_b_direction_consistency",
            "v5_probe_3": "probe_c_continuous_core",
            "v5_probe_4": "probe_v_grounded_vegetation_cap",
        },
        "policies": {
            "v5_probe_1": asdict(policy.geometric_gap),
            "v5_probe_2": asdict(policy.direction),
            "v5_probe_3": asdict(policy.continuous_core),
            "v5_probe_4": asdict(policy.vegetation_cap),
        },
        "v5_probe_5_ownership_gate_included": False,
        "candidate_fallback_included": False,
        "joint_refit_ablation_included": False,
        "extension_included": False,
        "curve_model_included": False,
    }
    return result, masks
