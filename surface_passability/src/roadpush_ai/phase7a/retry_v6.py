"""Authoritative Phase 7 Retry V6 straight-boundary API.

Retry V6 promotes the reviewed Retry V5 Probe 1-4 composition. Historical
probe-named diagnostic fields remain present for artifact traceability, while
new callers use this single versioned entry point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from roadpush_ai.phase7a.retry_v5_runwise import (
    RunwiseDirectionPolicy,
    apply_runwise_direction_consistency,
)
from roadpush_ai.phase7a.retry_v5_joint_fit import (
    ContinuousCorePolicy,
    fit_trapezoid_boundary_retry_v5_probe_c,
)
from roadpush_ai.phase7a.retry_v5_support_window import (
    GeometricGapPolicy,
    fit_trapezoid_boundary_retry_v5_probe_g,
)
from roadpush_ai.phase7a.retry_v5_vegetation import (
    GroundedVegetationCapPolicy,
    apply_grounded_vegetation_cap,
)


POLICY_ID = "phase7_trapezoid_boundary_retry_v6"


@dataclass(frozen=True)
class TrapezoidBoundaryRetryV6Policy:
    probe_1_geometric_gap: GeometricGapPolicy = field(
        default_factory=GeometricGapPolicy
    )
    probe_2_direction: RunwiseDirectionPolicy = field(
        default_factory=RunwiseDirectionPolicy
    )
    probe_3_continuous_core: ContinuousCorePolicy = field(
        default_factory=ContinuousCorePolicy
    )
    probe_4_vegetation_cap: GroundedVegetationCapPolicy = field(
        default_factory=GroundedVegetationCapPolicy
    )


def fit_trapezoid_boundary_retry_v6(
    selected_surface: str | None,
    selector_state: str,
    structural_mask: np.ndarray | None,
    blocker_mask: np.ndarray | None,
    *,
    semantic_labels: np.ndarray | None = None,
    obstacle_silhouette_mask: np.ndarray | None = None,
    raw_structural_mask: np.ndarray | None = None,
    policy: TrapezoidBoundaryRetryV6Policy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy = policy or TrapezoidBoundaryRetryV6Policy()
    fit_kwargs: dict[str, Any] = {
        "core_policy": policy.probe_3_continuous_core,
        "direction_policy": policy.probe_2_direction,
        "geometric_gap_policy": policy.probe_1_geometric_gap,
    }
    if obstacle_silhouette_mask is not None:
        fit_kwargs["obstacle_silhouette_mask"] = obstacle_silhouette_mask
    if raw_structural_mask is not None:
        fit_kwargs["raw_structural_mask"] = raw_structural_mask
    if selected_surface is None or structural_mask is None:
        result, masks = fit_trapezoid_boundary_retry_v5_probe_g(
            selected_surface,
            selector_state,
            structural_mask,
            blocker_mask,
            policy=policy.probe_1_geometric_gap,
        )
        result, masks = apply_runwise_direction_consistency(
            result,
            masks,
            mode="candidate_gate",
            direction_policy=policy.probe_2_direction,
        )
    else:
        result, masks = fit_trapezoid_boundary_retry_v5_probe_c(
            selected_surface,
            selector_state,
            structural_mask,
            blocker_mask,
            **fit_kwargs,
        )

    if selected_surface is not None and structural_mask is not None:
        if semantic_labels is None:
            raise ValueError("semantic_labels are required for a selected surface")
        result, masks = apply_grounded_vegetation_cap(
            result,
            masks,
            semantic_labels,
            policy.probe_4_vegetation_cap,
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
            "v5_probe_1": asdict(policy.probe_1_geometric_gap),
            "v5_probe_2": asdict(policy.probe_2_direction),
            "v5_probe_3": asdict(policy.probe_3_continuous_core),
            "v5_probe_4": asdict(policy.probe_4_vegetation_cap),
        },
        "v5_probe_5_ownership_gate_included": False,
        "candidate_fallback_included": False,
        "joint_refit_ablation_included": False,
        "extension_included": False,
        "curve_model_included": False,
    }
    return result, masks


__all__ = [
    "POLICY_ID",
    "TrapezoidBoundaryRetryV6Policy",
    "fit_trapezoid_boundary_retry_v6",
]
