"""Project relative passability into nullable physical-width results."""

from __future__ import annotations

import math
from numbers import Real

from surface_passability.contracts.passability import (
    PassabilityBoundaryHandoff,
    PassabilityState,
    PhysicalPassabilityResult,
    RelativePassabilityMetrics,
)


def optional_finite_number(
    name: str,
    value: float | None,
    *,
    minimum: float,
    inclusive: bool,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    valid = converted >= minimum if inclusive else converted > minimum
    if not valid:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum}")
    return converted


def validated_minimum_ratio(
    metrics: RelativePassabilityMetrics,
) -> float | None:
    ratio = metrics.minimum_passable_ratio
    if not metrics.representative_relative_metrics_available:
        if ratio is not None:
            raise ValueError(
                "unavailable relative metrics cannot expose a minimum ratio"
            )
        return None
    if ratio is None or isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise ValueError(
            "available relative metrics require a numeric minimum ratio"
        )
    converted = float(ratio)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(
            "minimum_passable_ratio must be finite and within [0, 1]"
        )
    return converted


def calculate_physical_width(
    handoff: PassabilityBoundaryHandoff,
    metrics: RelativePassabilityMetrics,
    *,
    surface_width_m: float | None = None,
    agent_width_m: float | None = None,
    safety_factor: float | None = None,
) -> PhysicalPassabilityResult:
    """Apply optional real-world widths without inferring metres from pixels."""

    if handoff.sample_id != metrics.sample_id:
        raise ValueError("sample_id mismatch between handoff and relative metrics")
    if handoff.measurement_scope != metrics.measurement_scope:
        raise ValueError(
            "measurement_scope mismatch between handoff and relative metrics"
        )
    width = optional_finite_number(
        "surface_width_m", surface_width_m, minimum=0.0, inclusive=False
    )
    agent = optional_finite_number(
        "agent_width_m", agent_width_m, minimum=0.0, inclusive=False
    )
    safety = optional_finite_number(
        "safety_factor", safety_factor, minimum=1.0, inclusive=True
    )
    ratio = validated_minimum_ratio(metrics)

    absolute = width * ratio if width is not None and ratio is not None else None
    effective_blocked = (
        width - absolute
        if width is not None and absolute is not None
        else None
    )
    required_passable_ratio = (
        agent / width if agent is not None and width is not None else None
    )
    required_safe_width = (
        agent * safety if agent is not None and safety is not None else None
    )
    required_safe_ratio = (
        required_safe_width / width
        if required_safe_width is not None and width is not None
        else None
    )

    flags = list(metrics.flags)
    if handoff.measurement_scope == "partial":
        flags.append("partial_scope_only")
    if handoff.measurement_scope == "provisional":
        flags.append("provisional_review_only")
    if ratio is None:
        state: PassabilityState = "undetermined"
    elif ratio == 0.0:
        state = "blocked"
    elif width is None or agent is None:
        state = "relative_only"
    elif absolute is not None and absolute < agent:
        state = "blocked"
    elif safety is None:
        state = "relative_only"
        flags.append("safety_factor_not_supplied")
    elif (
        absolute is not None
        and required_safe_width is not None
        and absolute < required_safe_width
    ):
        state = "narrowed"
    else:
        state = "passable"

    if ratio is not None and width is None:
        flags.append("surface_width_not_supplied")
    if ratio is not None and agent is None:
        flags.append("agent_width_not_supplied")
    return PhysicalPassabilityResult(
        schema_version="phase7_b_physical_contract_v1",
        sample_id=handoff.sample_id,
        measurement_scope=handoff.measurement_scope,
        human_review_required=handoff.human_review_required,
        measurement_confidence=metrics.measurement_confidence,
        minimum_passable_ratio=ratio,
        surface_width_m=width,
        absolute_passable_width_m=absolute,
        effective_blocked_width_m=effective_blocked,
        agent_width_m=agent,
        safety_factor=safety,
        required_safe_width_m=required_safe_width,
        required_passable_ratio=required_passable_ratio,
        required_safe_ratio=required_safe_ratio,
        passability_state=state,
        abstention_reasons=metrics.abstention_reasons,
        flags=tuple(dict.fromkeys(flags)),
    )
