"""Authoritative Phase 7 Retry V5 boundary API.

The implementation remains in the preserved global-support experiment module
so its promotion is traceable to the reviewed A probe. New callers should use
this module rather than the historical probe entry point.
"""

from roadpush_ai.phase7a.retry_v5_global_support import (
    POLICY_ID,
    TrapezoidBoundaryRetryV5Policy,
    fit_trapezoid_boundary_retry_v5,
)

__all__ = [
    "POLICY_ID",
    "TrapezoidBoundaryRetryV5Policy",
    "fit_trapezoid_boundary_retry_v5",
]
