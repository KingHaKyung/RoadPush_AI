"""Separate low vertical barrier family, not a replacement for permanent-wall truth."""

from pathlib import Path
import sys
from .measure_v5 import evaluate, Policy


def evaluate_barrier(record, prior, p, labels, scores):
    policy = Policy(min_wall_height_m=0.35, min_wall_points=30, min_wall_span_m=1.0)
    result = evaluate(record, prior, p, labels, scores, policy)
    result.update(
        boundary_family="low_continuous_vertical_barrier_hypothesis",
        permanent_structure_verified=False,
        production_eligible=False,
        policy_reason="separate structural-family hypothesis; not selected by numeric yield; not operational adoption",
    )
    return result
