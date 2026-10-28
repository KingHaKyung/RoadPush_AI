"""Conditional bounds for the original metres-first maximin corridor objective.

Unknown may be occupied or free. Bounds enclose every such completion; they do
not certify boundary identity, ray-observed free volume, or surveyed width.
"""

from pathlib import Path
import sys
from .measure_v5 import measure, merge
from .bottleneck import select


def path_width(layers):
    path = select(layers)
    if path is None:
        return (0.0, None)
    return (min((layers[i][j][1] - layers[i][j][0] for i, j in enumerate(path))), path)


def certificate(rows):
    if not rows:
        return dict(state="no_bilateral_domain", width_bounds_m=None, ratio_bounds=None)
    for a, b in zip(rows, rows[1:]):
        if b["station_index"] != a["station_index"] + 1:
            raise ValueError("consecutive domain required")
    lower = []
    upper = []
    denominators = []
    for r in rows:
        span = r["structural_span_m"]
        d = span[1] - span[0]
        if d <= 0:
            raise ValueError("positive bilateral denominator required")
        lower.append(merge(r["free_intervals_m"]))
        upper.append(merge(r["free_intervals_m"] + r["unknown_intervals_m"]))
        denominators.append(d)
    low, pathlow = path_width(lower)
    high, pathhigh = path_width(upper)
    unknown = any((r["unknown_intervals_m"] for r in rows))
    ratio = [max(0.0, low / max(denominators)), min(1.0, high / min(denominators))]
    return dict(
        state=(
            "conditional_interval" if unknown else "conditional_fully_supported_domain"
        ),
        station_range=[rows[0]["station_index"], rows[-1]["station_index"]],
        width_bounds_m=[low, high],
        ratio_bounds=ratio,
        full_width_bounds_m=[min(denominators), max(denominators)],
        unknown_present=unknown,
        known_blockage_proven=high == 0,
        domain_station_count=len(rows),
        representative_minimum5_policy_met=len(rows) >= 5,
        lower_path=pathlow,
        upper_path=pathhigh,
        meaning="Bounds on metres-first optimal bottleneck and its same-section ratio under any unknown completion; conditional on fixed structural spans.",
        independent_truth_verified=False,
        production_eligible=False,
        exact_ratio_available=not unknown,
    )


def all_domains(sections):
    groups = []
    current = []
    for row in sections:
        if current and row["station_index"] != current[-1]["station_index"] + 1:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return [certificate(rows) for rows in groups if len(rows) >= 2]
