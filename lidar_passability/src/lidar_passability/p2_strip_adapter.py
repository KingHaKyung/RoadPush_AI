"""Review-only P2 cells -> common axis-aligned strip -> existing P3 metric.

No physical boundary confirmation or ray-based free volume is required for a
provisional estimate. Neither is invented: source unknowns and fitted-boundary
provenance are retained. Direction and estimation policies must be explicit.
"""

from copy import deepcopy
import math
from .p2_gate import VERSION as P2_VERSION, validate_p2
from .width_measure import VERSION, measure, number

ADAPTER_VERSION = "p3_p2_strip_review_v1"


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 1e-10:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def complement(span, covered):
    out, cursor = ([], span[0])
    for a, b in merge(covered):
        if a > cursor + 1e-10:
            out.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < span[1] - 1e-10:
        out.append([cursor, span[1]])
    return out


def fit(anchors, max_residual):
    if len({a["s_m"] for a in anchors}) < 2:
        return None
    ms = sum((a["s_m"] for a in anchors)) / len(anchors)
    mt = sum((a["t_m"] for a in anchors)) / len(anchors)
    den = sum(((a["s_m"] - ms) ** 2 for a in anchors))
    slope = sum(((a["s_m"] - ms) * (a["t_m"] - mt) for a in anchors)) / den
    intercept = mt - slope * ms
    residual = max((abs(a["t_m"] - (slope * a["s_m"] + intercept)) for a in anchors))
    if residual > max_residual:
        return None
    return dict(
        slope=slope,
        intercept=intercept,
        max_residual_m=residual,
        observed_s_range_m=[
            min((a["s_m"] for a in anchors)),
            max((a["s_m"] for a in anchors)),
        ],
        anchors=anchors,
    )


def adapt_p2(record, spec):
    validate_p2(record)
    meta = record["metadata"]
    if meta.get("units") != "m" or meta.get("z_up_verified") is not True:
        raise ValueError("metric z-up frame evidence required")
    if record["grid"].get("axis_order") != "x,y":
        raise ValueError("grid axis order changed")
    axis = spec["longitudinal_axis"]
    if (
        isinstance(axis, bool)
        or axis not in (0, 1)
        or (not spec.get("direction_evidence"))
    ):
        raise ValueError("explicit x/y strip direction with evidence required")
    lateral = 1 - axis
    indexes = spec["station_indices"]
    if (
        not indexes
        or any((isinstance(i, bool) or not isinstance(i, int) for i in indexes))
        or indexes != list(range(indexes[0], indexes[-1] + 1))
    ):
        raise ValueError("consecutive station indices required")
    shape = record["grid"]["shape"]
    if indexes[0] < 0 or indexes[-1] >= shape[axis]:
        raise ValueError("requested strip outside P2 grid")
    resolution = number(record["config"]["resolution_m"])
    if resolution <= 0:
        raise ValueError("invalid resolution")
    max_extension = number(spec["max_boundary_extrapolation_m"])
    max_residual = number(spec["max_boundary_fit_residual_m"])
    if min(max_extension, max_residual) < 0:
        raise ValueError("negative boundary policy")
    tmin, tmax = map(number, spec["lateral_search_range_m"])
    if tmin >= tmax:
        raise ValueError("invalid lateral search domain")
    origin = list(map(number, record["grid"]["origin_xy_m"]))
    cells = {tuple(c["index"]): c for c in record["cells"]}
    if len(cells) > 10000 or len(indexes) > 2000:
        raise ValueError("review prototype input limit")
    anchors = {"left": [], "right": []}
    for edge in record["boundary_candidates"]:
        if edge["kind"] not in (
            "semantic_transition_candidate",
            "height_step_candidate",
        ):
            continue
        a, b = (tuple(edge["cell"]), tuple(edge["neighbor"]))
        if (
            a not in cells
            or b not in cells
            or a[axis] != b[axis]
            or (abs(a[lateral] - b[lateral]) != 1)
        ):
            continue
        if a[axis] not in indexes or cells[a]["surface"] != "sidewalk":
            continue
        s = origin[axis] + (a[axis] + 0.5) * resolution
        t = origin[lateral] + max(a[lateral], b[lateral]) * resolution
        if not tmin <= t <= tmax:
            continue
        source_ids = sorted(
            set(cells[a]["support_point_ids"]) | set(cells[b]["support_point_ids"])
        )
        if not source_ids:
            continue
        side = "left" if b[lateral] < a[lateral] else "right"
        anchors[side].append(
            dict(
                s_m=s,
                t_m=t,
                kind=edge["kind"],
                cell=list(a),
                neighbor=list(b),
                support_point_ids=source_ids,
                physical_boundary_confirmed=False,
            )
        )
    models = {side: fit(values, max_residual) for side, values in anchors.items()}
    sections = []
    for i in indexes:
        s = origin[axis] + (i + 0.5) * resolution
        row = dict(
            s_m=s,
            structural_span_m=None,
            boundary_supported=False,
            occupancy_observed=False,
            candidate_evidence_supported=False,
            free_intervals_m=None,
            unknown_intervals_m=[],
            boundary_provenance={},
            source_cell_indices=[],
            low_nonblocking_point_ids=[],
        )
        if all(models.values()):
            limits, supported = ([], True)
            for side in ("left", "right"):
                model = models[side]
                lo, hi = model["observed_s_range_m"]
                extrap = max(lo - s, s - hi, 0)
                supported &= extrap <= max_extension + 1e-10
                limits.append(model["slope"] * s + model["intercept"])
                direct = any(
                    (math.isclose(s, a["s_m"], abs_tol=1e-09) for a in model["anchors"])
                )
                row["boundary_provenance"][side] = dict(
                    kind=(
                        "observed_candidate"
                        if direct
                        else "fitted_extension" if extrap else "fitted_interpolation"
                    ),
                    extrapolation_m=extrap,
                    physical_boundary_confirmed=False,
                )
            if limits[0] < limits[1] and supported:
                row["structural_span_m"] = limits
                row["boundary_supported"] = True
                candidate, occupied, all_ids, low_ids = ([], [], [], [])
                for j in range(shape[lateral]):
                    key = (i, j) if axis == 0 else (j, i)
                    cell = cells.get(key)
                    a = max(limits[0], origin[lateral] + j * resolution)
                    b = min(limits[1], origin[lateral] + (j + 1) * resolution)
                    if a >= b or cell is None:
                        continue
                    row["source_cell_indices"].append(list(key))
                    all_ids.extend(cell["raw_point_ids"])
                    low_ids.extend(cell["low_nonblocking_point_ids"])
                    unresolved = cell["height_unresolved_point_ids"] or set(
                        cell["out_of_height_band_point_ids"]
                    ) - set(cell["low_nonblocking_point_ids"])
                    if cell["occupancy_candidate_point_ids"]:
                        occupied.append([a, b])
                    elif cell["surface"] == "sidewalk" and (not unresolved):
                        candidate.append([a, b])
                row.update(
                    free_intervals_m=merge(candidate),
                    unknown_intervals_m=complement(limits, candidate + occupied),
                    occupied_candidate_intervals_m=merge(occupied),
                    candidate_evidence_supported=bool(candidate or occupied),
                    raw_point_ids=sorted(set(all_ids)),
                    low_nonblocking_point_ids=sorted(set(low_ids)),
                )
        sections.append(row)
    payload = dict(
        schema_version=VERSION,
        sample_id=meta["frame_id"],
        scope="partial",
        upstream_contract_version=P2_VERSION,
        frame_kind="common_straight_strip_st",
        station_spacing_m=resolution,
        evidence_mode="surface_candidate_review",
        sections=sections,
    )
    result = measure(payload)
    has_unknown = any((r["unknown_intervals_m"] for r in sections))
    if result["minimum_passable_ratio"] == 0 and has_unknown:
        for key in (
            "minimum_passable_ratio",
            "raw_minimum_passable_ratio",
            "raw_minimum_width_m_on_selected_path",
            "robust_width_m_on_selected_path",
        ):
            result[key] = None
        result.update(
            result_authority="none",
            topology_status="unknown",
            measurement_coverage=0.0,
            selected_corridor_path=[],
            bottleneck=None,
        )
        result["abstention_reasons"].append("candidate_disconnected_with_unknown_space")
    if result["minimum_passable_ratio"] is None:
        result["requested_measurement_scope"] = result["measurement_scope"]
        result["measurement_scope"] = "none"
        if not all(models.values()):
            result["abstention_reasons"].append(
                "insufficient_bilateral_boundary_anchors"
            )
        if has_unknown:
            result["abstention_reasons"].append(
                "unknown_space_prevents_complete_candidate_measurement"
            )
    result.update(
        adapter_version=ADAPTER_VERSION,
        input_sha256=meta["input_sha256"],
        spatial_evidence=dict(
            boundary_models=models,
            sections=sections,
            strip_spec=deepcopy(spec),
            source_metadata=deepcopy(meta),
            width_policy=deepcopy(record["width_policy"]),
            source_free_space_inference=record["free_space_inference"],
            unknown_space_present=has_unknown,
            numerator_semantics="supported_surface_candidate_minus_occupied_cells",
            footprint_assumption="occupied_return_blocks_its_grid_cell_for_review_only",
            verified_free_volume_required_for_review=False,
        ),
        physical_filter_support_m=dict(
            station_spacing_m=resolution,
            first_to_last_sample_span_m=2 * resolution,
            nominal_three_strip_extent_m=3 * resolution,
        ),
        safety_verdict=None,
    )
    return result
