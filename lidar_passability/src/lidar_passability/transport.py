"""Draft P4 successor envelope; explicit semantics, nulls and demo provenance."""

from copy import deepcopy
import hashlib
import json
import math
from .bottleneck import VERSION, POLICY


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def bridge(result, binding, *, data_kind, metadata, versions):
    canonical(result)
    if result.get("schema_version") != VERSION or result.get("policy") != POLICY:
        raise ValueError("unsupported bottleneck semantics")
    if result["frame_id"] != binding["frame_id"]:
        raise ValueError("frame mismatch")
    sha = binding["sha256"]
    if len(sha) != 64 or any((c not in "0123456789abcdef" for c in sha)):
        raise ValueError("invalid input hash")
    if result.get("input_sha256", sha) != sha:
        raise ValueError("input hash mismatch")
    if result.get("spatial_evidence"):
        source = result["spatial_evidence"]["source_metadata"]
        if (source["frame_id"], source["input_sha256"]) != (binding["frame_id"], sha):
            raise ValueError("spatial source mismatch")
    if data_kind not in (
        "synthetic_contract_test",
        "historical_actual_inference",
        "actual_inference",
    ):
        raise ValueError("unregistered input provenance")
    ratio, width, full = (
        result[k]
        for k in (
            "minimum_passable_ratio",
            "bottleneck_width_m",
            "same_section_full_width_m",
        )
    )
    if ratio is None:
        if (
            width is not None
            or full is not None
            or (not result["reasons"])
            or (result["authority"] != "none")
        ):
            raise ValueError("invalid null result")
    else:
        if result["authority"] != "provisional_review":
            raise ValueError("authority mismatch")
        if (
            any(
                (
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or (not math.isfinite(x))
                    for x in (ratio, width)
                )
            )
            or not 0 <= ratio <= 1
            or width < 0
        ):
            raise ValueError("invalid numeric result")
        if full is None:
            if not (
                ratio == width == 0
                and result["bottleneck"]["kind"]
                == "known_disconnection_in_sampled_strip"
            ):
                raise ValueError("missing same-section denominator")
        elif (
            isinstance(full, bool)
            or not isinstance(full, (int, float))
            or (not math.isfinite(full))
            or (full <= 0)
            or (not math.isclose(width / full, ratio, abs_tol=1e-10))
        ):
            raise ValueError("same-section ratio mismatch")
        if width > 0:
            path = result["selected_corridor_path"]
            narrow = min(
                path, key=lambda r: (r["width_m"], r["ratio"], r["section_index"])
            )
            if (width, full, ratio) != (
                narrow["width_m"],
                narrow["full_width_m"],
                narrow["ratio"],
            ):
                raise ValueError("bottleneck path mismatch")
    if (
        result["production_eligible"]
        or result["safety_verdict"] is not None
        or result["smoothing_applied"]
    ):
        raise ValueError("unsupported production claim or smoothing")
    if metadata.get("site_time_kind") != "fictional_demo_not_acquisition":
        raise ValueError("explicit demo provenance required")
    if any(
        (
            metadata.get(k) is not None
            for k in ("actual_capture_time", "actual_location")
        )
    ):
        raise ValueError("unknown actual metadata must remain null")
    key_input = dict(
        input=binding,
        versions=versions,
        policy=POLICY,
        geometric_parameters=result.get("geometric_parameters"),
        spatial_evidence=result.get("spatial_evidence"),
    )
    out = dict(
        schema_version="roadpush.lidar.integration/0.2-draft",
        data_kind=data_kind,
        input=deepcopy(binding),
        versions=deepcopy(versions),
        display=dict(
            bottleneck_ratio=ratio,
            bottleneck_m_est=width,
            denominator_at_bottleneck_m=full,
            ratio_basis="same_section_as_minimum_width",
            measurement_scope=result["scope"],
            authority=result["authority"],
        ),
        metadata=deepcopy(metadata),
        passability="undetermined",
        auto_aggregation_allowed=False,
        raw_result=deepcopy(result),
        computation_key=hashlib.sha256(canonical(key_input)).hexdigest(),
    )
    canonical(out)
    return out
