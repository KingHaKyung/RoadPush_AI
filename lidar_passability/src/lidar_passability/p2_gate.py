"""P2 v0.2 validation + diagnostic null envelope, not a ban on review estimates."""

import math

VERSION = "p2-spatial-draft-0.2"


def validate_p2(record):
    if record.get("schema_version") != VERSION:
        raise ValueError("P2 version changed: review contract before consuming")
    policy = record.get("width_policy", {})
    if (
        policy.get("low_objects") != "excluded_from_width_blocking_by_user_assumption"
        or isinstance(policy.get("max_height_m"), bool)
        or (not isinstance(policy.get("max_height_m"), (int, float)))
        or (not math.isclose(policy["max_height_m"], 0.18, abs_tol=1e-09))
        or (policy.get("decision_date") != "2026-10-26")
        or (policy.get("safety_validated") is not False)
    ):
        raise ValueError("P2 low-object policy changed or missing: review required")
    cells = record.get("cells")
    if not isinstance(cells, list):
        raise ValueError("P2 cells required")
    indexes, raw_ids = (set(), set())
    for cell in cells:
        index = tuple(cell["index"])
        if (
            len(index) != 2
            or any(
                (isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in index)
            )
            or index in indexes
        ):
            raise ValueError("invalid or duplicate P2 cell")
        indexes.add(index)
        ids = cell["raw_point_ids"]
        if (
            any((isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids))
            or len(set(ids)) != len(ids)
            or set(ids) & raw_ids
        ):
            raise ValueError("P2 raw IDs must be preserved and uniquely assigned")
        raw_ids.update(ids)
        for key in (
            "occupancy_candidate_point_ids",
            "low_nonblocking_point_ids",
            "height_unresolved_point_ids",
            "out_of_height_band_point_ids",
        ):
            values = cell.get(key)
            if not isinstance(values, list) or not set(values) <= set(ids):
                raise ValueError("P2 evidence IDs not preserved: " + key)
        low = set(cell["low_nonblocking_point_ids"])
        if low & set(cell["occupancy_candidate_point_ids"]):
            raise ValueError("low object incorrectly blocks width")
        if not low <= set(cell["out_of_height_band_point_ids"]):
            raise ValueError("P2 low-object source evidence lost")
    return record


def inspect_p2(record):
    validate_p2(record)
    reasons = []
    if record.get("full_sidewalk_boundary_confirmed") is not True:
        reasons.append("bilateral_full_boundary_unconfirmed")
    if record.get("total_width_m") is None:
        reasons.append("full_width_unobserved")
    if record.get("free_space_inference") != "validated_volume_free_space":
        reasons.append("free_volume_unverified")
    reasons.append("strip_direction_and_boundary_hypothesis_not_supplied")
    return dict(
        schema_version="p3_p2_gate_draft_v1",
        upstream_contract_version=VERSION,
        minimum_passable_ratio=None,
        raw_minimum_passable_ratio=None,
        absolute_passable_width_m=None,
        measurement_scope="none",
        result_authority="none",
        production_eligible=False,
        measurement_mode="diagnostic_gate_only",
        review_estimates_require_verified_free_volume=False,
        abstention_reasons=reasons,
    )
