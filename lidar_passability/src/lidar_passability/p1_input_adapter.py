"""Join P1 validated ROI table to independently read raw geometry; no inference.

Call P1 prediction_contract.validate/adapt first. This guard does not replace it.
P1 currently says axis orientation not verified: explicit external evidence needed.
"""

import numpy as np

P1_VERSION = "roadpush.point_prediction/0.1-draft"
P1_CLASS_IDS = np.array([1, 2, 3, 8, 10, 12, 40, 48])


def prepare_input(raw_xyz, table, p1_metadata, *, axis_evidence, frame_id):
    raw = np.asarray(raw_xyz)
    ids = np.asarray(table["point_id"])
    xyz = np.asarray(table["xyz"])
    classes = np.asarray(table["class_id"])
    probs = np.asarray(table["scores"])
    order = np.asarray(table["class_ids"])
    if p1_metadata.get("schema_version") != P1_VERSION:
        raise ValueError("P1 version mismatch; re-review upstream contract")
    if (
        not isinstance(axis_evidence, dict)
        or axis_evidence.get("metric_z_up_verified") is not True
        or (not axis_evidence.get("source"))
    ):
        raise ValueError("P1 does not verify axes; external evidence required")
    if p1_metadata.get("unit") != "m":
        raise ValueError("metric input required")
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind != "f":
        raise ValueError("independently read float raw XYZ required")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid P1 raw IDs")
    if (
        (ids < 0).any()
        or (ids >= len(raw)).any()
        or (not np.array_equal(raw[ids], xyz))
    ):
        raise ValueError("raw identity/coordinates mismatch")
    if not np.array_equal(order, P1_CLASS_IDS):
        raise ValueError("class columns mismatch; expected original dataset IDs")
    if (
        classes.shape != (len(ids),)
        or probs.shape != (len(ids), 8)
        or (not np.isfinite(probs).all())
        or (probs < 0).any()
        or (probs > 1).any()
        or (not np.allclose(probs.sum(1), 1, atol=1e-05, rtol=0))
        or (not np.array_equal(classes, order[probs.argmax(1)]))
    ):
        raise ValueError("class/score mismatch")
    mask = np.asarray(table["original_prediction_mask"])
    expected_mask = np.zeros(len(raw), dtype=bool)
    expected_mask[ids] = True
    if mask.dtype != np.dtype(bool) or not np.array_equal(mask, expected_mask):
        raise ValueError("P1 ROI mask mismatch")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("explicit frame ID required")
    sha = p1_metadata.get("pcd_sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any((c not in "0123456789abcdef" for c in sha))
    ):
        raise ValueError("PCD hash required")
    labels = np.full(len(raw), -1, dtype=int)
    score = np.full(len(raw), np.nan)
    labels[ids] = classes
    score[ids] = probs.max(1)
    metadata = dict(
        input_sha256=sha,
        frame_id=frame_id,
        coordinate_frame=p1_metadata["coord_frame"],
        units="m",
        z_up_verified=True,
        axis_evidence=dict(axis_evidence),
        prediction_provenance=dict(p1_metadata),
        raw_geometry_scope="all independently read raw rows; P2 ROI explicit in config",
    )
    return dict(
        xyz=raw.copy(),
        original_point_id=np.arange(len(raw)),
        predicted_class=labels,
        score=score,
        class_ids={"road": 40, "sidewalk": 48},
        metadata=metadata,
    )
