"""Draft P1 adapter. CPU/Numpy only; never loads a checkpoint or runs inference.

Validation needs independently reconstructed ROI IDs/coordinates from the exact
PCD and reader version. Self-consistent arrays alone cannot prove point identity.
"""

from pathlib import Path
import hashlib
import numpy as np

VERSION = "roadpush.point_prediction/0.1-draft"
CLASS_IDS = np.array([1, 2, 3, 8, 10, 12, 40, 48], dtype=np.int16)
CLASS_NAMES = [
    "pedestrian",
    "two_wheeler",
    "car",
    "truck_bus",
    "traffic_light",
    "traffic_sign",
    "road",
    "sidewalk",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def valid_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((c in "0123456789abcdef" for c in value))
    )


def validate(
    arrays,
    report,
    *,
    expected_pcd_sha256,
    expected_checkpoint_sha256,
    expected_original_count,
    expected_raw_ids,
    expected_roi_coord,
    expected_coverage_parts,
):
    """Return summary or fail closed. Expectations must come from trusted input.

    Does not establish model quality, class calibration, spatial free space,
    signed provenance, checkpoint acceptance, or real-world metric calibration.
    """
    for key in (
        "probabilities",
        "predicted_class_id",
        "observed_roi_mask",
        "raw_pcd_index",
        "roi_coord",
        "class_ids",
    ):
        require(key in arrays, "Missing array: " + key)
    p, pred, mask, ids, xyz, classes = [
        np.asarray(arrays[k])
        for k in (
            "probabilities",
            "predicted_class_id",
            "observed_roi_mask",
            "raw_pcd_index",
            "roi_coord",
            "class_ids",
        )
    ]
    n = expected_original_count
    require(type(n) is int and 0 < n <= 5000000, "Invalid original count")
    expected_ids = np.asarray(expected_raw_ids)
    expected_xyz = np.asarray(expected_roi_coord)
    require(
        expected_ids.ndim == 1 and expected_ids.dtype.kind in "iu",
        "Invalid trusted point IDs",
    )
    require(
        len(expected_ids) > 0 and len(np.unique(expected_ids)) == len(expected_ids),
        "Invalid trusted ID uniqueness",
    )
    require(
        np.all((expected_ids >= 0) & (expected_ids < n)),
        "Trusted IDs outside original PCD",
    )
    require(
        expected_xyz.shape == (len(expected_ids), 3)
        and np.isfinite(expected_xyz).all(),
        "Invalid trusted coordinates",
    )
    require(p.dtype.kind == "f" and p.shape == (n, 8), "Probability shape/dtype")
    require(pred.dtype.kind in "iu" and pred.shape == (n,), "Class shape/dtype")
    require(mask.dtype == np.dtype(bool) and mask.shape == (n,), "Mask must be bool[N]")
    require(
        ids.dtype.kind in "iu" and ids.shape == expected_ids.shape,
        "Point ID shape/dtype",
    )
    require(len(np.unique(ids)) == len(ids), "Duplicate raw point IDs")
    require(np.array_equal(ids, expected_ids), "Point ID order or identity mismatch")
    expected_mask = np.zeros(n, dtype=bool)
    expected_mask[expected_ids] = True
    require(np.array_equal(mask, expected_mask), "ROI mask does not match trusted IDs")
    require(
        xyz.dtype.kind == "f" and xyz.shape == expected_xyz.shape,
        "Coordinate shape/dtype",
    )
    require(
        np.isfinite(xyz).all() and np.array_equal(xyz, expected_xyz),
        "Coordinate to point ID mismatch",
    )
    require(
        classes.dtype.kind in "iu" and np.array_equal(classes, CLASS_IDS),
        "Class column order mismatch",
    )
    require(np.isfinite(p[mask]).all(), "Nonfinite ROI scores")
    require(np.all((p[mask] >= 0) & (p[mask] <= 1)), "Scores outside [0,1]")
    require(
        np.allclose(p[mask].sum(1), 1, rtol=0, atol=1e-05), "Unnormalized ROI scores"
    )
    require(np.isnan(p[~mask]).all(), "Outside ROI scores must be NaN")
    require(np.all(pred[~mask] == -1), "Outside ROI class must be -1")
    require(
        np.array_equal(pred[mask], CLASS_IDS[p[mask].argmax(1)]),
        "Class IDs disagree with score argmax",
    )
    require(report.get("status") == "inference_export_complete", "Incomplete export")
    require(
        report.get("model_id") == "litept-s-aihub8-official-v2", "Unexpected model ID"
    )
    for key, expected in [
        ("pcd_sha256", expected_pcd_sha256),
        ("checkpoint_sha256", expected_checkpoint_sha256),
    ]:
        require(
            valid_hash(expected) and report.get(key) == expected,
            "Provenance mismatch: " + key,
        )
    require(report.get("coverage") == "rotating5cm", "Unsupported coverage protocol")
    require(report.get("official_2_5cm_TTA") is False, "Incorrect official TTA claim")
    require(report.get("pointrope") in ("cuda", "pytorch"), "Missing PointROPE backend")
    require(
        report.get("class_ids") == CLASS_IDS.tolist()
        and report.get("class_names") == CLASS_NAMES,
        "Report class mapping mismatch",
    )
    require(
        type(report.get("roi_points")) is int and report["roi_points"] == len(ids),
        "Report ROI count mismatch",
    )
    require(
        type(report.get("original_points")) is int and report["original_points"] == n,
        "Report original count mismatch",
    )
    require(
        type(report.get("coverage_parts")) is int and report["coverage_parts"] > 0,
        "Missing coverage parts",
    )
    require(
        type(expected_coverage_parts) is int
        and expected_coverage_parts > 0
        and (report["coverage_parts"] == expected_coverage_parts),
        "Coverage part count differs from input grid",
    )
    require(report.get("width_computed") is False, "Unexpected width claim")
    return dict(
        schema_version=VERSION,
        original_points=n,
        predicted_points=len(ids),
        unpredicted_points=n - len(ids),
        roi_point_coverage=1.0,
        scope="export structural validation only; not spatial visibility or model quality",
    )


def adapt(arrays, report, **expected):
    """Produce a compact ROI table, preserving mask and identity; no class merge."""
    summary = validate(arrays, report, **expected)
    ids = np.asarray(arrays["raw_pcd_index"])
    table = dict(
        point_id=ids.copy(),
        xyz=np.asarray(arrays["roi_coord"]).copy(),
        class_id=np.asarray(arrays["predicted_class_id"])[ids].copy(),
        scores=np.asarray(arrays["probabilities"])[ids].copy(),
        original_prediction_mask=np.asarray(arrays["observed_roi_mask"]).copy(),
        class_ids=CLASS_IDS.copy(),
    )
    metadata = dict(
        **summary,
        pcd_sha256=report["pcd_sha256"],
        checkpoint_sha256=report["checkpoint_sha256"],
        coverage_recipe="rotating5cm",
        score_semantics="mean softmax across votes; uncalibrated",
        point_id_semantics="zero-based row in exact original PCD payload",
        coord_frame="input_pcd_local; axis orientation not verified",
        unit="m",
        spatial_visibility="not_established",
        semantic_open_set_unknown="not_established",
        final_model_accepted=False,
    )
    return (table, metadata)
