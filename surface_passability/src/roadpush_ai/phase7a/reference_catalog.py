from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from roadpush_ai.phase7a.retry_v1 import (
    TrapezoidBoundaryRetryV1Policy,
    fit_trapezoid_boundary_retry_v1,
)


ROOT = get_settings().project_root
OUTPUT_ROOT = ROOT / "outputs" / "surface_generalization"
DEFAULT_OUTPUT = OUTPUT_ROOT / "phase7_trapezoid_boundary_retry_v1_targeted"
DATASETS = (
    ("canonical40", OUTPUT_ROOT / "phase6_production_selector_canonical40_v1"),
    ("alley_core40", OUTPUT_ROOT / "phase6_production_selector_alley_core40_v1"),
    ("expansion_challenge80", OUTPUT_ROOT / "phase6_production_selector_expansion_challenge80_v1"),
)
TARGETED_ORDER = (
    "commons_01",
    "commons_04",
    "IMG_4282",
    "MP_SEL_PN000601",
    "kr_independent_01",
    "IMG_4548",
    "MP_SEL_PN000646",
    "MP_SEL_PN000698",
    "alley_001",
    "alley_009",
    "alley_033",
    "alley_038",
)
AUDIT_MANIFEST = ROOT / "outputs" / "walkway_detection_ablation_v2" / "model_audit_40" / "audit_manifest.json"
ALLEY_MANIFEST = ROOT / "outputs" / "data_selection" / "alley_road_selection_v1" / "final_alley_test_set_v1" / "manifest.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value > 0


def _branch_paths(root: Path, sample_id: str, surface: str) -> tuple[Path, Path, Path]:
    base = root / "artifacts" / sample_id / surface
    if surface == "sidewalk":
        return (
            base / "07_proposed_pruned_boundary.png",
            base / "08_applied_blocker.png",
            base / "09_final_free.png",
        )
    return (
        base / "08_structural_road.png",
        base / "09_road_blocker.png",
        base / "10_free_road.png",
    )


def _source_images() -> dict[str, str]:
    audit = _read_json(AUDIT_MANIFEST)
    rows = audit["samples"] if isinstance(audit, dict) else audit
    mapping = {row["sample_id"]: str((ROOT / row["relative_path"]).resolve()) for row in rows}
    for row in _read_json(ALLEY_MANIFEST):
        mapping[row["sample_id"]] = str((ALLEY_MANIFEST.parent / "images" / row["output_filename"]).resolve())
    return mapping


def _save_masks(destination: Path, masks: dict[str, np.ndarray]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name, mask in masks.items():
        ok = cv2.imwrite(str(destination / f"{name}.png"), np.asarray(mask, dtype=np.uint8) * 255)
        if not ok:
            raise OSError(f"Could not write {destination / f'{name}.png'}")


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    return {
        label: float(np.quantile(data, quantile))
        for label, quantile in (("min", 0.0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("max", 1.0))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 trapezoid-boundary retry_v1 targeted offline probe")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite retry_v1 output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    source_images = _source_images()
    phase6_rows: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    for dataset, dataset_root in DATASETS:
        for row in _read_json(dataset_root / "results.json"):
            sample_id = row["sample_id"]
            if sample_id in TARGETED_ORDER:
                if sample_id in phase6_rows:
                    raise RuntimeError(f"Duplicate targeted sample: {sample_id}")
                phase6_rows[sample_id] = (dataset, dataset_root, row)
    missing = [sample_id for sample_id in TARGETED_ORDER if sample_id not in phase6_rows]
    if missing:
        raise RuntimeError(f"Missing targeted samples: {missing}")

    policy = TrapezoidBoundaryRetryV1Policy()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sample_id in TARGETED_ORDER:
        dataset, phase6_root, phase6 = phase6_rows[sample_id]
        selection = phase6["surface_selection"]
        selected = selection.get("selected_surface")
        saved: dict[str, Any] | None = None
        if selected is None:
            result, masks = fit_trapezoid_boundary_retry_v1(None, selection["state"], None, None, policy=policy)
        else:
            structural_path, blocker_path, free_path = _branch_paths(phase6_root, sample_id, selected)
            structural = _read_mask(structural_path)
            blocker = _read_mask(blocker_path)
            result, masks = fit_trapezoid_boundary_retry_v1(
                selected,
                selection["state"],
                structural,
                blocker,
                row_reliability=phase6[selected].get("rows"),
                policy=policy,
            )
            artifact_dir = output / "artifacts" / sample_id
            _save_masks(artifact_dir, masks)
            saved = {
                "phase6_structural_path": str(structural_path.resolve()),
                "phase6_blocker_path": str(blocker_path.resolve()),
                "phase6_free_path": str(free_path.resolve()),
                "phase6_structural_file_sha256": _file_sha256(structural_path),
                "phase6_blocker_file_sha256": _file_sha256(blocker_path),
                "phase6_free_file_sha256": _file_sha256(free_path),
                "retry_artifact_dir": str(artifact_dir.resolve()),
            }
        warnings = result["blocker_reuse"]["warnings"]
        if selected is None:
            disposition = "uncertain"
        elif result["envelope_fit"]["status"] != "accepted":
            disposition = "rejected"
        elif warnings:
            disposition = "boundary_valid_but_blocker_suspect"
        else:
            disposition = "accepted"
        result.update(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "input_profile": selection.get("input_profile"),
                "selector_state": selection.get("state"),
                "selector_reasons": selection.get("reasons", []),
                "source_image": source_images[sample_id],
                "phase6_output_root": str(phase6_root.resolve()),
                "saved_phase6_artifacts": saved,
                "final_disposition": disposition,
                "normal_control": sample_id == "IMG_4282",
            }
        )
        results.append(result)
    measurement_runtime = time.perf_counter() - started

    accepted_ratios = [
        row["minimum_passable_ratio"]
        for row in results
        if row["minimum_passable_ratio"] is not None
    ]
    summary = {
        "status": "AUTOMATIC CHECKS PENDING / USER REVIEW REQUIRED",
        "phase7_status": "IN PROGRESS",
        "phase7_complete": False,
        "thresholds_frozen": False,
        "full_160_authorized": False,
        "production_integration_authorized": False,
        "policy": policy.__dict__,
        "sample_count": len(results),
        "targeted_manifest": list(TARGETED_ORDER),
        "selection_counts": dict(Counter(row["selected_surface"] or "uncertain" for row in results)),
        "fit_counts": dict(Counter(row["envelope_fit"]["status"] for row in results)),
        "disposition_counts": dict(Counter(row["final_disposition"] for row in results)),
        "rejection_reason_counts": dict(
            Counter(reason for row in results for reason in row["envelope_fit"].get("rejection_reasons", []))
        ),
        "blocker_coverage_suspect_samples": [
            row["sample_id"] for row in results if "blocker_coverage_suspect" in row["blocker_reuse"]["warnings"]
        ],
        "representative_ratio_samples": [row["sample_id"] for row in results if row["minimum_passable_ratio"] is not None],
        "minimum_passable_ratio_distribution": _quantiles(accepted_ratios),
        "measurement_runtime_seconds": measurement_runtime,
        "render_runtime_seconds": None,
        "semantic_inference_count": 0,
        "context_build_count": 0,
        "geometry_extraction_count": 0,
        "selector_call_count": 0,
        "blocker_hash_mismatch_count": sum(
            not row["blocker_reuse"].get("hash_match", True) for row in results if row["selected_surface"] is not None
        ),
        "historical_phase7_decisions_reused": False,
        "notes": [
            "This run reads saved Phase 6 results and masks only.",
            "All thresholds are retry_v1 review candidates and are not frozen.",
            "Existing Phase 7 outputs and decisions were not used as measurement inputs.",
            "The targeted run requires user review before any full-160 execution.",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

