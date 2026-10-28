"""Saved prediction -> independent PCD/ROI validation -> unchanged development geometry."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    AnalysisRequest,
    AnalysisResponse,
    ContractError,
    envelope,
    sha256,
)
from .pcd import read_all_raw
from .input_frame import prepare_frame
from .prediction_contract import adapt
from .p1_input_adapter import prepare_input
from .spatial_grounded import Config
from .geometry_first import build_grounded
from .boundary_recovery import recover
from .curb_contacts import contacts
from .measure_v5 import fit_floor, evaluate


def development_config() -> Config:
    return Config(
        bounds_xy=(-4.0, 4.0, -3.0, 3.0),
        resolution_m=0.1,
        support_radius_m=0.4,
        min_support=8,
        score_threshold=0.7,
        plane_residual_m=0.03,
        max_slope_deg=25,
        max_rms_m=0.025,
        min_inlier_fraction=0.65,
        obstacle_height_min_m=0.18,
        obstacle_height_max_m=2.1,
        component_step_m=0.1,
        ransac_trials=24,
    )


class LidarAnalyzer:
    def analyze_pcd(
        self,
        *,
        runtime,
        request_id,
        frame_id,
        pcd_path,
        expected_pcd_sha256,
        export_dir,
        model_version,
        axis_evidence,
        execute_gpu=False,
        max_seconds=120,
        boundary_profile: str = "v5_review",
        include_unknown_bounds: bool = False,
    ) -> AnalysisResponse:
        """Explicit live inference + analysis; export_dir preserves inference artifacts.

        Requires an independently supplied FrozenRuntime and explicit opt-in.
        Does not create environments, install packages, rent or restart GPUs.
        """
        if execute_gpu is not True:
            raise ContractError(
                "Live inference requires execute_gpu=True; use runtime.plan() for preflight"
            )
        # Validate non-GPU arguments before any paid computation.
        provisional = AnalysisRequest(
            request_id=request_id,
            frame_id=frame_id,
            pcd_path=Path(pcd_path),
            prediction_path=Path(export_dir) / "prediction.npz",
            report_path=Path(export_dir) / "result.json",
            expected_pcd_sha256=expected_pcd_sha256,
            expected_prediction_sha256="0" * 64,
            expected_report_sha256="0" * 64,
            expected_checkpoint_sha256=runtime.expected_checkpoint_sha256,
            model_version=model_version,
            source_receipt_sha256="0" * 64,
            axis_evidence=axis_evidence,
            boundary_profile=boundary_profile,
            include_unknown_bounds=include_unknown_bounds,
            data_kind="actual_inference",
        )
        provisional.validate()
        # Reject unsupported PCDs before launching a paid GPU prediction. The
        # downstream raw reader has a tighter contract than the GPU predictor.
        if sha256(pcd_path) != expected_pcd_sha256:
            raise ContractError("PCD hash mismatch")
        read_all_raw(pcd_path)
        prepare_frame(Path(pcd_path).read_bytes())
        exported = runtime.export(
            pcd_path,
            export_dir,
            expected_pcd_sha256=expected_pcd_sha256,
            max_seconds=max_seconds,
            execute=True,
        )
        from dataclasses import replace

        request = replace(
            provisional,
            prediction_path=exported["prediction_path"],
            report_path=exported["report_path"],
            expected_prediction_sha256=sha256(exported["prediction_path"]),
            expected_report_sha256=sha256(exported["report_path"]),
            source_receipt_sha256=exported["source_receipt_sha256"],
        )
        return self.analyze_prediction(request)

    def analyze_prediction(self, request: AnalysisRequest) -> AnalysisResponse:
        """Pure library call: no directories/files written and no GPU/model imported."""
        request.validate()
        for path, expected in (
            (request.pcd_path, request.expected_pcd_sha256),
            (request.prediction_path, request.expected_prediction_sha256),
            (request.report_path, request.expected_report_sha256),
        ):
            if sha256(path) != expected:
                raise ContractError("Input content SHA mismatch")
        report = json.loads(Path(request.report_path).read_text(encoding="utf8"))
        raw = read_all_raw(request.pcd_path)
        frame = prepare_frame(Path(request.pcd_path).read_bytes())
        with np.load(request.prediction_path, allow_pickle=False) as arrays:
            table, meta = adapt(
                arrays,
                report,
                expected_pcd_sha256=request.expected_pcd_sha256,
                expected_checkpoint_sha256=request.expected_checkpoint_sha256,
                expected_original_count=frame["original_pcd_count"],
                expected_raw_ids=frame["raw_pcd_index"],
                expected_roi_coord=frame["coord"],
                expected_coverage_parts=int(frame["count"].max()),
            )
        prepared = prepare_input(
            raw,
            table,
            meta,
            axis_evidence=request.axis_evidence,
            frame_id=request.frame_id,
        )
        labels = prepared["predicted_class"]
        # Original v3/v5 runners reduce the exported float32 probabilities.
        # P1's generic join uses float64 storage; preserve the export dtype here
        # so mean support scores remain bit-for-bit identical to those results.
        scores = np.full(len(raw), np.nan, dtype=table["scores"].dtype)
        scores[table["point_id"]] = table["scores"].max(1)
        labels_before = labels.copy()
        scores_before = scores.copy()
        raw_before = raw.copy()
        base = dict(config=asdict(development_config()), metadata=prepared["metadata"])
        try:
            spatial, ground_evidence = build_grounded(base, raw, labels, scores)
        except ValueError as exc:
            if not any(
                t in str(exc)
                for t in (
                    "Insufficient scored",
                    "No sufficiently supported",
                    "Direction ambiguous",
                )
            ):
                raise
            return envelope(
                request,
                spatial=None,
                recovery=None,
                conditional=dict(state="geometry_direction_hold", reason=str(exc)),
                diagnostics={},
            )
        tr = spatial["metadata"]["raw_to_processing_xy_transform"]
        # v5's recorded diagnostic runner intentionally computes this stage in
        # float64; preserve its dtype instead of inheriting the raw float32.
        p = raw.astype(float).copy()
        p[:, :2] = (raw[:, :2] - tr["origin_xy_m"]) @ np.array(
            tr["source_xy_to_strip_matrix"]
        ).T
        recovery = recover(spatial, raw, labels, scores)
        conditional = self._conditional(
            spatial, recovery, p, labels, scores, request.boundary_profile
        )
        diagnostics = dict(
            ground_admission=ground_evidence,
            raw_rows=len(raw),
            reference_labels_used=False,
            final_model_accepted=False,
            thresholds_operationally_adopted=False,
        )
        if request.include_unknown_bounds and conditional.get("evaluated_hypotheses"):
            from .interval_certificate import all_domains

            chosen = conditional["evaluated_hypotheses"][0]
            diagnostics["unknown_bounds"] = all_domains(chosen["sections"])
        if (
            not np.array_equal(raw, raw_before, equal_nan=True)
            or not np.array_equal(labels, labels_before)
            or not np.array_equal(scores, scores_before, equal_nan=True)
        ):
            raise RuntimeError("Source arrays changed")
        diagnostics["source_arrays_unchanged"] = True
        return envelope(
            request,
            spatial=spatial,
            recovery=recovery,
            conditional=conditional,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _conditional(spatial, recovery, p, labels, scores, profile):
        prev = deepcopy(recovery)
        try:
            floor, _ = fit_floor(spatial)
            contact = contacts(spatial, p, labels, scores, floor)
        except ValueError as exc:
            contact = dict(
                models={"left": None, "right": None},
                anchors={"left": [], "right": []},
                diagnostic=dict(hold=str(exc)),
            )
        missing = not any(prev["models"].values())
        if missing:
            prev["models"] = contact["models"]
        if profile == "low_barrier_review":
            from .barriers import evaluate_barrier

            result = evaluate_barrier(spatial, prev, p, labels, scores)
        else:
            result = evaluate(spatial, prev, p, labels, scores)
        result.update(
            frame_id=spatial["metadata"]["frame_id"],
            curb_contact_recovery=contact,
            previous_curb_models_present=not missing,
        )
        return result
