# Phase Review and History Source Map

Purpose: identify the frozen, already-computed review evidence that may be
copied into `RoadPush_AI` without running any model or phase computation.

All paths are relative to the original `RoadPush` repository root. A path is
canonical here only for review/history preservation; production runtime
authority remains defined by the phase reports and result contracts.

## Canonical sources

| Phase | Disposition | Production / regression runner | Review renderer or review writer | Frozen canonical review HTML | Final report(s) | Representative image |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Passed; behavior-preserving `SurfaceContext` regression | `scripts/walkway_v3/run_phase1_sidewalk_context_regression.py` | `scripts/walkway_v3/run_obstacle_connected_component_probe.py` (called by the Phase 1 wrapper and writes the review) | `outputs/surface_generalization/phase1_sidewalk_context_regression_40_v1/review.html` | `surface_generalization/PHASE1_REGRESSION_REPORT.md` | `outputs/surface_generalization/phase1_sidewalk_context_regression_40_v1/artifacts/commons_01/summary.png` |
| 2 | Passed; obstacle geometry separated from surface ownership | `scripts/walkway_v3/run_surface_pipeline.py` | `scripts/walkway_v3/run_surface_pipeline.py` | `outputs/surface_generalization/phase2_obstacle_geometry_ownership_regression_40_v3/review.html` | `surface_generalization/PHASE2_REGRESSION_REPORT.md`; `surface_generalization/PHASE2_AUDIT_REMEDIATION.md` | `outputs/surface_generalization/phase2_obstacle_geometry_ownership_regression_40_v3/artifacts/commons_01/summary.png` |
| 3 | Enhanced Road V2 accepted; original Phase 3 retained for rollback | `scripts/walkway_v3/run_phase3_road_context_v2.py` | `scripts/walkway_v3/run_phase3_road_context_v2.py` | `outputs/surface_generalization/phase3_road_context_enhanced_40_v1/review.html`; alley-core companion: `outputs/surface_generalization/phase3_road_context_enhanced_alley_core40_v1/review.html` | `surface_generalization/PHASE3_FINAL_ACCEPTANCE_REPORT.md`; `surface_generalization/PHASE3_ENHANCED_ROAD_V2_PRODUCTION_REPORT.md` | `outputs/surface_generalization/phase3_road_context_enhanced_40_v1/artifacts/commons_01/road/road_summary.png` |
| 4 | No independent final artifact. Its structural/component work was absorbed into the accepted Phase 3/5 path. | No standalone canonical runner | No standalone canonical renderer | No standalone canonical HTML | Historical roadmap/report context only; do not invent a Phase 4 review | Use Phase 3 and Phase 5 representatives |
| 5 | Accepted and closed; sparse row-span support integrated | `scripts/walkway_v3/run_phase5_sparse_row_span_support_surface_probe.py` and accepted integration validation in `scripts/walkway_v3/validate_phase5_sparse_support_integration.py` | `scripts/walkway_v3/validate_phase5_targeted.py` writes the targeted index; the targeted runner writes the full review | `outputs/surface_generalization/phase5_targeted_expansion_challenge_80_v1/targeted_review.html`; full companion: `outputs/surface_generalization/phase5_targeted_expansion_challenge_80_v1/review.html` | `surface_generalization/PHASE5_FINAL_ACCEPTANCE_REPORT.md`; `surface_generalization/PHASE5_SURFACE_NEUTRAL_SPARSE_ROW_SPAN_SUPPORT_REPORT.md`; `outputs/surface_generalization/phase5_sparse_support_integration_regression_v5/REGRESSION.md` | `outputs/surface_generalization/phase5_targeted_expansion_challenge_80_v1/artifacts/alley_060/road/road_summary.png` |
| 6 | Complete; fixed input-profile selector integrated without changing either surface branch | `scripts/walkway_v3/run_phase6_surface_pipeline.py` | `scripts/walkway_v3/run_phase6_auto_selector_probe.py` produced the frozen selector review; `scripts/walkway_v3/validate_phase6_input_profiles.py` validates it | `outputs/surface_generalization/phase6_image_upload_selector_160_v2/review.html` | `surface_generalization/PHASE6_PRODUCTION_INTEGRATION_REPORT.md`; `surface_generalization/PHASE6_INPUT_PROFILE_SELECTOR_REPORT.md` | `outputs/surface_generalization/phase6_image_upload_selector_160_v2/cards/alley_060.png` |
| 7-A | Approved and closed for MVP; S2 deferred | `scripts/walkway_v3/run_phase7_extension_track_s1_targeted.py` using the frozen V6 reconstruction path | The same runner writes the result review; `scripts/walkway_v3/build_phase7_extension_track_s1_full160_feedback_review.py` is retained as the final user-feedback review builder | `outputs/surface_generalization/phase7_extension_track_s1_full160_v4_probe_final/review.html` | `surface_generalization/PHASE7_A_SURFACE_BOUNDARY_FINAL_REPORT.md` | `outputs/data_selection/alley_road_selection_v1/final_alley_test_set_v1/images/081_MP_SEL_SUR_002256.jpg` |
| 7-B | V2 provisional revision complete; final audit passed | backend: `scripts/walkway_v3/phase7_b_pipeline.py`; batch: `scripts/walkway_v3/run_phase7_b_full160.py` | `scripts/walkway_v3/render_phase7_b_v2_full160_dashboard.py` using `scripts/walkway_v3/build_phase7_b_focused_review.py` panel code | `outputs/surface_generalization/phase7_b_full160_v2_final_dashboard/review.html`; 16-case companion: `outputs/surface_generalization/phase7_b_full160_v2_provisional_final/ambiguous_review.html` | `surface_generalization/PHASE7_B_PASSABILITY_FINAL_REPORT.md`; `surface_generalization/PHASE7_B_PASSABILITY_DESIGN.md`; `surface_generalization/PHASE7_B_IMPLEMENTATION_WORKFLOW.md` | `outputs/surface_generalization/phase7_b_full160_v2_final_dashboard/panels/alley_081.png` |

## Frozen-copy bundles

The registry copies complete review roots rather than an HTML file alone so
relative image links remain valid.

| Phase | Frozen copy root(s) |
| --- | --- |
| 1 | `outputs/surface_generalization/phase1_sidewalk_context_regression_40_v1` |
| 2 | `outputs/surface_generalization/phase2_obstacle_geometry_ownership_regression_40_v3` |
| 3 | `outputs/surface_generalization/phase3_road_context_enhanced_40_v1`; `outputs/surface_generalization/phase3_road_context_enhanced_alley_core40_v1`; dependency for `diff_review.html`: `outputs/surface_generalization/phase3_road_context_40_v1` |
| 4 | None; index points to the Phase 3/5 absorption note only |
| 5 | `outputs/surface_generalization/phase5_targeted_expansion_challenge_80_v1`; `outputs/surface_generalization/phase5_sparse_support_integration_regression_v5` |
| 6 | `outputs/surface_generalization/phase6_image_upload_selector_160_v2`; `outputs/surface_generalization/phase6_production_selector_integration_validation_v1.json` |
| 7-A | `outputs/surface_generalization/phase7_extension_track_s1_full160_v4_probe_final` |
| 7-B | `outputs/surface_generalization/phase7_b_full160_v2_final_dashboard`; `outputs/surface_generalization/phase7_b_full160_v2_provisional_final` |

## Interpretation rules

- Frozen mode copies bytes; it never imports a runner or performs inference.
- A copied review is historical evidence, not a request to recompute a phase.
- Phase 4 must remain an absorption record, not a fabricated empty result.
- Phase 5 report text names regression v4, while the Phase 6 final report points
  to v5. Their machine-readable `regression.json` SHA-256 values are identical;
  the registry uses v5 as the latest referenced copy.
- Phase 7-B automatic aggregates must exclude
  `result_authority=provisional_review` unless a later human approval creates a
  new authoritative disposition.

