# RoadPush_AI migration staging

This directory contains read-only inventory and design evidence generated from
the accepted RoadPush source before production files are moved or renamed.

## Gate status

- Stage 0 baseline: complete for Phase 7-B full-160 computation and dashboard.
- Stage 1 inventory: complete; 49 static + 14 dynamic production modules.
- Stage 2 mapping/refactor design: complete; 63/63 unique mappings, no unresolved production source.
- Stage 3 behavior-preserving package migration: complete for the accepted final calculation path.
  - Phase 7-B: 160/160 normalized records equal, mismatch 0.
  - Phase 6 import/path smoke: `IMG_4282`, five JSON artifacts exact SHA-256 equal.
  - Dynamic Python loading from `outputs/` removed from the package candidate.
- Stage 4 phase review tools: in progress.

## Confirmed production entry points

- Photo through Phase 6: `scripts/walkway_v3/run_phase6_surface_pipeline.py`
- Phase 7-A handoff/reconstruction: `scripts/walkway_v3/phase7_b_adapter.py`
- Phase 7-B single sample: `scripts/walkway_v3/phase7_b_pipeline.py:run_phase7_b_sample`
- Phase 7-B batch: `scripts/walkway_v3/run_phase7_b_full160.py`
- Final review: `scripts/walkway_v3/render_phase7_b_v2_full160_dashboard.py`

## Structural findings

1. Phase 1–6 still dynamically imports fourteen Python files from
   `outputs/walkway_detection_ablation_v2`. These are runtime code despite
   their historical location and must be promoted into the package.
2. Phase 7-B is not standalone. Its adapter reconstructs and hash-verifies the
   accepted Phase 7-A surface using the complete Phase 7-A/V6 closure.
3. Accepted artifacts contain absolute `D:\Furiosa\mini\RoadPush` paths and
   require relocation-safe artifact references.
4. No single orchestrator currently runs photo input through Phase 7-B.
5. The review renderer hardcodes Windows Malgun fonts. Font resolution needs a
   portable fallback and must remain separate from numeric processing.
6. Canonical 160 inputs are split between `data/samples` and the alley
   selection set; copying one folder is insufficient.

## Canonical outputs

- Phase 1: `phase1_sidewalk_context_regression_40_v1`
- Phase 2: `phase2_obstacle_geometry_ownership_regression_40_v3`
- Phase 3: `phase3_road_context_enhanced_40_v1` and alley-core counterpart
- Phase 4: absorbed; no independent final artifact
- Phase 5: `phase5_sparse_support_integration_regression_v5`
- Phase 6: three `phase6_production_selector_*_v1` result sets and
  `phase6_image_upload_selector_160_v2/review.html`
- Phase 7-A: `phase7_extension_track_s1_full160_v4_probe_final`
- Phase 7-B: `phase7_b_full160_v2_final_dashboard`

The detailed file/hash inventory is generated as
`manifests/production_source_manifest.json` in the target tree.
