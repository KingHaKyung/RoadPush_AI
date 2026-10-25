# RoadPush Surface Generalization Development History

This document records only outcomes supported by the original repository's
accepted reports and frozen review artifacts. Source paths are inventoried in
[`PHASE_REVIEW_AND_HISTORY_SOURCE_MAP.md`](PHASE_REVIEW_AND_HISTORY_SOURCE_MAP.md).

## Evidence completeness contract

This readable chronology is the entry point; it is not a replacement for the
original experiment records. The Drive bundle preserves all 59 Markdown files
from the original `surface_generalization/` history plus every still-resolvable
image referenced by those files under `experiments/history_assets/`. The
byte-level ledger is `experiments/manifests/development_history_sources.json`.
Consequently, rejected probes, failed ideas, accepted promotions, handoffs,
and their contemporary images remain available without compressing them into
later hindsight. The eight canonical phase review bundles are separately
frozen under `experiments/reviews/` and validated by
`experiments/golden/review_bundle_validation.json`.

When this summary and a contemporary report differ in level of detail, the
contemporary report plus its frozen hash is the historical evidence. The
production behavior is governed by `PHASE7B_PRODUCTION_LOGIC.md` and the
checked-in code, not by an abandoned probe report.

## Phase 1 — Sidewalk context extraction

**Outcome:** passed.

Phase 1 introduced an explicit `SurfaceContext` around the frozen Sidewalk
pipeline and demonstrated behavior preservation on the canonical 40 samples.
The report records 40/40 processing, zero automatic failures, equal result and
summary values, and zero binary-mask differences across 360 comparisons.

- Canonical review:
  `outputs/surface_generalization/phase1_sidewalk_context_regression_40_v1/review.html`
- Evidence:
  `surface_generalization/PHASE1_REGRESSION_REPORT.md`
- Next accepted direction: separate common obstacle geometry from surface
  ownership.

## Phase 2 — Obstacle geometry and surface ownership

**Outcome:** passed after audited remediation.

Phase 2 separated surface-neutral obstacle geometry from per-surface ownership.
The frozen regression remained value- and byte-identical to the accepted
baseline, with zero binary-mask differences across 360 comparisons. The report
also records the independent-runner and no-probe-import architecture checks as
passing.

- Canonical review:
  `outputs/surface_generalization/phase2_obstacle_geometry_ownership_regression_40_v3/review.html`
- Evidence:
  `surface_generalization/PHASE2_REGRESSION_REPORT.md`
- Audit remediation:
  `surface_generalization/PHASE2_AUDIT_REMEDIATION.md`

## Phase 3 — Enhanced Road V2

**Outcome:** enhanced Road V2 accepted for production; the original Phase 3
runner and output were preserved for regression and rollback.

The accepted runner added Road ownership and linear-drain behavior while
retaining canonical Sidewalk artifacts exactly. The final report records
pixel-identical equivalence between Production V2 and the accepted combined
probe.

- Primary canonical review:
  `outputs/surface_generalization/phase3_road_context_enhanced_40_v1/review.html`
- Alley-core companion:
  `outputs/surface_generalization/phase3_road_context_enhanced_alley_core40_v1/review.html`
- Evidence:
  `surface_generalization/PHASE3_FINAL_ACCEPTANCE_REPORT.md`
- Production integration report:
  `surface_generalization/PHASE3_ENHANCED_ROAD_V2_PRODUCTION_REPORT.md`

## Phase 4 — Absorbed work

**Outcome:** no independent canonical final artifact.

The repository does not identify a standalone final Phase 4 runner/review.
Its structural/component work is represented in the accepted Phase 3 and
Phase 5 path. The migrated history therefore links those phases rather than
inventing a Phase 4 result or empty review.

## Phase 5 — Sparse row-span support

**Outcome:** accepted and closed for the current project scope.

The expansion/challenge set processed 80/80 with zero automatic failures, and
the final production regression processed 160/160. The accepted sparse-support
rule changed four surface/sample pairs while preserving the canonical
Sidewalk-40 masks. The report explicitly retains limitations for Parking
coverage, absent Service Lane positives, limited Road-owned YOLO ambiguity,
drain omissions, deferred vehicle-silhouette tuning, and small fragments.

- Targeted review index:
  `outputs/surface_generalization/phase5_targeted_expansion_challenge_80_v1/targeted_review.html`
- Full companion review:
  `outputs/surface_generalization/phase5_targeted_expansion_challenge_80_v1/review.html`
- Current regression evidence:
  `outputs/surface_generalization/phase5_sparse_support_integration_regression_v5/REGRESSION.md`
- Evidence:
  `surface_generalization/PHASE5_FINAL_ACCEPTANCE_REPORT.md`

## Phase 6 — Production surface selector

**Outcome:** complete.

Phase 6 integrated the fixed input-profile selector into a generalized runner
without rebuilding or changing the accepted Sidewalk and Road branches. The
production regression processed all 160 accepted Phase 5 inputs. The report
records 63 Sidewalk, 88 Road, and 9 uncertain image-upload outcomes, zero
selector differences, and zero differences in the accepted Phase 5 branch
results.

- Canonical image-upload review:
  `outputs/surface_generalization/phase6_image_upload_selector_160_v2/review.html`
- Evidence:
  `surface_generalization/PHASE6_PRODUCTION_INTEGRATION_REPORT.md`
- Input-profile report:
  `surface_generalization/PHASE6_INPUT_PROFILE_SELECTOR_REPORT.md`

The CCTV result is a call-path smoke test, not evidence of real-CCTV accuracy.

## Phase 7-A — Surface boundary

**Outcome:** approved and closed for MVP deployment.

Phase 7-A froze the S1 straight-extension result. The 160-input result contains
64 `selected`, 10 `selected_partial`, 16 `ambiguous_hold`, 3 `unchanged`, and
67 `failed` cases. The accepted behavior includes narrow-end stopping,
independent far/near stops, same-side occluder-neutral validation, Banner in
that validation set, and safe-prefix preservation.

- Canonical review:
  `outputs/surface_generalization/phase7_extension_track_s1_full160_v4_probe_final/review.html`
- Evidence:
  `surface_generalization/PHASE7_A_SURFACE_BOUNDARY_FINAL_REPORT.md`

The S2 curve/piecewise model is not an MVP deployment prerequisite and remains
a long-term project. It must restart from frozen V6 evidence rather than use
S1-generated geometry.

## Phase 7-B — Connected passability measurement

**Outcome:** V2 provisional revision complete; final audit passed.

Phase 7-B measures a connected corridor inside the Phase 7-A structural
domain. It uses a consecutive eligible-row run, adjacent-row interval overlap,
an exact full-run corridor DAG, and a three-row rolling-median robust
bottleneck. It keeps metre-valued outputs null unless a backend supplies
`surface_width_m`.

V2 separates authority explicitly:

- `authoritative`: accepted full/partial Phase 7-A geometry;
- `provisional_review`: `ambiguous_hold` review-only calculation;
- `none`: no publishable representative measurement.

- Canonical full dashboard:
  `outputs/surface_generalization/phase7_b_full160_v2_final_dashboard/review.html`
- 16-case ambiguous companion:
  `outputs/surface_generalization/phase7_b_full160_v2_provisional_final/ambiguous_review.html`
- Evidence:
  `surface_generalization/PHASE7_B_PASSABILITY_FINAL_REPORT.md`
- Design:
  `surface_generalization/PHASE7_B_PASSABILITY_DESIGN.md`

Accepted limitations include the provisional 50% eligible-run coverage gate,
limited persisted occupancy evidence, and the deferred portable-barrier gap in
`alley_081`. Provisional results must not enter authoritative aggregates unless
a human approval creates a new authoritative disposition.

## Frozen review operation

The migration candidate intentionally separates historical review access from
new computation.

```powershell
python experiments/tools/run_phase_review.py --list
python experiments/tools/run_phase_review.py --phase phase7b --mode frozen --dry-run
python experiments/tools/run_phase_review.py --phase phase7b --mode frozen --destination <new-empty-directory>
```

Frozen mode copies existing bytes, preserves relative assets, and writes a
SHA-256 manifest. Runner mode is deliberately disabled until migrated phase
runners have independent parity approval.

## Drive-only research/replay workspace

The distributed ZIP additionally contains ignored `scripts/`, `models/`,
`data/`, `outputs/`, and `artifacts/` trees. They exist so a teammate can
inspect or rerun a historical phase without polluting the production Git
repository. A rerun must always target a new empty output directory. Frozen
review evidence must never be overwritten.

The final production refactor itself has independent parity evidence:

- Phase 7-B: 160/160 records equal (`stage3_phase7b_parity.json`);
- Phase 6 smoke sample `IMG_4282`: all five compared JSON artifacts equal
  (`stage3_phase6_smoke_parity.json`);
- frozen phase-review bundle: 7,472 files, 20 HTML documents, no missing file,
  hash mismatch, or broken local link (`review_bundle_validation.json`).

