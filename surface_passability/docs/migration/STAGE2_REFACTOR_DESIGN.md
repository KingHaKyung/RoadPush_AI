# Stage 2 — Production API and relocation design

Status: **Gate 2 candidate**  
Behavior policy: all mappings in `manifests/source_to_target_mapping.csv` have
`behavior_changed=false`.  Thresholds, formulas, masks, and status semantics
must remain byte/value equivalent until Gate 3 passes.

## Architecture boundary

The supported production flow is:

```text
PhotoRequest
  -> ArtifactLocator / input validation
  -> semantic cache or Mask2Former inference
  -> common obstacle geometry (+ conditional YOLO at most once)
  -> Sidewalk and Road surface post-processing
  -> Phase 6 surface selection
  -> Phase 7-A boundary fitting and extension decision
  -> Phase 7-B occupancy, connected corridor, relative width
  -> nullable physical-width projection
  -> canonical JSON response
  -> optional review presentation
```

Numeric processing must not import the presentation layer.  Rendering may
consume canonical records, but may not recompute or mutate them.

## Public API

The eventual stable entry point is:

```python
from roadpush_ai.app.photo_pipeline import run_photo

result = run_photo(request)
```

`run_photo` coordinates responsibilities only.  It does not contain model,
geometry, policy, serialization, or rendering formulas.

Fresh-photo mode passes in-memory Phase 6 and 7-A results forward.  It must
not look up historical accepted JSON.  Frozen-regression mode uses an explicit
`FrozenRegressionSource` adapter and remains available only for parity checks.

## Orchestration order

1. Resolve and validate request/model/cache/output roots.
2. Apply EXIF orientation and load compatible semantic cache, otherwise infer.
3. Build Sidewalk and Road contexts once.
4. Build common obstacle geometry once.
5. Run conditional YOLO at most once; expose surface-specific ownership views.
6. Run Sidewalk and Road pipelines and Phase 6 selector.
7. Return typed null if the selected surface is uncertain.
8. Fit the accepted Phase 7-A V6 boundary and apply S1 validation/grading.
9. Hash-check the canonical Phase 7-A record and effective mask.
10. Route Phase 7-A authority/scope into Phase 7-B.
11. Build clipped measurement occupancy from current evidence.
12. Build eligible rows, connected corridor DAG, exact maximin path, raw
    minimum and three-row median bottleneck.
13. Project to physical widths only when `surface_width_m` is supplied.
14. Serialize canonical JSON. Render only when explicitly requested.

## Relocation-safe artifacts

Canonical JSON stores logical references, never repository-specific absolute
paths.

```python
@dataclass(frozen=True)
class ArtifactRef:
    namespace: str
    artifact_id: str
    version: str
    relative_path: PurePosixPath
    sha256: str | None = None

@dataclass(frozen=True)
class ArtifactLocator:
    package_root: Path
    data_root: Path
    model_root: Path
    cache_root: Path
    artifact_root: Path
    report_root: Path
```

The locator rejects `..`, root escape, and symlink escape; verifies immutable
model/reference hashes; supports environment/config root overrides; and records
old absolute path, logical ref, source hash, and target hash in the relocation
manifest.  Historical JSON is never silently string-replaced.

## Font isolation

`FontResolver` belongs exclusively to presentation.  Resolution order is:
configured font, bundled licensed Korean font, Windows Malgun, known Noto CJK
font, then an explicit rendering error.  The selected font ID and hash are
recorded with the dashboard.  Font availability cannot change numeric results.

## Legacy promotion

Fourteen modules currently executed from `outputs/walkway_detection_ablation_v2`
are promoted to `roadpush_ai.legacy_policy`.  Import-time batch/HTML side effects
are forbidden.  `importlib.spec_from_file_location` is removed only after direct
package imports produce mismatch 0 over the frozen 160 samples.

## Gate 2 checks

- The AST/importlib audit found 49 static production modules and 14 dynamic
  legacy modules.
- Every one of the 63 source modules has exactly one row in the mapping CSV.
- All mappings declare `behavior_changed=false`.
- No production source remains unmapped.
- External dependencies and conditional Deeplab support are recorded; Deeplab
  is not part of the accepted Mask2Former production profile.

