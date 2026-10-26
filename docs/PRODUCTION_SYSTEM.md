# Surface Passability production 시스템

이 문서는 `surface_passability`의 현재 production 동작과 운영 계약을 설명한다.
과거 실험 과정은 [개발 이력](DEVELOPMENT_HISTORY.md)에 분리한다. 이 모델은
사진 속 장애물의 이름을 사용자에게 알려 주는 `obstacle_yolo`와 별개이며,
통행 가능 폭을 계산하는 데 필요한 semantic 장애물 mask와 조건부 vehicle
instance 처리는 `surface_passability`가 소유한다.

## 1. 공개 경계

Python 공개 진입점은 `surface_passability.analyze_photo` 하나다.

```python
from surface_passability import AnalysisRequest, analyze_photo

response = analyze_photo(
    AnalysisRequest(
        request_id="req-001",
        sample_id="photo-001",
        image_uri=r"C:\inputs\photo-001.jpg",
        image_content_sha256="<64자리 SHA-256>",
        input_profile="image_upload",
        surface_width_m=None,
        agent_width_m=None,
        safety_factor=None,
    )
)
payload = response.to_dict()
```

입력과 출력은 각각 `AnalysisRequest`와 `AnalysisResponse` typed contract다.
내부 mask 배열은 응답에 싣지 않고, provenance·상태·수치·hash만 직렬화한다.
정식 JSON Schema는 `surface_passability/schemas/`에 있다.

## 2. 요청 계약

| 필드 | 계약 |
|---|---|
| `request_id` | 비어 있지 않은 요청 식별자 |
| `sample_id` | 비어 있지 않은 사진 식별자 |
| `image_uri` | 실행 호스트의 로컬 이미지 경로 |
| `image_content_sha256` | 파일 바이트의 SHA-256, 64자리 16진수 |
| `input_profile` | `image_upload` 또는 제한적으로 지원하는 `cctv` |
| `surface_width_m` | 선택값, 유한한 양수 |
| `agent_width_m` | 선택값, 유한한 양수 |
| `safety_factor` | 선택값, 유한한 `1` 이상 수치 |
| `render_review` | 현재 public 응답에서는 review artifact를 만들지 않으며 기본값 `false` |

수치 필드는 `bool`, `NaN`, 양·음의 무한대를 거부한다. 사진의 실제
SHA-256이 요청값과 다르면 추론 전에 실패한다.

## 3. 전체 데이터 흐름

```text
AnalysisRequest
  -> PreparedImage
  -> SemanticEvidence
  -> ObstacleEvidence
  -> SurfaceCandidates
  -> SurfaceSelection
  -> BoundaryResult
  -> PassabilityResult
  -> AnalysisResponse
```

모듈 의존 방향은 `contracts/config`에서 시작해 `imaging/inference`,
`geometry/obstacles/surfaces`, `selection`, `boundary`, `passability`, 최종
`pipeline/cli`로만 흐른다. production은 tests, experiments, 과거
`roadpush_ai` 패키지를 import하지 않는다.

## 4. 사진 준비

1. 로컬 파일 존재 여부와 요청 SHA-256을 확인한다.
2. Pillow로 이미지를 읽고 EXIF orientation을 픽셀에 반영한다.
3. RGB로 정규화한다.
4. 긴 변이 700px를 넘으면 aspect ratio를 유지해 LANCZOS로 축소한다.
5. 원본 크기와 분석 크기를 모두 보존한다.

semantic native class map과 분석 크기 class map은 별도로 hash한다. 따라서
원본 좌표와 분석 좌표를 혼동하거나, 입력 파일이 조용히 바뀌는 것을 탐지할
수 있다.

## 5. 모델과 semantic cache

semantic 모델은 `facebook/mask2former-swin-large-mapillary-vistas-semantic`
계열의 승인된 로컬 snapshot이며 class 수는 65다. 모델 디렉터리 전체,
weights, config, preprocessor, class mapping hash는
`surface_passability/manifests/model_assets.json`에 고정되어 있다.

cache schema는 `surface_passability_semantic_cache_v2`, 재사용 정책은
`device_independent_canonical_evidence_v1`이다. cache key에는 다음이 모두
포함된다.

- 사진 content SHA-256
- semantic 모델 profile ID
- 모델 tree identity
- class mapping identity

cache hit에서도 metadata, 파일 hash, 배열 dtype·shape, native/analysis 크기와
모델 provenance를 검증한다. stale 또는 손상된 cache는 사용하지 않는다.
cache miss에서는 추론을 정확히 한 번 실행하고 PNG와 metadata를 임시 파일에
쓴 뒤 원자적으로 교체한다. CPU와 GPU는 같은 canonical evidence를 재사용할
수 있지만 요청 장치와 producer provenance는 응답에 별도로 기록한다.

## 6. 장애물 evidence

Mapillary semantic class에서 통행 공간을 가리는 class를 공통 obstacle
geometry로 변환한다. 차량처럼 semantic blob만으로 containment를 판정하기
어려운 경우 `yolo11n-seg.pt` instance mask를 조건부로 사용한다. 이 로직은
폭 계산의 구성요소이므로 미래의 사용자-facing `obstacle_yolo` 모델로
이동하지 않는다.

YOLO는 항상 실행되지 않는다. surface와의 접촉·위치·geometry 조건이 필요할
때만 호출하며, vehicle instance와 semantic mask의 결합 및 containment 예외는
현재 production 정책 그대로 유지한다.

## 7. Phase 1~5: surface 후보 생성

Sidewalk와 Road는 공통 전처리와 class evidence를 공유하되 서로 다른 후보
정책을 적용한다.

- 연결요소, 바닥 접촉, 하단 seed와 구조적 연속성을 평가한다.
- Sidewalk는 통합 후보와 경계/외곽 구조를 구성한다.
- Road는 차도 문맥, ownership, linear drain과 강화된 road context를 반영한다.
- sparse support는 장면에 근거가 부족한 경우에만 제한적으로 후보를 보강한다.
- 각 후보는 mask와 함께 선택에 필요한 면적·접촉·연속성·장애물 metadata를 낸다.

이 단계는 surface를 최종 선택하지 않는다. Sidewalk와 Road 양쪽 evidence를
모두 만들어 Phase 6에 전달한다.

## 8. Phase 6: surface 선택

선택기는 두 후보의 profile-aware evidence를 비교해 다음 네 상태 중 하나를
낸다.

- `sidewalk`: 보도 후보 선택
- `road`: 도로 후보 선택
- `uncertain_both`: 양쪽 후보가 있으나 안전하게 단일 선택할 수 없음
- `uncertain_none`: 사용할 후보가 없음

응답에는 선택 surface뿐 아니라 fallback, score margin, reasons, 양쪽 후보의
profile metrics를 남긴다. 불확실 상태를 임의의 확정값으로 바꾸지 않는다.

## 9. Phase 7A: 경계 결정과 연장

선택된 structural surface에서 행별 좌·우 관측을 만들고, 강건한 직선 적합과
run-wise 방향·전역 support·기하 gap·식생/차폐 containment 검사를 적용한다.
관측이 충분한 구간을 우선하며, S1 연장은 검증된 방향과 safe-prefix 범위
안에서만 생성된다. 과거 실험 이름이 일부 serialized `policy_id`에 남아 있는
이유는 기존 backend/회귀 계약을 byte-level로 보존하기 위해서다. 현재 모듈과
함수 이름은 실제 책임을 나타낸다.

최종 status와 측정 권한은 다음과 같다.

| status | 범위 | 측정 권한 | 결과 권위 | 사람 검토 |
|---|---|---|---|---|
| `selected` | full | allowed | authoritative | 아니요 |
| `unchanged` | full | V6 신뢰도 조건부 | authoritative | 아니요 |
| `selected_partial` | partial | allowed | authoritative | 예 |
| `ambiguous_hold` | provisional | review only | provisional review | 예 |
| `failed` | none | denied | none | 아니요 |

응답은 observed/generated row, authorized row range, effective structural mask
hash와 area를 포함한다. `failed`는 추정 mask를 만들어 주지 않고 typed null로
Phase 7B에 전달한다.

## 10. Phase 7B: 통행 가능 폭

Phase 7B는 허용된 structural mask 내부에서 행별 통행 가능 구간을 만들고,
인접 행 구간을 연결한 DAG에서 하나의 연속 corridor를 선택한다. 각 행의
가능 폭을 surface 폭으로 나눈 ratio를 계산하고 3-row rolling median으로
국소 잡음을 완화한다. 전체 경로의 최소값을 최대화하는 maximin 경로가
`minimum_passable_ratio`를 결정하며 raw 값도 함께 보존한다.

주요 출력은 다음과 같다.

- `minimum_passable_ratio`: 선택된 corridor의 대표 최소 상대 폭
- `raw_minimum_passable_ratio`: smoothing 전 최소 상대 폭
- `maximum_obstructed_width_ratio`, `obstacle_area_ratio`
- bottleneck 위치, corridor path, 최대 obstruction
- authority, confidence, review 여부, abstention reasons

Phase 7A가 계산을 거부하면 상대 폭도 `null`이며 state는 `undetermined`다.
provisional 상태를 authoritative 수치로 승격하지 않는다.

## 11. 절대 폭의 nullable 계약

픽셀만으로 미터 단위 폭을 추정하지 않는다. `surface_width_m`이 주어지고
상대 폭이 유효할 때만 다음을 계산한다.

```text
absolute_passable_width_m = surface_width_m * minimum_passable_ratio
effective_blocked_width_m = surface_width_m - absolute_passable_width_m
required_safe_width_m = agent_width_m * safety_factor
```

필수 실측값이 없으면 절대값은 `null`이다. 상태는 다음 우선순위를 따른다.

1. 상대 측정 자체가 없으면 `undetermined`
2. 상대 폭이 0이면 `blocked`
3. 실측 surface/agent 폭이 없으면 `relative_only`
4. 절대 가능 폭이 agent 폭보다 작으면 `blocked`
5. safety factor가 없으면 `relative_only`
6. 안전 폭보다 작으면 `narrowed`
7. 그 밖에는 `passable`

## 12. backend 응답

전체 응답 version은 `surface_passability_response_v1`, Phase 7B record는
`phase7_b_result_v2`다. 최상위 필드는 다음과 같다.

- `image`: 입력 hash, 원본/분석 크기, EXIF와 resize 정보
- `semantic`: class map hash, cache 상태, 장치와 모델 provenance
- `surface_selection`: Phase 6 결정과 양쪽 evidence
- `boundary`: Phase 7A status, authority, mask hash와 decision diagnostics
- `passability`: Phase 7B normalized record
- `artifacts`: production 기본 경로에서는 빈 배열
- `stage_timings_s`: 각 책임 구간의 비음수 실행 시간

backend는 수신 시 bundled JSON Schema로 구조를 검증해야 한다. 이 schema는
필드·형식·enum 경계를 확인하는 구조 계약이며, 아래의 필드 간 null/authority와
계산식 invariant까지 대신하지 않는다. backend는 package validator 또는 같은
규칙을 구현한 검증기를 함께 사용해야 한다. 또한 strict JSON parser로
`NaN`·`Infinity`·`-Infinity`를 wire format에서 거부해야 한다. Python 객체의
비표준 float를 JSON Schema validator에 직접 넘기는 것만으로는 이 거부를
보장할 수 없다. 새로운 필드를 추가하거나 enum을 바꿀 때는 schema version을
올려야 하며, nullable 절대 폭을 0으로 치환해서는 안 된다.

### Null·authority invariant

- `minimum_passable_ratio=null`이면 다른 대표 relative metric과 `bottleneck`도
  null이며 state는 `undetermined`다.
- surface 실폭 또는 ratio가 null이면 absolute/effective blocked width도 null이다.
- reliable 대표 ratio는 유한한 `[0,1]`이며 bottleneck을 가진다.
- `selected_partial`은 partial/authoritative/review-required 조합을 유지한다.
- `ambiguous_hold`는 provisional/provisional-review/review-required 조합을
  유지하며 자동 aggregate에서 제외한다.
- scope가 none이면 authority도 none이고 대표 ratio가 없다.
- 실행 전후 upstream mask hash는 같아야 한다.

### 멱등성과 오류 경계

backend 멱등 key는 사진 hash, input profile, optional physical input, semantic
model identity, Phase 6/7A policy, Phase 7B schema와 occupancy evidence version의
canonical JSON SHA-256으로 만든다. renderer 여부는 수치 key에 넣지 않는다.
동일 key에 다른 input이 오면 충돌이고, upstream mask hash가 바뀌면 같은
sample ID여도 새 계산이다. 기존 output은 덮어쓰지 않는다.

| 조건 | 외부 분류 | retry |
|---|---|---|
| 잘못된 수치·status·schema | `INVALID_INPUT` | 입력 수정 전 금지 |
| 사진·모델·필수 자산 누락 | `ARTIFACT_NOT_FOUND` | 자산 복구 뒤 가능 |
| Phase 7A mask hash 불일치 | `UPSTREAM_INTEGRITY_ERROR` | 게시 금지 |
| 실행 중 upstream 변경 | `UPSTREAM_MUTATION_ERROR` | 게시 금지 |
| 같은 key에 다른 input | `IDEMPOTENCY_CONFLICT` | 새 key 필요 |
| output overwrite 시도 | `OUTPUT_ALREADY_EXISTS` | 새 run ID 필요 |
| 계산 가능하나 증거 부족 | 정상 typed-null 응답 | HTTP 오류 아님 |

batch backend는 항목별 성공, typed-null, hard error를 분리하고 한 항목의
hard error를 다른 항목의 `undetermined`로 숨기면 안 된다. 외부 응답에 stack
trace나 로컬 절대경로를 노출하지 않는다.

### 소비자 금지사항

- `provisional_review`를 routing, SLA, production 통계에 사용하지 않는다.
- `raw_minimum_passable_ratio`로 상태를 결정하지 않는다.
- 중앙 장애물 양쪽의 서로 다른 free interval 폭을 합산하지 않는다.
- `obstacle_area_ratio`를 폭 비율로 해석하지 않는다.
- `effective_blocked_width_m`를 장애물 자체 실폭으로 표시하지 않는다.
- 모든 nullable 값을 0으로 대체하지 않는다.

## 13. 설치와 실행

지원 환경은 64-bit CPython 3.12다. 저장소 루트의 `setup.bat`는 CPU 또는
CUDA 12.8 프로필을 선택해 별도 가상환경을 만든다. `py` launcher가 없으면
설치된 `uv`의 Python 3.12를 사용한다.

```powershell
setup.bat
```

CPU/GPU 공통 의존성은 `surface_passability/requirements/inference-common.txt`,
PyTorch profile은 `torch-cpu.txt`와 `torch-gpu-cu128.txt`에 고정되어 있다.
GPU 설치 후에는 CUDA runtime 12.8과 `torch.cuda.is_available()`을 검사한다.
검증 대상은 RTX 4060 Laptop GPU와 RTX 5070 Ti Laptop GPU이며, CUDA 12.8
wheel에는 runtime이 포함되므로 별도 CUDA Toolkit은 필수가 아니다. 다만
호환되는 최신 NVIDIA/OEM notebook driver가 필요하고 Windows 570.65 이상을
권장한다.

최종 배포 검증에서는 RTX 5070 Ti Laptop GPU에서 PyTorch
`2.7.1+cu128`, torchvision `0.22.1+cu128`, CUDA runtime 12.8의 import,
`torch.cuda.is_available()`, 장치 선택과 CUDA tensor 연산을 통과했다. 같은
환경의 semantic cache-hit 단일 사진은 내부 pipeline total 0.960초, CLI
process wall 2.503초였다. 이 값은 해당 검증 장비·cache-hit 사례의 smoke
측정이며 보편적인 SLA 보장은 아니다.

모델 파일은 Git에 넣지 않는다.

```text
models/
  surface_passability/
    semantic/
    yolo11n-seg.pt
  obstacle_yolo/          # 다른 모델 소유, 이 패키지는 참조하지 않음
```

주요 환경 변수는 다음과 같다.

| 변수 | 용도 |
|---|---|
| `SURFACE_PASSABILITY_ROOT` | 기본 runtime root |
| `SURFACE_PASSABILITY_DATA_ROOT` | 입력 data root |
| `SURFACE_PASSABILITY_MODEL_ROOT` | `models/surface_passability` 대체 |
| `SURFACE_PASSABILITY_CACHE_ROOT` | semantic cache root |
| `SURFACE_PASSABILITY_ARTIFACT_ROOT` | artifact namespace root |
| `SURFACE_PASSABILITY_REPORT_ROOT` | report root |
| `SURFACE_PASSABILITY_DEVICE` | semantic 장치 |
| `ROADPUSH_YOLO_DEVICE` | 조건부 YOLO 장치 |

CLI는 단일 파일과 batch 디렉터리를 지원하며 JSON을 임시 파일에 쓴 뒤
원자적으로 교체한다.

```powershell
roadpush-surface-analyze --image C:\inputs\one.jpg --output C:\results
roadpush-surface-analyze --input-dir C:\inputs --output C:\results --device cpu
```

## 14. 배포 경계와 운영 점검

Git production에는 실행 코드, 이 문서 두 개, schema, manifest, requirements만
포함한다. tests·baseline runner·review renderer·전체 160장 증거는
`experiments/surface_passability/` 팀 ZIP에만 포함한다. `migration_work/`는
Git과 ZIP 모두에서 제외한다. 조직 저장소로 옮길 때는 allowlist overlay를
사용해 팀원 소유 `obstacle_yolo/**`를 변경하지 않는다.

배포 전 다음을 확인한다.

- model asset hash 일치
- CPU clean install/import/single/batch smoke
- GPU dependency/import/device smoke
- 원래 개발 `D:` workspace 없이 relocation 실행
- 요청·응답 schema validation
- Git/ZIP allowlist 및 공식 Markdown 개수
- canonical 40과 전체 160 parity

현재 frozen 전체 160장 normalized JSON SHA-256은
`2e78ac33f529da695c1133dce424d2a6d80844113abef138b0d171958cddd9bb`이며
mismatch는 0이다.

## 15. 알려진 제한과 보류사항

- `cctv` profile은 실행 smoke는 있지만 `image_upload`와 같은 수준의 대규모
  현장 검증을 마치지 않았다.
- 주차장·service lane 등 보도/차도 의미가 혼합된 장면에는 coverage gap이 있다.
- portable fence/barrier 사례(`alley_081`)의 장애물 분류 개선은 보류했다.
- 곡선 경계용 Phase 7A S2는 이번 behavior-neutral refactor에 포함하지 않았다.
- Phase 7B eligible coverage 50% 정책은 현재 고정값이지만 추가 현장 검증이
  필요한 provisional 정책이다.
- 절대 폭은 외부 실측 `surface_width_m`이 없으면 의도적으로 `null`이다.
- 모델 weights 재배포 전 Mask2Former/Mapillary Vistas와 Ultralytics license를
  별도로 검토해야 한다.

