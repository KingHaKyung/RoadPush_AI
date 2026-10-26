# Surface Passability 개발 이력

이 문서는 현재 production 결과에 도달하기까지의 문제 정의, 실험, 실패와
채택 결정을 기록한다. 현재 실행 계약은 [production 시스템](PRODUCTION_SYSTEM.md)을
따른다. 상세 renderer, 전체 HTML·이미지, 기준 구현과 회귀 도구는 Git에서
제외된 `experiments/surface_passability/` validation bundle에 보존한다.

## 1. 개발 원칙과 데이터셋

개발은 사진 한 장의 semantic segmentation에서 시작해, 통행 가능한 surface,
경계, 연속 통로, 상대 폭을 단계적으로 확정하는 방식으로 진행했다. 초기
canonical 40장, 확대 80장, 최종 160장 묶음을 사용했다. 다음 원칙은 전
과정에서 유지했다.

- 사람 눈에 그럴듯한 그림보다 mask·metadata·normalized JSON을 우선한다.
- 불확실한 결과는 억지로 선택하지 않고 typed uncertainty/null로 남긴다.
- 새 정책은 기존 승인 샘플을 보존하는 회귀 gate를 통과해야 한다.
- threshold 조정은 실패 사례와 전체 분포 근거가 있을 때만 한다.
- production과 review renderer/test harness를 분리한다.

### 근거 보존 계약

이 요약은 당시 보고서를 hindsight로 대체하지 않는다. 원본 59개 Markdown,
리팩터링 전 문서·manifest·대표 이미지는 Git에서 제외된
`experiments/surface_passability/history/`에 byte 보존한다. 기준/최종 소스와
tests는 `validation/`, 새로 확인한 전체 160장 dashboard는 `final_review/`가
소유한다. 용량이 큰 다른 canonical review root는 원본 Drive 위치와 frozen
hash를 source map에 남긴다. 서로 충돌할 때는 당시 report와 frozen hash가
역사적 사실의 우선 근거이고, 현재 production 동작은 코드와
[production 시스템](PRODUCTION_SYSTEM.md)이 우선한다.

| 구간 | 당시 최종 report | canonical review / 대표 증거 |
|---|---|---|
| Phase 1 | `surface_generalization/PHASE1_REGRESSION_REPORT.md` | `outputs/surface_generalization/phase1_sidewalk_context_regression_40_v1/review.html`; `artifacts/commons_01/summary.png` |
| Phase 2 | `PHASE2_REGRESSION_REPORT.md`, `PHASE2_AUDIT_REMEDIATION.md` | `phase2_obstacle_geometry_ownership_regression_40_v3/review.html`; `artifacts/commons_01/summary.png` |
| Phase 3 | `PHASE3_FINAL_ACCEPTANCE_REPORT.md`, `PHASE3_ENHANCED_ROAD_V2_PRODUCTION_REPORT.md` | `phase3_road_context_enhanced_40_v1/review.html`, `phase3_road_context_enhanced_alley_core40_v1/review.html` |
| Phase 4 | 독립 final artifact 없음 | Phase 3·5 흡수 기록을 사용하며 빈 결과를 만들지 않음 |
| Phase 5 | `PHASE5_FINAL_ACCEPTANCE_REPORT.md`, sparse-support report | `phase5_targeted_expansion_challenge_80_v1/targeted_review.html`, `review.html` |
| Phase 6 | `PHASE6_PRODUCTION_INTEGRATION_REPORT.md`, `PHASE6_AUTO_SELECTOR_PROBE_REPORT.md`, input-profile report | `phase6_image_upload_selector_160_v2/review.html`; `cards/alley_060.png` |
| Phase 7A | `PHASE7_A_SURFACE_BOUNDARY_FINAL_REPORT.md` | `phase7_extension_track_s1_full160_v4_probe_final/review.html` |
| Phase 7B | `PHASE7_B_PASSABILITY_FINAL_REPORT.md`, design, workflow | `phase7_b_full160_v2_final_dashboard/review.html`; 16-case `ambiguous_review.html`; `panels/alley_081.png` |

## 2. Phase 1: 공통 semantic surface 기반

Phase 1의 확인 대상은 새 알고리즘 발명이 아니라 frozen Sidewalk pipeline을
명시적인 `SurfaceContext`로 감싸면서 결과가 바뀌지 않는지였다. 새 context
경계를 채택하고 다음 단계에서 공통 obstacle geometry와 surface ownership을
분리하기로 결정했다. 당시 report가 기록하지 않은 실패 알고리즘을 이 문서에서
추정해 추가하지 않는다.

### 결과

canonical 40장에서 40/40 처리, 자동 실패 0, result/summary 값 동일,
360개 binary mask 비교에서 파일·픽셀 diff 0을 확인했다.

## 3. Phase 2: Sidewalk ownership와 통합 후보

수치 회귀 통과 뒤 구조 감사에서 다섯 결함이 발견되어 Phase 2를 다시 열었다.

1. geometry 함수가 Sidewalk mask를 입력받아 surface-neutral하지 않았다.
2. Road trigger가 Sidewalk instance view를 바꿀 수 있었다.
3. generalized pipeline이 frozen probe의 private 함수를 import했다.
4. EXIF orientation 사진의 cache miss/hit 픽셀이 달랐다.
5. architecture 검사가 실제 호출이 아니라 manifest flag를 신뢰했다.

최종 remediation은 `extract_obstacle_geometry(image, labels)`에서 surface 입력을
제거하고, semantic parent object를 보존한 채 shared YOLO detection을 최대 한
번 만든 뒤 surface별 parent/split view를 선택하게 했다. component selection,
obstacle-source 변환, hash, rendering은 production 모듈로 옮겼고, miss/hit 모두
EXIF-normalized source를 사용하며 cache dimension 불일치는 재생성하게 했다.
정적 signature/import 검사와 실제 instrumented single-sample 호출 검사를 함께
채택했다.

### 결과

최종 `_v3`는 40/40, 자동 실패 0, 360 mask diff 0, per-sample JSON/PNG byte
diff 0, dependency-free contract 12개 통과였다. 조건부 YOLO 샘플
`commons_04`, `IMG_4548`도 동일했다.

## 4. Phase 3: Road 후보와 강화 문맥

보도가 없는 골목을 위해 Raw Road positive ownership과 transverse Catch Basin
reconnection을 함께 검토했다. 사용자는 drain correction target
`alley_011`, `alley_034`, Raw Road ownership target `alley_027`을 유지했고,
차량 사례 `alley_017`, `alley_030`, `alley_040`의 동작을 승인했다. 최종 정책
ID는 `phase3_road_context_v2_drain_ownership_v1`이다.

Catch Basin은 component 크기·PCA elongation/thickness·route 대비 transverse
angle·양쪽 Road evidence·corridor 교차를 모두 통과한 connected object만
복원한다. drain evidence를 먼저 평가하고, drain-repaired route로 ownership을
평가한 뒤 승인된 drain object를 replay하는 단방향 순서를 채택했다. ownership
결과가 drain eligibility에 되먹임되지 않는다. fixed acceptance review 이후
threshold 변경은 없었다.

### 결과

alley core 40의 Road mask 440개가 accepted combined probe와 pixel-identical,
Sidewalk 차이 0이었다. 승인 drain 38,718px, ownership removal 13,242px였다.
original 40은 Sidewalk 440개 diff 0을 유지했고, Road는 ownership에 의해 9개
mask에서 추가 0px·제거 6,622px만 발생했다. `alley_022` semantic false-negative,
`alley_008` ownership underreach, Service Lane/다양한 Parking coverage는 Phase 5
과제로 이관했다.

## 5. Phase 4: 공통 장애물 geometry 통합

Phase 4는 독립 final runner/review가 없다. 원래 요구사항인 frozen Sidewalk
보존, combined semantic/surface/obstacle review, Parking·curb·component·vehicle
검사는 Phase 3 acceptance와 Phase 5 확대 검증으로 닫혔다. 따라서 별도 결과를
만들어 내지 않고 흡수 기록만 유지한다.

## 6. Phase 5: sparse support와 160장 확대

expansion 35장과 challenge 45장으로 Road/Parking/Service Lane, drain-like
negative, ownership 위험과 추가 YOLO ambiguity를 찾았다. 80장에는 Service
Lane positive와 새 Road-owned YOLO trigger가 없었으므로 이를 성공으로 꾸미거나
trigger를 강제하지 않았다. `alley_082`의 pole-connected false Road restoration을
고치는 surface-neutral sparse row-span support만 production에 채택했다.

### 결과

확대 80장 80/80, 최종 160장 160/160을 통과했다. 변경은 정확히 네 pair다.

| sample / surface | 확정 pixel delta |
|---|---|
| `alley_014` Sidewalk | structural/free -532px |
| `alley_060` Sidewalk | structural/free -5,194px |
| `alley_082` Road | structural -5,800px, blocker -2,548px, free -3,252px |
| `alley_084` Sidewalk | structural -25,408px, blocker -5,900px, free -19,508px |

15개 Parking-positive 사례는 conservative identity policy 유지로 승인했다.
Service Lane positive 부재, 추가 Road-owned YOLO ambiguity 부재, `alley_065`와
`alley_073` drain omission, `alley_048`·`050`·`074` vehicle silhouette threshold,
`alley_088` 작은 flower-pot fragment는 새 threshold 없이 향후 근거 과제로
이관했다. 기존 Sidewalk 40 binary mask는 변화 0이다.

## 7. Phase 6: 대표 surface 선택

caller가 target surface가 아니라 `input_profile`만 한 번 지정하는 계약을
채택했다. `image_upload`는 bottom-center anchor의 structural majority를 사용하되
차이가 5% 미만이면 uncertain으로 남긴다. `cctv`는 largest Sidewalk source
component가 이미지의 0.25% 이상이고 세로 10% 이상을 span하면 Sidewalk를
우선하고, 아니면 persistent Road 또는 uncertain을 선택한다. 현재 사진은 CCTV
정확도 데이터가 아니므로 CCTV 결과를 accuracy로 해석하지 않는다.

채택 전에 하나의 자동 selector가 모든 입력을 처리하는 방안을 검토했다. 이
probe는 160장 중 forced 124, dataset expected mode와 agreement 120, conflict 4,
uncertain 36으로 coverage 77.5%였고 production으로 승격하지 않았다. 단순
largest-mask 방식은 122/160만 일치했고 37장은 잘못 강제 선택했으며 1장은
동률이었다. context-only 방식도 실제 surface semantic evidence 및 제품 의도와
충돌했다. 여기서 expected mode는 pixel truth가 아니라 dataset/product intent다.
이 실패 때문에 입력 목적을 명시하는 profile 계약과
`uncertain` abstention을 최종 정책으로 유지했다.

### 결과

image-upload 160장은 Sidewalk 63, Road 88, uncertain 9였다. CCTV functional
run은 95/63/2였고 두 profile은 34장에서 다른 결정을 냈지만 저장된 Sidewalk와
Road evidence 차이는 0이었다. production integration은 selector diff 0/160,
양 branch dictionary diff 0/160, binary mask 4,640개 byte diff 0, 새 semantic
inference 0을 확인했다.

## 8. Phase 7A: 경계 관측, 적합, 연장과 등급

### 문제

선택 surface mask만으로는 어느 행의 폭을 신뢰할 수 있는지 알 수 없다.
장애물 차폐, 식생, 끊긴 관측과 원근 때문에 경계를 끝까지 무조건 연장하면
잘못된 통행 폭을 만든다.

### 주요 실험

- 초기 row evaluator와 trapezoid 직선 적합
- 전역 longest support와 run-wise 방향 보정
- geometric gap, joint fit, support window 비교
- grounded vegetation cap과 containment 예외
- S1 generation, safe-prefix truncation, quality grading

과거 `retry_v*`, `probe_*` 명칭은 이 비교 과정의 실험 순서를 나타낸다.
리팩터링에서는 최종 호출 closure만 `boundary` 책임 모듈로 평탄화했지만,
기존 JSON과 증거를 깨지 않기 위해 serialized `policy_id` 문자열은 유지했다.

retry V1~V6와 V5의 global support, run-wise direction, joint refit,
geometric-gap, grounded-vegetation ablation을 비교했다. 개별 probe의 실패와
promotion 근거는 history bundle의 당시 report를 그대로 보존한다. 최종 S1은
양쪽 evidence, 좁은 끝 해상도 정지, 독립 far/near·frame stop, 같은 쪽 causal
occluder 중립화, Banner validation, safe-prefix 보존을 채택했다. 곡선·복수
직선 S2는 frozen V6 evidence에서 독립 시작해야 하며 MVP에서 보류했다.

### 채택

관측 구간을 우선하고, 검증된 방향과 containment를 통과한 S1 연장만 허용한다.
safe prefix에서 위험 신호가 나오면 이후 생성 구간을 잘라낸다. 결과는
`selected`, `unchanged`, `selected_partial`, `ambiguous_hold`, `failed`의
권한 상태로 분리했다.

### 결과

최종 policy ID는 `phase7_extension_track_s1_v4_probe_human_review_grading`이다.
전체 160장 status는 selected 64, selected_partial 10, ambiguous_hold 16,
unchanged 3, failed 67이었다. authoritative Phase 7A result SHA-256은
`5ed57526cc785a911f17da042ce15062a642938b19fcd51c091633880adfd4d8`이다.

## 9. Phase 7B: 연결 corridor와 상대 폭

### 문제

행별로 가장 넓은 구간을 고르면 다음 행에서 다른 쪽 구간으로 순간 이동하는
경로가 생긴다. 한 행의 작은 semantic noise가 전체 최소 폭을 지배하는 문제도
있었다.

설계 검토에서 행별 독립 interval을 합산하거나 다른 통로로 순간 이동하는
계산, 평균으로 bottleneck을 숨기는 계산, raw 단일 행 minimum을 상태값으로
쓰는 계산을 금지했다. exact adjacent-row overlap DAG, full-run corridor,
3-row rolling-median bottleneck과 raw 진단값 분리를 채택했다.

### 채택

행별 통행 가능 interval을 만들고 인접 행 사이의 연결 관계를 DAG로 구성했다.
3-row rolling median으로 국소 잡음을 완화한 뒤, 전체 경로의 최소 상대 폭을
최대화하는 maximin path를 선택한다. raw minimum과 최대 obstruction도 함께
남긴다. Phase 7A authority에 따라 full/partial/provisional/none 범위를 엄격히
전파한다.

절대 폭은 픽셀에서 추정하지 않기로 사용자와 합의했다. backend가 실측
`surface_width_m`을 줄 때만 상대 비율을 미터로 투영하며, 값이 없으면
`absolute_passable_width_m`은 `null`이다.

V1은 Phase 7A의 `ambiguous_hold` 16건을 typed-null로만 처리했다. V2에서는
다른 144건의 값을 그대로 보존하고 이 16건만 provisional/review-only 범위로
재계산했다. 최종 authority는 authoritative 77, provisional_review 16, none
67이며, calculation scope는 full 67, partial 10, provisional 16, none 67이다.
자동 집계는 provisional 16건을 제외한다.

### 결과

V2 frozen results SHA-256은
`4bc9c4071eee1a2ba02398194597a61f291d90eaed65f609f9929a1d745e96ea`,
normalized SHA-256은
`917e97dc58181ac08df374c13e4d39bf07096703ed420edd324d6c95e59bbd3e`다.
최종 160 panel dashboard HTML SHA-256은
`1ec2d6f2c6999bb63b7b32e2885a9eba6ab928fe41cade7d581085d7e8c2b3c8`이며
160/160 panel과 card가 생성됐다. 리팩터링 후 normalized JSON은 동일 의미를
새 public provenance로 직렬화한 normalized SHA-256
`2e78ac33f529da695c1133dce424d2a6d80844113abef138b0d171958cddd9bb`로
mismatch 0을 확인했다.

소비자는 `provisional_review`를 routing·SLA·production aggregate에 넣거나,
`raw_minimum_passable_ratio`로 상태를 정하거나, 좌우 free interval을 합산하거나,
`obstacle_area_ratio`를 폭으로 해석하거나, `effective_blocked_width_m`를 장애물
실폭으로 표시하거나, `null`을 0으로 치환하면 안 된다. eligible-run 50%는
보편 상수가 아닌 provisional MVP threshold다.

## 10. production refactor

### 목표

기존 package에는 `legacy_policy`, `retry_v*`, `probe_*`, `test4_*`처럼 개발
연대가 production 구조에 드러났고, 단계별 runner·renderer와 실제 알고리즘이
섞여 있었다. 최종 동작과 threshold는 바꾸지 않고 책임 경계를 다시 세웠다.

### 수행

- 코드를 독립 `surface_passability/` package로 이동했다.
- `contracts`, `config`, `imaging`, `inference`, `geometry`, `obstacles`,
  `surfaces`, `selection`, `boundary`, `passability` 책임으로 분리했다.
- 사진부터 `AnalysisResponse`까지 `analyze_photo` 단일 진입점을 만들었다.
- semantic cache hit/miss, corrupt/stale 거부와 atomic write를 계약화했다.
- 배포 경로를 env 기반으로 만들어 원래 `D:` workspace 의존성을 제거했다.
- tests, renderer, baseline source와 review evidence를 ignored experiments bundle로
  격리했다.
- 미래 `obstacle_yolo`와 코드/weights 소유권을 분리했다.

AST audit는 80개 기존 Python 파일과 862개 함수·클래스·상수를 분류했다.
동적 import, re-export, import-time side effect와 삭제 예정 caller를 점검했다.
AST만 삭제 근거로 사용하지 않고 runtime import trace와 단계별 parity를 함께
통과한 뒤 경계를 정리했다.

### 최종 회귀

| 검증 | 결과 |
|---|---|
| canonical 40 | mismatch 0, normalized SHA-256 `f3a4ca6681854160077878e51bd6e21898799d3d7c6cdeae1146045ab8824a62` |
| 전체 160 | mismatch 0, normalized SHA-256 `2e78ac33f529da695c1133dce424d2a6d80844113abef138b0d171958cddd9bb` |
| production import graph | 122 Python 파일, cycle 0 |
| legacy/runtime 경계 | `roadpush_ai`, tests, experiments, 절대 `D:` import/reference 0 |
| active characterization | 179개 파일, 1,735 test 수집; full run 1,729 pass·5 harness/environment 재검증 대상·1 skip, 대상 5/5 재검증 pass |
| retired dead-path characterization | 17개 파일, 삭제하지 않고 bundle에 보존 |
| GPU smoke | RTX 5070 Ti Laptop, torch 2.7.1+cu128, CUDA 12.8 import/device/tensor pass |
| cache-hit single smoke | pipeline 0.960초, process wall 2.503초, artifact 0 |

리팩터링 중 알고리즘과 확정 threshold는 변경하지 않았다. 비교 실패가 생겼을
때 comparator를 느슨하게 하지 않고 source geometry와 계약 차이를 수정했다.

## 11. 운영 성능·메모리 최적화 감사

리팩터링 완료 후 실제 160장 cold run에서 GPU VRAM이 9GB를 넘고 CPU 후처리가
추론보다 오래 걸리는 문제를 별도 감사했다. 원인은 모델 입력이 아니라 semantic
후처리와 반복 계산이었다.

- Mask2Former 입력은 이미 384×384였지만, 기존 후처리는 65개 class 확률을
  5712×4284 같은 원본 크기로 GPU에서 동시에 보간했다. 이 tensor만 약 5.9GiB였고
  장비 관측 peak는 9.30GB였다.
- surface 후보 생성은 동일 mask의 weighted route를 한 요청에서 반복 계산했다.
- Phase 7A tail 사례는 같은 점 집합에 robust line fit을 반복하고, 더 나은 후보가
  될 수 없는 경우에도 `median`을 계산했다.
- 기본 진입점은 장기 실행 worker에서도 dependency와 모델 asset 검증을 반복할
  수 있었고, 826MiB semantic weight hash도 한 초기화에서 두 번 읽었다.

채택한 개선은 class 4개 단위 memory-bounded native interpolation, 요청 범위 exact
route/line-fit cache, 후보 수가 현재 best보다 적을 때의 안전한 median 생략,
모델 hash 재사용, predict lock, 장기 실행 `SurfacePassabilityAnalyzer`다. threshold,
mask 정책, 선택 규칙과 결과 schema는 바꾸지 않았다.

최초의 마스크 동일 fast path는 최종 Phase 7B는 유지했지만 5장의 Phase 6 진단
수치를 바꿨다. behavior-neutral 원칙에 따라 이 최적화는 폐기했다. 수정 후 CPU는
과거 CPU의 semantic·surface selection·Phase 7A·Phase 7B raw JSON과 160/160
동일했고, GPU도 과거 GPU raw JSON과 160/160 동일했다. CPU와 GPU의 기존
semantic 수치 차이는 승인된 provenance normalization 뒤 최종 Phase 7B에서
160/160 동일하다.

| 지표 | 최적화 전 | 최적화 후 |
|---|---:|---:|
| CPU wall / 평균 | 847.18초 / 5.24초 | 548.24초 / 3.39초 |
| GPU wall / 평균 | 674.69초 / 4.18초 | 378.79초 / 2.34초 |
| GPU peak VRAM | 9.30GB | 3.10GB |
| GPU peak RSS | 2.39GB | 1.98GB |
| GPU Phase 7A·7B 평균 | 0.555초 | 0.143초 |

최종 보강 검사는 112 pass·1 권한 의존 symlink 검사 제외, production compile
성공, CPU/GPU cold-cache 각각 실제 semantic 추론 160회와 cache hit 0회를
확인했다. 수치 evidence는 로컬 runtime 기록에 두며 Git production에는 넣지
않는다.

## 12. 증거 위치

Git에 포함하지 않는 validation bundle의 기본 위치는 다음과 같다.

```text
experiments/surface_passability/validation/
  src/roadpush_ai/                         # baseline 80개 Python 파일
  surface_passability/src/surface_passability/
                                            # 최종 production snapshot
  surface_passability/tests/                # active characterization
  retired_tests/characterization/           # dead-path 전용 보존 tests
  VALIDATION_BUNDLE.json
  README.md
```

Phase별 원본 결과, 전체 160장 HTML과 대표 이미지는 같은
`experiments/surface_passability/` 아래의 review/evidence 디렉터리에 보존한다.
대표 사례에는 `alley_002`, `alley_018`, `alley_081`, `commons_01`,
`IMG_4282`, `IMG_4508`이 포함된다. 파일을 Git production 문서에 중복 복사하지
않는 이유는 모델 코드와 재현용 대용량 증거의 배포 수명주기가 다르기 때문이다.

## 13. 확정 결정과 향후 작업

확정된 결정은 다음과 같다.

- 현재 모델 공식 명칭은 `surface_passability`다.
- 팀원의 후속 장애물 분류 모델 명칭은 `obstacle_yolo`다.
- 현재 폭 계산에 필요한 obstacle 처리 코드는 `surface_passability` 소유다.
- 두 모델에서 실제 공용 사용이 입증되기 전에는 `shared/`를 만들지 않는다.
- 모델 weights는 `models/surface_passability/`와 `models/obstacle_yolo/`로
  소유권을 나눈다.
- 조직 저장소는 과거 migration Git 이력을 복사하지 않고 clean initial
  production overlay를 사용한다.
- 조직 저장소 반영은 전체 검증과 사용자 최종 승인 뒤에만 수행한다.

향후 보완 후보는 curved-boundary S2, portable fence/barrier 분류,
parking/service lane coverage, CCTV 현장 확대 검증과 Phase 7B coverage 정책
재평가다. 이 항목들은 이번 behavior-neutral refactor에 구현하지 않았다.

