# RoadPush LiDAR 개발 패키지

PCD와 저장된 실제 LitePT 예측을 독립적으로 검증하고, 현재 개발용 기하·경계·최협부를
계산합니다. 모델·운영정책·품질 확정판이 아닙니다. 사진AI API/설치환경은 변경하지 않습니다.

## 설치

별도 Python 환경에서 설치합니다. CPU 후처리는 Python 3.12–3.14와 NumPy/SciPy를 사용합니다.
실제 설치/회귀 검증한 버전은 검증 보고서를 따릅니다.

2026-10-28 현재 전체50epoch·±60m 추가5회·자동 방향/경계 후보의 상태는
[현재 상태](../docs/LIDAR_STATUS.md)를 따릅니다. 패키지 기본 모델을 바꾼 것은 아닙니다.

```console
python -m pip install ./lidar_passability
roadpush-lidar --request request.json --output new-result.json
```

`request.json` 필드는 `AnalysisRequest`와 같습니다. PCD/prediction/report 경로는
요청 JSON 옆을 기준으로 해석합니다. 라이브러리 호출은 디렉터리나 파일을 만들지 않고,
CLI는 명시한 새 JSON 파일만 생성합니다. 기존 파일은 덮어쓰지 않습니다.

```python
from pathlib import Path
from lidar_passability import AnalysisRequest, LidarAnalyzer

request = AnalysisRequest(
    request_id="req-001", frame_id="frame-001",
    pcd_path=Path("input.pcd"), prediction_path=Path("prediction.npz"),
    report_path=Path("result.json"),
    expected_pcd_sha256="<trusted PCD SHA-256>",
    expected_prediction_sha256="<trusted prediction SHA-256>",
    expected_report_sha256="<trusted export report SHA-256>",
    expected_checkpoint_sha256="<accepted checkpoint SHA-256>",
    model_version="<explicit development model version>",
    source_receipt_sha256="<runtime receipt SHA-256>",
    axis_evidence={
        "metric_z_up_verified": True,
        "source": "explicit dataset coordinate convention and calibration receipt",
        "independent_metric_scale_verified": False,
    },
)
payload = LidarAnalyzer().analyze_prediction(request).to_dict()
```

해시는 신뢰한 인수 manifest에서 가져옵니다. 임의 파일의 해시를 계산하는 것만으로
그 파일의 모델 출처·수치 정확도를 인증한 것은 아닙니다. 좌표의 metre/z-up은 명시적
개발 가정/근거이며 독립 측량 검증과 구분해야 합니다. GT 주석은 받지 않습니다.

## 데이터 없이 계약 검사

테스트는 팀 저장소 정책에 따라 Git에서 제외한 별도 검증 묶음으로 제공합니다.
검증 ZIP을 저장소 루트에 풀고 위 설치 후 다음 명령을 실행합니다.

```console
python -m unittest discover -s experiments/lidar/tests/review_contract -v
```

여기에는 입력/해시/좌표근거·조건부/null·덮어쓰기 방지·명시 GPU 실행 게이트와
지원하지 않는 PCD의 GPU 실행 전 거부 검사가 있습니다. 실제 6장 계산 동일성 및
기존 기하 회귀는 별도 팀 검증 묶음이며 이 작은 검사만으로 대체하지 않습니다.

## 결과와 현재 정책

- 기본 `v5_review`: 기존 수평면 지지·도로/보도 접촉·수직면 가설의 계산을 보존합니다.
- `low_barrier_review`: 사용자가 명시적으로 고르는 미채택 저높이 구조물 검토 분기입니다.
- `include_unknown_bounds=True`: 양측 가설 아래 미관측 부분을 채웠을 때의 조건부 범위입니다.
- `conditional_runs`는 부분구간 검토 결과이고 `confirmed_width_m/confirmed_ratio`는 계속 null입니다.
- 처리 완료여도 `passability=undetermined`, `production_eligible=false`를 유지합니다.
- raw ID/unknown/부분범위/출처를 보존합니다. 부분값을 전체 프레임 값으로 요약하지 않습니다.
- 최소 미터폭을 최대화하는 연결 경로를 고른 뒤 그 최협부 단면의 전체폭으로 나눕니다.
  사진AI의 median3 최적화와 구분합니다.

`spatial_indexed`, `spatial_grounded`, `boundary_recovery`, `measure_v5`, `bottleneck`,
`proposals`, `interval_certificate` 등은 기존 계산의 상대 import 이관본입니다.
`width_measure`의 과거 median3 경로는 기존 입력검증 의존성을 보존하기 위해 내부에 남지만,
현재 최협부의 경로 선택은 `bottleneck.select`를 사용합니다.

## 선택적 GPU 추론

`prediction.FrozenRuntime`은 따로 받은 해시 고정 source와 checkpoint를 기존 Linux
Python에서 실행하는 어댑터입니다. 패키지 설치/import는 Torch나 CUDA 확장을 요구하지 않습니다.
`plan()` 또는 `export(execute=False)`는 파일해시·명령만 확인하고 실행하지 않습니다.
`export(execute=True)`는 명시적으로만 실행하며 GPU가 사용 중이면 거부합니다.

프로필 `small_b_cu124`는 소량B best35와 원래 CPython3.10/Torch2.4.1/cu124 source를 고정합니다.
`team271_cu126`은 별도 Linux3.12/Torch2.7.1/cu126 source 영수증입니다. checkpoint는 별도로
인수/지정해야 하며 최종학습 완료·채택을 뜻하지 않습니다. source hash/실행환경/전처리를
모델에 맞춰 함께 보존하고 두 프로필을 섞지 않습니다. 환경설치·GPU대여는 자동화하지 않습니다.

GPU 어댑터의 새 실추론은 이번 로컬 패키지 검증 범위에 포함되지 않습니다. 실제 모델 결과를
받은 뒤 같은 `AnalysisRequest`로 후처리합니다. 런타임 source/vendor/weights는 Git에 넣지 않고
모델 자산으로 별도 배포합니다. vendor LICENSE와 provenance는 그 자산에 보존합니다.

`LidarAnalyzer.analyze_pcd(...)`는 위 GPU 어댑터와 후처리를 묶는 선택적 진입점입니다.
명시적인 `runtime`, `export_dir`, `execute_gpu=True`가 필요합니다. 이 경로만 지정된
export_dir에 예측 원본을 남깁니다. `analyze_prediction()`은 저장 파일을 만들지 않습니다.
새 추론 결과는 `data_kind=actual_inference`, 저장 예측 재사용은 기본
`historical_actual_inference`로 구분합니다.

## 배포와 한계

Git 대상: 이 패키지, 코드/버전 manifest, 문서. 원본 PCD·사진·예측배열·모델·개인 볼트·카톡·
회귀 결과는 포함하지 않습니다. 팀 회귀 자료는 root `experiments/lidar/` 아래 별도 보관합니다.
`manifests/migration.json`은 이관 전 source와 패키지 코드 SHA 및 함수 AST 동일성을 기록합니다.
현재 일반적인 PCD 지원범위는 기존 개발 계약(≤150,000 raw행, 비압축 binary)입니다.
경계 정확도·최종 모델 품질·사용자 눈평가·서비스 BE/FE 연동은 별도의 미완료 과제입니다.

## 최신 자동 경계 실험

전방60m·사진/시드 없는 최신 경계 경로는 opt-in API로 제공합니다.
[사용법과 별도 상대폭 경로](../docs/LIDAR_EXPERIMENTAL_FRONT.md)를 따릅니다.
기존 CLI는 선행 소량 B/v5 경로이며 최신 anchor를 자동 실행하지 않습니다.
