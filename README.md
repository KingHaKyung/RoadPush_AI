# RoadPush AI

RoadPush의 사진 기반 보행로 분석 모델을 함께 관리하는 팀 저장소입니다.

- `surface_passability/`: 사진에서 보도·도로 surface를 선택하고, 경계를
  결정한 뒤 상대·절대 통행 가능 폭을 계산하는 모델
- `obstacle_yolo/`: 폭 계산 이후 사진 속 장애물 종류를 별도로 탐지하는
  팀원 소유 모델

두 모델은 독립적으로 설치·실행되며 서로의 내부 모듈을 import하지 않습니다.
현재 리팩터링은 `surface_passability`만 대상으로 했고, 기존
`obstacle_yolo` 코드는 변경하지 않습니다.

## `surface_passability` 처리 흐름

```text
사진
  -> EXIF 보정·SHA-256 검증·분석 크기 생성
  -> Mask2Former semantic cache 또는 실제 추론
  -> 공통 장애물 geometry와 조건부 YOLO vehicle instance
  -> Sidewalk/Road 후보 후처리(Phase 1~5)
  -> 대표 surface 선택(Phase 6)
  -> 경계 관측·직선 적합·S1 연장·등급 결정(Phase 7A)
  -> 연결 통로와 상대/절대 통행 가능 폭 계산(Phase 7B)
  -> `AnalysisResponse`
```

production 진입점은 하나입니다.

```python
from surface_passability import AnalysisRequest, analyze_photo

request = AnalysisRequest(
    request_id="req-001",
    sample_id="photo-001",
    image_uri=r"C:\inputs\photo-001.jpg",
    image_content_sha256="<사진 파일 SHA-256 64자리>",
    input_profile="image_upload",
    surface_width_m=None,
    agent_width_m=None,
    safety_factor=None,
)
response = analyze_photo(request)
payload = response.to_dict()
```

## 설치

지원 Python은 64-bit CPython 3.12입니다. 저장소 루트에서 `setup.bat`를
실행하고 CPU 또는 CUDA 12.8 GPU 프로필을 선택합니다.

```powershell
setup.bat
```

모델 파일은 Git에 포함하지 않습니다. 다음 위치에 별도로 배치해야 합니다.

```text
models/
  surface_passability/
    semantic/             # 승인된 Mask2Former 모델 디렉터리
    yolo11n-seg.pt        # 조건부 vehicle instance 모델
```

모델·cache·artifact 위치는 환경 변수로 바꿀 수 있습니다.

- `SURFACE_PASSABILITY_ROOT`
- `SURFACE_PASSABILITY_MODEL_ROOT`
- `SURFACE_PASSABILITY_CACHE_ROOT`
- `SURFACE_PASSABILITY_DATA_ROOT`
- `SURFACE_PASSABILITY_ARTIFACT_ROOT`
- `SURFACE_PASSABILITY_REPORT_ROOT`
- `SURFACE_PASSABILITY_DEVICE`: semantic 추론 장치
- `ROADPUSH_YOLO_DEVICE`: 조건부 YOLO 장치(`cpu` 또는 CUDA device)

## CLI

설치 후 단일 사진과 디렉터리 batch를 같은 명령으로 실행합니다.

```powershell
roadpush-surface-analyze --image C:\inputs\photo.jpg --output C:\results
roadpush-surface-analyze --input-dir C:\inputs --output C:\results --device cpu
```

결과는 sample ID별 JSON으로 원자적으로 기록됩니다. 기존 결과를 수치
artifact로 덮어쓰는 phase별 CLI와 renderer는 production에 포함하지 않습니다.

## 문서와 계약

- [production 시스템](docs/PRODUCTION_SYSTEM.md): 입력부터 backend 응답,
  CPU/GPU 운영, 제한사항까지 현재 시스템 전체
- [개발 이력](docs/DEVELOPMENT_HISTORY.md): Phase별 시도·실패·채택·회귀와
  frozen 증거 위치
- `surface_passability/schemas/`: 요청·전체 응답·Phase 7B JSON Schema
- `surface_passability/manifests/`: 모델 해시, release parity, Git/ZIP 경계

상세 테스트, 기준 구현, 전체 review HTML·이미지와 재현 도구는 Git에서
제외한 `experiments/surface_passability/` 팀 공유 번들에 둡니다.
`migration_work/`는 작업 재개용 로컬 기록이므로 Git과 팀 ZIP 모두에
포함하지 않습니다.

## 현재 보장

- 기존 160장 결과와 최종 production pipeline의 선택·경계·mask·폭 결과가
  mismatch 0
- normalized JSON SHA-256:
  `2e78ac33f529da695c1133dce424d2a6d80844113abef138b0d171958cddd9bb`
- production에서 기존 `roadpush_ai`, tests, experiments, 절대 `D:` 경로
  import/reference 0
- 리팩터링 중 알고리즘, 임계값, 결과 계약 변경 없음

공식 조직 저장소 반영과 배포는 최종 사용자 승인 뒤에만 수행합니다.

