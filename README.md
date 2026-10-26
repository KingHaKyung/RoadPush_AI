# RoadPush AI

RoadPush의 사진 기반 보행로 분석 모델을 함께 관리하는 팀 저장소입니다.

- `surface_passability/`: 사진에서 보도·도로 surface를 선택하고, 경계를
  결정한 뒤 상대·절대 통행 가능 폭을 계산하는 모델
- `obstacle_yolo/`: 사진 전체의 장애물 종류와 위치를 탐지하는 모델
- `src/roadpush_ai/`: 두 모델을 순차 실행하고 선택된 surface의 blocker와
  접점하는 탐지만 서비스 결과로 조립하는 최상위 파이프라인

통합 진입점은 한 사진을 한 번 읽어 Surface를 먼저 실행하고 Obstacle을 이어서
실행합니다. 사진별 병렬 처리와 요청 내부 병렬 처리는 사용하지 않으며 동시성은
1로 고정합니다. 두 모델 객체는 analyzer 수명 동안 재사용합니다.

## 통합 처리 흐름

```text
사진 검증·EXIF 보정
  -> surface_passability
  -> obstacle_yolo
  -> 선택된 surface blocker와 obstacle box 접점 판정
  -> 통합 응답
```

Surface 내부 `yolo11n-seg.pt`는 폭 계산용 vehicle mask 보강 모델이고,
Obstacle `obstacle_yolo11s_full_os4.pt`는 클래스·개수·box 탐지 모델입니다.
서로 목적이 달라 하나로 교체하거나 결과를 합산하지 않습니다.

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

두 모델을 함께 실행하는 최상위 진입점은 다음과 같습니다.

```python
from roadpush_ai import RoadPushAnalyzer

analyzer = RoadPushAnalyzer(requested_device="cuda")
response = analyzer.analyze_photo(request)
payload = response.to_dict()
```

`payload["surface"]`는 기존 Surface 응답을 그대로 유지합니다.
`payload["surface_obstacles"]`는 선택 surface에 접점한 탐지만, 진단용
`payload["diagnostics"]["all_detections"]`는 사진 전체 탐지를 담습니다.

백엔드처럼 같은 worker에서 여러 사진을 처리할 때는 검증된 모델과 provider를
재사용하는 장기 실행 진입점을 권장합니다.

```python
from surface_passability import SurfacePassabilityAnalyzer

analyzer = SurfacePassabilityAnalyzer(requested_device="cuda")
response = analyzer.analyze_photo(request)
```

`SurfacePassabilityAnalyzer`는 모델·cache provider와 추론 lock을 worker 수명 동안
재사용합니다. 요청마다 새 analyzer를 만들지 않습니다.

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
  obstacle_detection/
    obstacle_yolo11s_full_os4.pt
```

모델·cache·artifact 위치는 환경 변수로 바꿀 수 있습니다.

- `SURFACE_PASSABILITY_ROOT`
- `ROADPUSH_ROOT`
- `ROADPUSH_MODEL_ROOT`
- `SURFACE_PASSABILITY_MODEL_ROOT`
- `SURFACE_PASSABILITY_CACHE_ROOT`
- `SURFACE_PASSABILITY_DATA_ROOT`
- `SURFACE_PASSABILITY_ARTIFACT_ROOT`
- `SURFACE_PASSABILITY_REPORT_ROOT`
- `SURFACE_PASSABILITY_DEVICE`: semantic 추론 장치
- `ROADPUSH_YOLO_DEVICE`: 조건부 YOLO 장치(`cpu` 또는 CUDA device)

## CLI

설치 후 통합 파이프라인은 `roadpush-analyze`로 실행합니다. 디렉터리 입력도
사진을 한 장씩 순차 처리합니다.

```powershell
roadpush-analyze --image C:\inputs\photo.jpg --output C:\results --device cuda
roadpush-analyze --input-dir C:\inputs --output C:\results --device cpu
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

통합 테스트 코드·입력·결과·cache는 `experiments/integration/`에만 보관하며
Git과 배포 ZIP에서 제외합니다. 배포 ZIP은 저장소 폴더 전체가 아니라 production
허용 목록으로 생성합니다. 모델 가중치는 Git에서는 제외하고, 별도 모델 포함
배포본이 필요할 때만 검증된 `models/`를 추가합니다.

## 현재 보장

- 기존 160장 결과와 최종 production pipeline의 선택·경계·mask·폭 결과가
  mismatch 0
- normalized JSON SHA-256:
  `2e78ac33f529da695c1133dce424d2a6d80844113abef138b0d171958cddd9bb`
- production에서 tests, experiments, 절대 `D:` 경로 import/reference 0
- 리팩터링 중 알고리즘, 임계값, 결과 계약 변경 없음
- RTX 5070 Ti Laptop 기준 전체 160장 GPU cold run은 최종값 160/160 일치,
  통합 평균 2.29초/장, peak reserved VRAM 2.95GB로 검증
- 같은 160장 CPU cold run도 Surface mismatch 0, 통합 평균 3.38초/장
- Obstacle count/class/metadata mismatch 0, 총 detection 291
- 선택 surface 접점 장애물은 160장 합계 108개

공식 조직 저장소 반영과 배포는 최종 사용자 승인 뒤에만 수행합니다.

