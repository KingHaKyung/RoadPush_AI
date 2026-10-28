# LiDAR 개발 패키지 연결

현재 사진 파이프라인과 독립적인 [`lidar_passability`](../lidar_passability/README.md)를
추가했습니다. 개발용 버전은 `0.1.0.dev2`, 응답 계약은 `roadpush.lidar.analysis/0.1-dev`입니다.

전체50epoch·±60m 추가5회·최신 자동 방향/경계의 현재 실행/미정은
[LIDAR_STATUS](LIDAR_STATUS.md)를 따릅니다. 기존 소량 B 후처리와 최종모델 후보를 구분합니다.

## 입력과 호출

저장된 실제 예측은 `LidarAnalyzer.analyze_prediction(AnalysisRequest(...))`로 분석합니다.
PCD·prediction NPZ·export report·checkpoint의 SHA와 runtime source 영수증 SHA를 명시합니다.
원본 PCD에서 ROI ID/좌표/coverage를 재구성한 뒤 예측과 대조하므로, 서로 다른 PCD의
동일한 점 수나 임의 배열을 그대로 받아 폭으로 계산하지 않습니다.

`roadpush-lidar --request request.json --output new-result.json`은 명시한 새 JSON만 저장합니다.
기존 파일은 보존합니다. 모델/표시 sampling/ground seed admission을 서로 구분하며 원시행을
버리지 않습니다. 현재 PCD raw 읽기는 기존 개발 binary 계약/150,000행 상한을 유지합니다.

실제 GPU 추론까지 묶으려면 별도 자산으로 제공된 frozen runtime/source와 checkpoint를
`prediction.FrozenRuntime`에 지정하고 `analyze_pcd(..., export_dir=..., execute_gpu=True)`를
명시합니다. 기본·import·plan에서는 GPU가 실행되지 않습니다. GPU 실행에는 명시한 기존
Linux 환경을 사용하며 임의 설치·학습·대여·다른 작업 중단은 하지 않습니다.

## 환경과 배포

사진AI root installer와 Python/Torch/CUDA 프로필을 바꾸지 않았습니다. LiDAR CPU 후처리는
독립 설치 가능한 wheel이고 Torch/spconv/FlashAttention/Windows 글꼴이 필요 없습니다.
GPU는 별도 프로세스에서 원본 frozen source를 사용합니다. 소량 B의 기존 cu124 환경과
팀 버전 cu126 후보를 구분합니다. 최종 checkpoint 선택/실GPU 이식 성공을 자동 주장하지 않습니다.

Git에는 코드·환경/소스 영수증·설명을 넣습니다. weights·raw PCD·사진·예측·실험결과·vendor
runtime은 별도 자산입니다. 재현/검사 자료는 Git에서 제외된 `experiments/lidar/`에 둡니다.
데이터 없는 계약 검사도 같은 정책으로 별도 검증 ZIP의
`experiments/lidar/tests/review_contract/`에 둡니다. Git에는 싣지 않습니다.
vendor source가 포함된 runtime 자산에는 기존 LICENSE/provenance를 함께 보존합니다.

## 결과 해석

- 기본은 기존 v5 기하/경계 검토 흐름이며 알고리즘 함수 본문을 보존했습니다.
- `conditional_runs`는 구조경계 가설 아래의 부분구간 추정입니다.
- `confirmed_width_m`, `confirmed_ratio`는 null, 통행 판정은 undetermined입니다.
- `production_eligible=false`, `human_review_required=true`, 자동 집계 불가를 유지합니다.
- 저높이 분기/경계중심 후보·unknown 범위는 검토 도구입니다. 기본 정책으로 자동 채택하지 않습니다.
- 최협부는 평활화 없는 미터폭 기준이며 상대폭 분모는 바로 그 단면의 전체폭입니다.
  사진 median3 비율 최적화와 같다고 취급하지 않습니다.

`processing_status=completed`는 계산 성공입니다. 실제 경계·폭 정확도, 사용자 검토 완료,
기관 운영 적합성이나 최종 모델 채택을 뜻하지 않습니다.

## 검증과 남은 연결

이관 manifest는 원본 source/옮긴 source SHA와 16개 모듈의 함수·클래스 AST 동일성을 기록합니다.
패키지 후보는 독립 wheel 설치와 기존 실제6장에 대한 공간계산·경계복원·전체조건부결과
동일성, 기존 합성회귀·새 입출력검사를 통해 검증합니다. 실제 수행 결과/환경/제외항목은
별도 검증 보고서를 우선합니다. 코드가 있다는 것만으로 이 검사들이 성공했다고 가정하지 않습니다.

이번 패키지 작업에서는 새 GPU 추론/사진160장 회귀를 수행하지 않았습니다.
사진 파이프라인의 기존 소스·requirements·응답 API는 그대로 보존했습니다.
BE의 PCD 수신·실행 작업큐·결과저장과 FE의 점군/조건부결과 표시는 아직 별도 연결 작업입니다.

최신 전방 자동 경계는 [별도 opt-in API](LIDAR_EXPERIMENTAL_FRONT.md)입니다.
기존 LidarAnalyzer 기본 경로와 anchor 상대폭 결합은 구분합니다.
