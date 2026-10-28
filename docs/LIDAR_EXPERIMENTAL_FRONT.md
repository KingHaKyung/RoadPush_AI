# 전방 60m 자동 방향·경계 개발 경로 — 2026-10-28

PCD 좌표와 저장 모델 분류만 입력으로 사용합니다. 사진/RGB·사용자 표시·이전 경계 JSON은
계산 입력이 아닙니다. 모든 프레임에 설치 rig의 동일한 전방 규칙을 적용합니다.
도로가 직선이고 양쪽 경계가 평행하다는 이번 개발 가정 아래, 긴 수직면에서 방향을
우선 선택하고 차량 분류 점의 연결면을 대안으로 사용합니다. 선택된 면의 실제 정체와
방향 정확도는 보장하지 않습니다. 센싱 테두리를 방향 근거로 쓰지 않습니다.

## 호출

기존 `LidarAnalyzer`와 CLI 기본 경로는 이전 소량 B/v5입니다.
최신 경계 경로는 아래 API를 명시적으로 호출합니다. `config`는 패키지에 동봉한
데이터셋 rig 설정이며, 다른 장비에 그대로 쓰는 보정값이 아닙니다.

```python
import json
from importlib.resources import files
from lidar_passability.experimental import analyze_front_boundaries

config = json.loads(files("lidar_passability.experimental")
                    .joinpath("fixed-front-roi.json").read_text())
result = analyze_front_boundaries(xyz, labels, config, raw_ids=raw_ids)
```

`xyz`는 미터 단위 LiDAR 좌표 Nx3, `labels`는 각 행에 대응하는 원래 클래스 ID입니다.
1/2/3/8/10/12/40/48을 사용하며 40은 차도, 48은 보도, -1은 예측 없음입니다.
네트워크 출력 index 0~7을 넣지 않습니다. `raw_ids`는 원본 PCD 행 번호이며,
전체 원본 행 순서일 때만 생략할 수 있습니다. 필터링된 배열이면 반드시 함께 제공합니다.
저장 prediction의 원본 SHA/행 정합성 검증은 호출자가 기존 입력 어댑터로 먼저 수행합니다.
이 배열 API는 파일 출처를 인증하거나 모델 추론을 실행하지 않습니다.

함수 import/호출은 파일을 만들지 않습니다. 근거가 부족하면 유보하며
`confirmed_width_m/confirmed_ratio=null`, `production_eligible=false`를 유지합니다.
방향이 선택돼도 양쪽 경계가 없으면 폭을 만들지 않습니다. 채워진 직사각형은 가설이며,
명목 60m 연장과 관측 지지 구간, 미관측/가림을 구별해야 합니다.

## 현재 결과와 별도 상대폭 실험

저장 best42를 ±60m 입력으로 추론한 8장을 사용했습니다. 방향 후보 8장,
경계·영역 후보 3장/4개, 경계 유보 5장입니다. 위치 정확도·사용자 최종 평가·확정폭은 없습니다.
패키지 이관 검사에서 저장 전방 배열의 방향·경계 계산 일치를 확인합니다.
사진/3D 선택·점/선/영역 동기화 검토 UI와 원본 자료는 로컬 검토판에 보존하며 공개 Git에 싣지 않습니다.

`lidar_passability.experimental.auto_front_width.postprocess(xyz, labels, config)`는
**이전 v4 경계로 계산하는 별도 v5 실험**입니다. 최신 anchor 경계와 아직 결합되지 않았습니다.
높이 기반 장애물후보·미관측 분리·연결 통로·최협부/동일 단면 분모를 계산하지만,
기본 조건부 수치 1구간도 격자 변경 시 유보되므로 최종 채택값이 아닙니다.
이 결과를 최신 anchor의 폭으로 표시하지 않습니다. 60m 전체 상대폭과 이면도로 전체폭 분기는 미구현입니다.

임계값은 개발용 잠정값이며 학습 5cm나 팀 확정 정책과 구별합니다.
이관은 기존 계산을 보존합니다. 최초 v4 경계 추출이 유보되면 방향 선택용 바닥 기준은
기존과 동일하게 z=-1.7m 잠정값으로 돌아갑니다. 이 기준의 정확성도 미검증이며,
`floor_reference_provisional=true`를 반환합니다.
모델 선택 명세는 `manifests/model-candidates.json`을 따릅니다. ±60m 추가5회 모델을
위 8장 결과의 모델로 바꿔 적지 않습니다. GPU 환경은 별도이며 사진AI 의존성에 합치지 않습니다.

```console
python -m unittest discover -s experiments/lidar/tests/review_contract -v
```

테스트는 Git에서 제외한 별도 검증 ZIP에 있습니다. 저장소 루트에 푼 뒤 실행합니다.
이 검사는 합성/계약 검사입니다. 실제 데이터 8장 이관 일치, Linux/GPU 실행,
독립 test 정확도, 폭 정확도와 BE/FE 연결은 각각 별도의 검증 범위입니다.
