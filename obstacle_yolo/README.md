# obstacle_yolo — 보도 장애물 원인 탐지

보도를 막은 **원인(장애물 종류)** 을 사진에서 탐지해 처리 유형(A~D)으로 연결한다.
통행 가능 여부·막힌 영역은 팀 파이프라인(Phase 6 / 7-B)이 판단하고, 이 폴더는 "무엇이 막았나"만 붙인다.

| 클래스 | 원본 라벨 (AI Hub) | 처리 유형 |
|---|---|---|
| vehicle | car, truck, bus, motorcycle | C 단속 요청 |
| construction | barricade | D 도로점용 허가 확인 |
| debris | movable_signage, carrier, chair, table | B 점유자 계도 |
| micromobility | bicycle, scooter | A 운영사 수거 요청 |

## 폴더
```
configs/class_map.yaml   원본 29개 라벨 → 4개 클래스 매핑
scripts/                 데이터 확인·변환·학습 (step0 ~ step6)
service/
  config.yaml            모델 경로·버전, 클래스별 임계값·처리 유형
  predict.py             사진/폴더 → 장애물 JSON + 확인용 이미지
  attach_cause.py        Phase 6 blocker 마스크 + 탐지 → 막은 영역별 원인·처리 유형 (7-B 결과와 sample_id로 결합)
weights/                 (git 제외) 모델 가중치 — Drive에서 받아 넣기
outputs/                 (git 제외) 변환 데이터·학습 결과
```

## 실행
```
pip install -r requirements.txt          # torch는 requirements.txt 주석 참고
python service\predict.py <사진 또는 폴더>
python service\attach_cause.py --team-root <팀 파이프라인 폴더>
```
`attach_cause.py`는 팀 코드를 import하지 않고 Phase 6 출력 폴더(`results.json`, `artifacts/<id>/sidewalk/08_applied_blocker.png`)와
7-B `results.json` 파일만 읽는다. 결과는 새 폴더에 저장하고 입력·모델 SHA-256을 기록한다.

### 원인 판정 규칙 (`phase6_blocker_overlap_v1`)
- 선택된 길(보도/차도)의 blocker 마스크를 덩어리로 나눈다.
- 덩어리의 30% 이상이 박스 안에 있거나, 박스의 10% 이상이 덩어리 위에 있으면 그 박스를 원인으로 인정한다.
- 맞는 박스가 없는 덩어리는 `unclassified` (클래스 없는 쓰레기봉투·볼라드·의류수거함 등 → 추가 라벨링 후보).
- `auto_routable`은 원인이 있고 7-B `result_authority`가 `authoritative`일 때만 true.

## 모델 `obstacle_v2_yolo11s_full_os4`
- YOLO11s, AI Hub 「인도보행 영상」 Bounding Box 학습 사진 45,872장 (construction 사진 4배 오버샘플링), imgsz 640
- val 2,000장 (촬영 폴더 단위 분리): mAP50 0.774 / mAP50-95 0.551, 1.8ms/장 (RTX 4060 Laptop)
- 가중치 SHA-256 `b33b2d73…` (`service/config.yaml`)
- 알려진 약점: micromobility → vehicle 혼동 0.21, construction → debris 0.17 (라바콘이 movable_signage로 라벨됨)

## 데이터·라이선스 주의
- AI Hub 데이터는 국외 반출 금지 → 사진·라벨·변환 데이터는 git·외부 서비스에 올리지 않는다.
- Ultralytics(YOLO11)는 AGPL-3.0 → 서비스 배포 전 라이선스 검토 필요 (대안: RF-DETR, Apache 2.0).
- `scripts/`의 데이터 경로는 작성자 PC 기준 절대 경로다.
