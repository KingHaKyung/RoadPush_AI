# RoadPush 장애물 탐지 — 독립 실행 호환 계층

학습(`scripts/`)과 분리한 독립 실행 adapter입니다. v4 production 서비스는 루트의
`roadpush_ai` 패키지가 같은 detector를 메모리에서 직접 호출합니다.

```
service/
  config.yaml   모델 경로·버전, 클래스별 임계값·처리 유형 (조정은 여기서만)
  predict.py    사진/폴더 → JSON + 확인용 이미지
  attach_cause.py  파일 기반 과거 연결 방식 재현용
  outputs/      실행 결과 (실행할 때마다 날짜_시각 폴더가 생김)
```

## 출력 JSON 예시 (형식)
```json
{
  "image": "photo_001.jpg", "width": 1920, "height": 1080,
  "model_version": "obstacle_v2_yolo11s_full_os4",
  "obstacle_count": 2,
  "types": ["A", "C"],
  "obstacles": [
    {"class": "vehicle", "name_ko": "주정차 차량", "type": "C", "action": "단속 요청",
     "confidence": 0.91, "box_xyxy": [x1, y1, x2, y2]}
  ]
}
```

## 실제로 쓰기까지의 단계
| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 모델 확정 | 기준 모델 선택, 버전 이름 붙이기 | 전체 데이터 모델 결과 보고 결정 |
| 2. 추론 스크립트 | 사진 → JSON (`predict.py`) | **완료** |
| 3. 임계값 정하기 | 클래스별 conf. 학습 폴더의 `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` 참고 | 다음 |
| 4. 현장 사진 테스트 | AI Hub가 아닌 **직접 찍은 사진**으로 확인 (다른 카메라·지역·계절에서도 되는지) | 다음 |
| 5. "통행을 막았나" 판정 | 최상위 analyzer가 선택 surface blocker와 메모리에서 결합 | **완료** |
| 6. 배포 형태 | Python 장기 실행 analyzer 및 순차 CLI | **완료** |
| 7. 라이선스 | Ultralytics는 AGPL-3.0 → 상용/B2G 전 검토 (대안 RF-DETR, Apache 2.0) | 팀 결정 |

## 주의
- 모델은 AI Hub 사진(1920×1080, 보행자 시점)으로만 배웠다. 다른 각도(CCTV 등)나 야간 사진에서는 성능이 떨어질 수 있다 → 4단계에서 확인.
- 탐지 불가 대상: 상자·봉투·이동식 화분 등 (학습 데이터에 클래스 없음).
- 라바콘·주차금지 팻말은 debris(B)로 나온다.
