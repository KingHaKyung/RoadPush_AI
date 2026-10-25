# step6_train.py
# 목적: COCO로 미리 학습된 YOLO 모델(MODEL)을 우리 샘플 데이터셋(4개 클래스)으로 파인튜닝하고 val로 평가한다.
#
# 출력: obstacle_yolo/outputs/runs/<RUN_NAME>/
#         weights/best.pt   ← val 점수가 가장 좋았던 시점의 모델
#         weights/last.pt   ← 마지막 시점의 모델
#         results.csv, results.png        ← epoch별 손실·점수 기록과 그래프
#         confusion_matrix.png            ← 클래스끼리 헷갈린 정도
#         PR_curve.png 등                 ← 정밀도-재현율 곡선
#
# 데이터 규칙: 학습은 이 PC의 GPU에서만 한다. 사진·라벨을 외부로 보내지 않는다.
#   (사전학습 가중치 파일은 ultralytics 공식 저장소에서 내려받는다 — 데이터 업로드 아님)
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경) python scripts\step6_train.py

import os
from ultralytics import YOLO

BASE = r"C:\Furiosa\mini\RoadPush\obstacle_yolo"
DATASET = "dataset_v2_full_os4"               # 학습할 데이터셋 (step4의 DATASET과 같게)
MODEL = "yolo11s"                          # 사전학습 모델 이름 (n: 2.6M 파라미터 → s: 9.4M)
RUN_NAME = f"{MODEL}_{DATASET}"            # 결과 폴더 이름 (예: yolo11n_dataset_v2)

DATA = os.path.join(BASE, "outputs", DATASET, "data.yaml")
RUNS = os.path.join(BASE, "outputs", "runs")
PRETRAINED_DIR = os.path.join(BASE, "outputs", "pretrained")   # 내려받은 가중치를 모아 두는 곳

# Windows에서는 학습 시 여러 프로세스를 쓰기 때문에, 실행 코드를 반드시 이 블록 안에 둔다
if __name__ == "__main__":
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    os.chdir(PRETRAINED_DIR)              # 가중치 자동 다운로드가 이 폴더에 저장되도록 작업 위치를 옮긴다
    model = YOLO(f"{MODEL}.pt")           # COCO 사전학습 가중치 (없으면 자동으로 내려받음)

    model.train(
        data=DATA,            # 우리 데이터셋 설정 (경로, 클래스 이름 4개)
        epochs=50,            # 학습 데이터를 50번 반복
        imgsz=640,            # 사진을 640px 기준으로 줄여서 학습
        batch=16,             # 한 번에 16장씩
        patience=20,          # val 점수가 20 epoch 동안 안 오르면 조기 종료
        seed=20261023,        # 재현성 (같은 값이면 비슷한 결과)
        workers=2,            # 사진 읽기에 쓰는 보조 프로세스 수 (Windows 노트북은 작게)
        device=0,             # 첫 번째 GPU 사용
        project=RUNS,         # 결과 저장 폴더
        name=RUN_NAME,
        exist_ok=False,       # 같은 이름 폴더가 있으면 숫자를 붙여 새로 만든다 (덮어쓰기 방지)
    )
    # 학습이 끝나면 ultralytics가 best.pt로 val을 자동 재평가해 클래스별 표를 출력한다.
    # (이전 버전처럼 model.val()을 한 번 더 부르지 않는다 — 같은 결과가 runs/detect/val 에 중복 저장되던 문제)
