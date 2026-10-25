# step5_verify.py
# 목적: step4로 만든 YOLO 데이터셋이 올바른지 검증한다.
#   (1) 자동 검사: 좌표 범위(0~1), 클래스 번호 범위, 한 줄 형식, 사진-라벨 짝, 빈 라벨 수
#   (2) 눈으로 확인: YOLO 라벨을 픽셀 좌표로 되돌려 박스를 그린 이미지 20장을 outputs/check/ 에 저장
#
# 데이터 규칙: 이 PC 안에서만 처리한다. 화면에는 검사 결과(개수)만 출력한다.
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경) python scripts\step5_verify.py

import collections
import csv
import os
import random

import yaml
from PIL import Image, ImageDraw

BASE = r"C:\Furiosa\mini\RoadPush\obstacle_yolo"
DATASET = "dataset_v2_full"                              # 검증할 데이터셋 (step4의 DATASET과 같게)
DS = os.path.join(BASE, "outputs", DATASET)
CHECK = os.path.join(BASE, "outputs", "check" if DATASET == "dataset" else f"check_{DATASET}")
SEED = 20261023
COLORS = {0: (255, 0, 0), 1: (255, 160, 0), 2: (0, 160, 255), 3: (0, 200, 0)}   # 클래스별 테두리 색

with open(os.path.join(DS, "data.yaml"), encoding="utf-8") as f:
    names = {int(k): v for k, v in yaml.safe_load(f)["names"].items()}
n_cls = len(names)

# ------------------------------------------------------------
# (1) 자동 검사
# ------------------------------------------------------------
problems = collections.Counter()      # 문제 종류별 개수
label_files = {}
for split in ("train", "val"):
    imgs = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(DS, "images", split))}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(DS, "labels", split))}
    problems[f"{split}: 라벨 없는 사진"] += len(imgs - lbls)
    problems[f"{split}: 사진 없는 라벨"] += len(lbls - imgs)

    empty = 0
    for stem in sorted(lbls):
        path = os.path.join(DS, "labels", split, stem + ".txt")
        lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
        label_files[(split, stem)] = lines
        if not lines:
            empty += 1
        for l in lines:
            parts = l.split()
            if len(parts) != 5:                                   # 한 줄 = 숫자 5개
                problems[f"{split}: 형식 오류(숫자 5개 아님)"] += 1
                continue
            c, *xywh = parts
            if not c.isdigit() or not (0 <= int(c) < n_cls):        # 클래스 번호 0 ~ (클래스 수-1)
                problems[f"{split}: 클래스 번호 범위 밖"] += 1
            v = [float(x) for x in xywh]
            if any(x < 0 or x > 1 for x in v):                      # 모든 값은 0~1
                problems[f"{split}: 좌표 0~1 벗어남"] += 1
            cx, cy, w, h = v
            if w <= 0 or h <= 0:
                problems[f"{split}: 너비·높이 0 이하"] += 1
            if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                problems[f"{split}: 박스가 사진 밖으로 나감"] += 1
        if len(lines) != len(set(lines)):
            problems[f"{split}: 같은 줄 중복"] += 1
    print(f"[{split}] 사진 {len(imgs)}장, 라벨 {len(lbls)}개, 빈 라벨(배경) {empty}개")

bad = {k: v for k, v in problems.items() if v}
print("\n자동 검사:", "문제 없음" if not bad else bad)

# 빈 라벨 수가 manifest의 배경 사진 수와 같은지 확인 (배경이 아닌데 라벨이 비었으면 변환 실수)
with open(os.path.join(DS, "manifest.csv"), encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))
bg_manifest = sum(1 for r in manifest if r["boxes"] == "0")
empty_total = sum(1 for v in label_files.values() if not v)
print(f"빈 라벨 {empty_total}개 / manifest의 배경 사진 {bg_manifest}장 →", "일치" if empty_total == bg_manifest else "불일치")

# ------------------------------------------------------------
# (2) 확인용 이미지 20장: 클래스마다 5장씩 (YOLO 좌표 → 픽셀 좌표로 되돌려 그림)
# ------------------------------------------------------------
os.makedirs(CHECK, exist_ok=True)
rng = random.Random(SEED)
picked = []
for c in range(n_cls):
    pool = [k for k, lines in label_files.items() if any(l.split()[0] == str(c) for l in lines) and k not in picked]
    picked += rng.sample(pool, min(5, len(pool)))

for i, (split, stem) in enumerate(picked, 1):
    im = Image.open(os.path.join(DS, "images", split, stem + ".jpg")).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for l in label_files[(split, stem)]:
        c, cx, cy, w, h = l.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H   # 비율 → 픽셀
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2          # 중심·크기 → 꼭짓점
        draw.rectangle([x1, y1, x2, y2], outline=COLORS[int(c)], width=4)
        draw.text((x1 + 4, max(0, y1 - 14)), names[int(c)], fill=COLORS[int(c)])
    im.resize((1280, 720)).save(os.path.join(CHECK, f"{i:02d}_{split}_{stem[-20:]}.jpg"), quality=90)

print(f"\n확인용 이미지 {len(picked)}장 저장 → {CHECK}")
print("색: 빨강=vehicle, 주황=construction, 파랑=debris, 초록=micromobility")
