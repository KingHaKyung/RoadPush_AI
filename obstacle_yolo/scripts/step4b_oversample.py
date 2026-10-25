# step4b_oversample.py
# 목적: 사진 파일을 복사하지 않고 "train 목록"에서 construction(바리케이드) 사진만 여러 번 반복해
#       클래스 불균형을 줄인 데이터셋 설정을 만든다. (실험 ⑥)
#
# 원리: YOLO는 train에 폴더 대신 "사진 경로 목록(.txt)"을 받을 수 있다.
#       목록에 같은 사진을 4번 적으면 한 epoch에 그 사진을 4번 본다 (매번 증강이 달라 완전히 같은 장면은 아님).
#       라벨은 사진 경로의 images → labels 로 자동으로 찾으므로 원래 dataset_v2_full의 라벨을 그대로 쓴다.
#
# 출력: obstacle_yolo/outputs/dataset_v2_full_os4/
#         train.txt   ← train 사진 경로 목록 (construction 사진은 REPEAT번)
#         data.yaml   ← train=train.txt, val=dataset_v2_full의 val(= dataset_v2_10k와 같은 2,000장)
# 화면에는 개수만 출력한다.
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경) python scripts\step4b_oversample.py

import collections
import os

import yaml

BASE = r"C:\Furiosa\mini\RoadPush\obstacle_yolo"
SRC = os.path.join(BASE, "outputs", "dataset_v2_full")     # 사진·라벨이 실제로 있는 곳
REPEAT = 4                                                  # construction 사진 반복 횟수
TARGET_CLASS = 1                                            # 1 = construction
OUT = os.path.join(BASE, "outputs", f"dataset_v2_full_os{REPEAT}")

if os.path.exists(OUT) and os.listdir(OUT):
    raise SystemExit(f"{OUT} 폴더가 이미 있습니다. 다시 만들려면 먼저 지우세요.")
os.makedirs(OUT, exist_ok=True)

img_dir = os.path.join(SRC, "images", "train")
lbl_dir = os.path.join(SRC, "labels", "train")
with open(os.path.join(SRC, "data.yaml"), encoding="utf-8") as f:
    names = yaml.safe_load(f)["names"]

lines, before, after = [], collections.Counter(), collections.Counter()
n_repeated = 0
for fname in sorted(os.listdir(img_dir)):
    stem = os.path.splitext(fname)[0]
    with open(os.path.join(lbl_dir, stem + ".txt"), encoding="utf-8") as f:
        classes = {int(l.split()[0]) for l in f if l.strip()}
    times = REPEAT if TARGET_CLASS in classes else 1
    n_repeated += times > 1
    lines += [os.path.join(img_dir, fname)] * times          # 절대 경로로 적는다
    for c in classes:
        before[c] += 1
        after[c] += times

with open(os.path.join(OUT, "train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump({"path": OUT, "train": "train.txt",
                    "val": os.path.join(SRC, "images", "val"),
                    "names": names}, f, allow_unicode=True, sort_keys=False)

print(f"construction이 있는 사진 {n_repeated}장을 {REPEAT}번씩 → train 목록 {len(lines)}줄 (원래 {len(os.listdir(img_dir))}장)")
for c, n in names.items():
    print(f"  {c} {n:15s} 등장 사진 {before[c]:6d} → {after[c]:6d}  (vehicle 대비 1/{after[0] / max(1, after[c]):.0f})")
print(f"저장: {OUT}")
