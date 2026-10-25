# step2_stats.py
# 목적: BBox 전체 통계를 내고 outputs\obstacle\stats.csv 로 저장한다.
#   - 사진 수, 박스 없는 사진 수, 해상도 분포
#   - 클래스별: 박스 수 / 등장 사진 수 / 작은 박스(가로 또는 세로 20px 미만) 수
#               / 가려진 박스(occluded=1) 수 / 박스 너비·높이 중앙값(px)
#
# 데이터 규칙: ZIP을 풀지 않고 안의 XML만 읽는다. 개별 좌표는 출력하지 않고 "집계"만 출력·저장한다.
# 사용법: (obstacle_yolo 폴더에서) python scripts\step2_stats.py

import collections
import csv
import os
import statistics      # 중앙값(median) 계산
import zipfile
import xml.etree.ElementTree as ET

DATA_DIR = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"
OUT_DIR = r"C:\Furiosa\mini\RoadPush\obstacle_yolo\outputs"
SMALL_PX = 20          # 이 값보다 작은 너비나 높이를 가진 박스를 "작은 박스"로 본다

os.makedirs(OUT_DIR, exist_ok=True)   # 결과 폴더가 없으면 만든다

box_cnt = collections.Counter()        # 클래스별 박스 개수
img_cnt = collections.Counter()        # 클래스별 "그 클래스가 1개 이상 있는 사진" 수
small_cnt = collections.Counter()      # 클래스별 작은 박스 수
occ_cnt = collections.Counter()        # 클래스별 가려진 박스 수
widths = collections.defaultdict(list)    # 클래스별 박스 너비 목록 (중앙값 계산용, 화면에 출력하지 않음)
heights = collections.defaultdict(list)
sizes = collections.Counter()          # 사진 해상도 분포
n_img = n_empty = 0

for zip_name in sorted(os.listdir(DATA_DIR)):
    if not zip_name.lower().endswith(".zip"):
        continue
    with zipfile.ZipFile(os.path.join(DATA_DIR, zip_name)) as z:
        for xml_name in [n for n in z.namelist() if n.lower().endswith(".xml")]:
            root = ET.fromstring(z.read(xml_name))
            for img in root.iter("image"):          # 사진 한 장씩
                n_img += 1
                sizes[(img.get("width"), img.get("height"))] += 1
                boxes = img.findall("box")
                if not boxes:
                    n_empty += 1
                seen = set()                         # 이 사진에 나온 클래스 (사진 수 세기용)
                for b in boxes:                      # 박스 하나씩
                    label = b.get("label")
                    w = float(b.get("xbr")) - float(b.get("xtl"))   # 너비 = 오른쪽 x - 왼쪽 x
                    h = float(b.get("ybr")) - float(b.get("ytl"))   # 높이 = 아래쪽 y - 위쪽 y
                    box_cnt[label] += 1
                    seen.add(label)
                    widths[label].append(w)
                    heights[label].append(h)
                    if w < SMALL_PX or h < SMALL_PX:
                        small_cnt[label] += 1
                    if b.get("occluded") == "1":
                        occ_cnt[label] += 1
                img_cnt.update(seen)
    print(f"{zip_name} 처리 완료 (누적 사진 {n_img}장)")

# ---------- 화면 출력 (집계만) ----------
print(f"\n사진 {n_img}장 / 박스가 없는 사진 {n_empty}장")
print(f"해상도 분포: {dict(sizes.most_common(3))}")
total = sum(box_cnt.values())
print(f"박스 합계 {total}개, 클래스 {len(box_cnt)}개\n")

rows = []
for label, n in box_cnt.most_common():
    rows.append({
        "class": label,
        "boxes": n,
        "box_share_%": round(100 * n / total, 2),     # 전체 박스 중 비율 → 불균형 정도
        "images": img_cnt[label],
        "small_boxes": small_cnt[label],
        "small_%": round(100 * small_cnt[label] / n, 1),
        "occluded_boxes": occ_cnt[label],
        "occluded_%": round(100 * occ_cnt[label] / n, 1),
        "median_w_px": round(statistics.median(widths[label])),
        "median_h_px": round(statistics.median(heights[label])),
    })

print(f"{'class':25s}{'boxes':>9s}{'share%':>8s}{'images':>8s}{'small%':>8s}{'occl%':>7s}{'med_w':>7s}{'med_h':>7s}")
for r in rows:
    print(f"{r['class']:25s}{r['boxes']:9d}{r['box_share_%']:8.2f}{r['images']:8d}"
          f"{r['small_%']:8.1f}{r['occluded_%']:7.1f}{r['median_w_px']:7d}{r['median_h_px']:7d}")

# ---------- CSV 저장 ----------
out_path = os.path.join(OUT_DIR, "stats.csv")
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글이 안 깨짐
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n저장: {out_path}")
