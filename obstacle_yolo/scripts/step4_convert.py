# step4_convert.py
# 목적: AI Hub BBox 라벨(XML)을 YOLO 형식으로 바꾸고, 폴더 단위로 train/val을 나눠
#       클래스가 골고루 들어간 약 2,000장 샘플 데이터셋을 만든다.
#
# 출력: obstacle_yolo/outputs/<DATASET>/   (DATASET은 아래 설정값, 현재 dataset_v2)
#         images/train, images/val   ← 샘플 사진 (ZIP에서 해당 사진만 꺼냄)
#         labels/train, labels/val   ← 사진마다 같은 이름의 .txt 라벨
#         data.yaml                  ← YOLO 학습 설정 (경로, 클래스 이름)
#         manifest.csv               ← 사진별 원본 위치·폴더·분할 기록 (나중에 추적용)
#
# 데이터 규칙: 이 PC 안에서만 처리한다. 화면에는 개수(집계)만 출력한다.
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경) python scripts\step4_convert.py

import collections
import csv
import io
import os
import random
import zipfile
import xml.etree.ElementTree as ET

import yaml
from PIL import Image

DATA_DIR = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"
BASE = r"C:\Furiosa\mini\RoadPush\obstacle_yolo"
DATASET = "dataset_v2_full"  # v1(dataset): potted_plant→debris / v2: potted_plant 있는 사진 제외 / _10k: 1만 장 / _full: train 전부
OUT = os.path.join(BASE, "outputs", DATASET)

SEED = 20261023          # 난수 시드 (같은 값이면 같은 분할·같은 샘플이 나온다)
VAL_RATIO = 0.2          # 폴더의 20%를 val로
N_TRAIN, N_VAL = None, 2000  # 샘플 장수. None = train 폴더의 장애물 사진 전부 (dataset_v2_10k는 8000, 2000)
BG_RATIO = 0.05          # 장애물이 하나도 없는 배경 사진 비율
VAL_FROM = "dataset_v2_10k"  # val을 이 데이터셋과 "똑같은 사진"으로 쓴다 → 모델끼리 같은 시험지로 비교. None이면 새로 뽑음
TRAIN_RESIZE_W = 1280    # train 사진을 이 너비로 줄여 저장 (1920→1280, 용량 약 1/3). None이면 원본 그대로
                         # 학습은 어차피 640으로 줄여서 하므로 결과에 거의 영향 없음. val은 비교를 위해 원본 그대로 둔다

# 이미 만든 데이터셋 폴더가 있으면 멈춘다 (두 번 실행하면 이전 사진 위에 새 사진이 쌓이기 때문)
if os.path.exists(OUT) and os.listdir(OUT):
    raise SystemExit(f"{OUT} 폴더가 이미 있습니다. 다시 만들려면 이 폴더를 먼저 지우거나 DATASET 이름을 바꾸세요.")

# ------------------------------------------------------------
# 1) 매핑 읽기: 원본 라벨 이름 → YOLO 클래스 번호
# ------------------------------------------------------------
with open(os.path.join(BASE, "configs", "class_map.yaml"), encoding="utf-8") as f:
    cmap = yaml.safe_load(f)
assert cmap["status"] == "confirmed", "class_map.yaml이 확정(confirmed) 상태가 아닙니다"

names = []               # YOLO 클래스 이름 (번호 순서대로)
label_to_id = {}         # 예) "car" → 0
for key in sorted(cmap["classes"], key=lambda k: int(k.split("_")[0])):   # "0_vehicle" → 0 순서 정렬
    cid, cname = key.split("_", 1)
    names.append(cname)
    for raw in cmap["classes"][key]:
        label_to_id[raw] = int(cid)
print("YOLO 클래스:", dict(enumerate(names)))
drop_labels = set(cmap.get("drop_images_with", []))   # 이 라벨이 있는 사진은 통째로 뺀다
print("사진째 제외할 라벨:", sorted(drop_labels) or "없음")
n_dropped = 0

# ------------------------------------------------------------
# 2) 모든 XML을 읽어 사진별 라벨을 만든다 (아직 파일로 쓰지 않음)
# ------------------------------------------------------------
images = []              # 사진 하나당 dict 하나
for zip_name in sorted(os.listdir(DATA_DIR)):
    if not zip_name.lower().endswith(".zip"):
        continue
    zip_path = os.path.join(DATA_DIR, zip_name)
    with zipfile.ZipFile(zip_path) as z:
        all_names = set(z.namelist())
        # sorted()가 꼭 필요하다: set은 실행할 때마다 순서가 바뀌어서, 정렬하지 않으면
        # 같은 시드여도 매번 다른 사진이 뽑힌다 (재현성 깨짐)
        for xml_name in sorted(n for n in all_names if n.lower().endswith(".xml")):
            folder = xml_name.rsplit("/", 1)[0]               # 촬영 묶음 폴더 (분할 단위)
            for img in ET.fromstring(z.read(xml_name)).iter("image"):
                jpg = f"{folder}/{img.get('name')}"
                if jpg not in all_names:
                    continue
                if any(b.get("label") in drop_labels for b in img.findall("box")):
                    n_dropped += 1                            # 예) 화분이 있는 사진 → 학습 후보에서 제외
                    continue
                W, H = float(img.get("width")), float(img.get("height"))
                lines, classes = [], set()
                for b in img.findall("box"):
                    cid = label_to_id.get(b.get("label"))
                    if cid is None:                           # exclude 클래스 → 라벨에서 뺀다(배경)
                        continue
                    # 사진 밖으로 나간 좌표를 사진 안으로 자른다(clip)
                    x1 = min(max(float(b.get("xtl")), 0), W); x2 = min(max(float(b.get("xbr")), 0), W)
                    y1 = min(max(float(b.get("ytl")), 0), H); y2 = min(max(float(b.get("ybr")), 0), H)
                    if x2 - x1 < 1 or y2 - y1 < 1:            # 너비·높이가 1px 미만인 박스는 버린다
                        continue
                    # 꼭짓점(픽셀) → 중심·크기(0~1 비율)
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    bw, bh = (x2 - x1) / W, (y2 - y1) / H
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    classes.add(cid)
                lines = list(dict.fromkeys(lines))            # 원본에 똑같은 박스가 두 번 있으면 하나만 남긴다
                images.append({"zip": zip_path, "jpg": jpg, "folder": f"{zip_name}|{folder}",
                               "lines": lines, "classes": classes})
print(f"학습 후보 사진 {len(images)}장 (제외 라벨 때문에 뺀 사진 {n_dropped}장)")

# ------------------------------------------------------------
# 3) 폴더 단위 train/val 분할 → 같은 촬영 묶음이 양쪽에 섞이지 않는다
# ------------------------------------------------------------
rng = random.Random(SEED)
folders = sorted({im["folder"] for im in images})
rng.shuffle(folders)
n_val_folders = round(len(folders) * VAL_RATIO)
val_folders = set(folders[:n_val_folders])
for im in images:
    im["split"] = "val" if im["folder"] in val_folders else "train"
print(f"폴더 {len(folders)}개 → train {len(folders) - n_val_folders}개 / val {n_val_folders}개")

# ------------------------------------------------------------
# 4) 클래스 균형 샘플링: 드문 클래스부터 할당량을 채운다
# ------------------------------------------------------------
def sample(pool, n_total):
    rng.shuffle(pool)
    n_bg = round(n_total * BG_RATIO)
    quota = (n_total - n_bg) // len(names)                    # 클래스마다 목표 장수
    rarity = collections.Counter(c for im in pool for c in im["classes"])
    order = sorted(range(len(names)), key=lambda c: rarity[c])  # 드문 클래스 먼저
    chosen, chosen_ids, have = [], set(), collections.Counter()
    for c in order:
        for im in pool:
            if have[c] >= quota:
                break
            if c in im["classes"] and id(im) not in chosen_ids:
                chosen.append(im); chosen_ids.add(id(im))
                have.update(im["classes"])                     # 한 사진에 여러 클래스가 있으면 모두 센다
    bg = [im for im in pool if not im["classes"]][:n_bg]        # 배경 사진
    chosen += bg; chosen_ids.update(id(im) for im in bg)
    rest = [im for im in pool if id(im) not in chosen_ids and im["classes"]]
    chosen += rest[: max(0, n_total - len(chosen))]             # 모자라면 아무 장애물 사진으로 채운다
    return chosen[:n_total]

def take_all(pool):
    # 장애물 사진은 전부 쓰고, 배경 사진은 전체의 BG_RATIO만큼만 섞는다
    obj = [im for im in pool if im["classes"]]
    bg = [im for im in pool if not im["classes"]]
    rng.shuffle(bg)
    n_bg = round(len(obj) * BG_RATIO / (1 - BG_RATIO))
    return obj + bg[:n_bg]

train_pool = [im for im in images if im["split"] == "train"]
val_pool = [im for im in images if im["split"] == "val"]
train = take_all(train_pool) if N_TRAIN is None else sample(train_pool, N_TRAIN)

if VAL_FROM:
    # 기존 데이터셋의 manifest에 적힌 val 사진(ZIP 이름 + ZIP 안 경로)과 똑같은 사진을 고른다
    with open(os.path.join(BASE, "outputs", VAL_FROM, "manifest.csv"), encoding="utf-8-sig") as f:
        keys = {(r["zip"], r["zip_path"]) for r in csv.DictReader(f) if r["split"] == "val"}
    val = [im for im in val_pool if (os.path.basename(im["zip"]), im["jpg"]) in keys]
    assert len(val) == len(keys), f"{VAL_FROM}의 val {len(keys)}장 중 {len(val)}장만 찾음"
    print(f"val: {VAL_FROM}와 같은 사진 {len(val)}장 사용")
else:
    val = sample(val_pool, N_VAL)

# ------------------------------------------------------------
# 5) 파일 쓰기: 사진은 ZIP에서 꺼내고, 라벨은 .txt로 저장
# ------------------------------------------------------------
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(OUT, *sub.split("/")), exist_ok=True)

manifest = []
zips = {}
for split, items in (("train", train), ("val", val)):
    for im in items:
        z = zips.setdefault(im["zip"], zipfile.ZipFile(im["zip"]))
        stem = im["jpg"].replace("/", "__")[:-4]               # 폴더명__파일명 → 이름 겹침 방지
        dst = os.path.join(OUT, "images", split, stem + ".jpg")
        if split == "train" and TRAIN_RESIZE_W:
            # 사진 크기를 줄여 저장. 라벨은 0~1 비율 좌표라서 그대로 써도 맞는다
            pic = Image.open(io.BytesIO(z.read(im["jpg"])))
            pic.resize((TRAIN_RESIZE_W, round(pic.height * TRAIN_RESIZE_W / pic.width)), Image.BILINEAR).save(dst, quality=95)
        else:
            with open(dst, "wb") as f:
                f.write(z.read(im["jpg"]))
        with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(im["lines"]) + ("\n" if im["lines"] else ""))   # 배경 사진은 빈 파일
        manifest.append({"split": split, "file": stem + ".jpg", "zip": os.path.basename(im["zip"]),
                         "zip_path": im["jpg"], "folder": im["folder"].split("|")[1],
                         "boxes": len(im["lines"]), "classes": ";".join(names[c] for c in sorted(im["classes"]))})

with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)

# YOLO 학습 설정 파일
with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump({"path": OUT, "train": "images/train", "val": "images/val",
                    "names": dict(enumerate(names))}, f, allow_unicode=True, sort_keys=False)

# ------------------------------------------------------------
# 6) 집계 출력
# ------------------------------------------------------------
for split, items in (("train", train), ("val", val)):
    boxes = collections.Counter(int(l.split()[0]) for im in items for l in im["lines"])
    imgs = collections.Counter(c for im in items for c in im["classes"])
    n_bg = sum(1 for im in items if not im["classes"])
    print(f"\n[{split}] 사진 {len(items)}장 (배경 {n_bg}장, 클래스당 목표 {(len(items) - n_bg) // len(names)}장)")
    for c, n in enumerate(names):
        print(f"  {c} {n:15s} 박스 {boxes[c]:6d}  등장 사진 {imgs[c]:5d}")
print(f"\n저장: {OUT}")
