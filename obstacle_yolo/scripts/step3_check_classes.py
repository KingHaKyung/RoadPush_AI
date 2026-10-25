# step3_check_classes.py
# 목적: class_map.yaml의 check_before_training에 적힌 클래스(scooter, carrier, chair, potted_plant)마다
#       무작위로 박스 10개를 골라, 사진에 박스를 그린 확인용 이미지를 저장한다.
#       → 사람이 직접 열어 보고 "그 클래스가 실제로 무엇인지" 판단하기 위한 자료
#
# 데이터 규칙: 이 PC 안에서만 처리한다. 화면에는 저장 경로와 개수만 출력한다(좌표·파일 내용은 출력하지 않음).
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경) python scripts\step3_check_classes.py

import io
import os
import random
import zipfile
import xml.etree.ElementTree as ET

import yaml                      # ultralytics 설치 시 함께 설치됨 (yaml 파일 읽기)
from PIL import Image, ImageDraw  # ultralytics 설치 시 함께 설치됨 (그림 그리기)

DATA_DIR = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"
BASE = r"C:\Furiosa\mini\RoadPush\obstacle_yolo"
OUT_DIR = os.path.join(BASE, "outputs", "check_classes")
PER_CLASS = 10          # 클래스마다 몇 장 뽑을지
SEED = 20261022         # 난수 시드: 같은 값이면 매번 같은 사진이 뽑힌다(재현 가능)

# 1) 확인할 클래스 목록을 매핑 파일에서 읽는다
with open(os.path.join(BASE, "configs", "class_map.yaml"), encoding="utf-8") as f:
    targets = list(yaml.safe_load(f)["check_before_training"].keys())
print("확인할 클래스:", targets)

# 2) 모든 XML을 훑어 "대상 클래스 박스가 있는 사진" 후보를 모은다
#    후보 = (zip 경로, 사진 경로, 박스 좌표) — 좌표는 그림 그리기에만 쓰고 출력하지 않는다
candidates = {c: [] for c in targets}
for zip_name in sorted(os.listdir(DATA_DIR)):
    if not zip_name.lower().endswith(".zip"):
        continue
    zip_path = os.path.join(DATA_DIR, zip_name)
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for xml_name in [n for n in names if n.lower().endswith(".xml")]:
            folder = xml_name.rsplit("/", 1)[0]          # XML이 들어 있는 촬영 묶음 폴더
            for img in ET.fromstring(z.read(xml_name)).iter("image"):
                jpg = f"{folder}/{img.get('name')}"
                if jpg not in names:                      # 사진 경로가 다르게 저장된 경우 대비
                    jpg = next((n for n in names if n.endswith("/" + img.get("name"))), None)
                    if jpg is None:
                        continue
                for b in img.findall("box"):
                    if b.get("label") in candidates:
                        box = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
                        candidates[b.get("label")].append((zip_path, jpg, box))

# 3) 클래스마다 무작위로 PER_CLASS개를 골라 박스를 그려 저장한다
rng = random.Random(SEED)
for cls, items in candidates.items():
    picked = rng.sample(items, min(PER_CLASS, len(items)))
    cls_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for i, (zip_path, jpg, (x1, y1, x2, y2)) in enumerate(picked, 1):
        with zipfile.ZipFile(zip_path) as z:
            im = Image.open(io.BytesIO(z.read(jpg))).convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)   # 대상 박스를 빨간 테두리로
        # 보기 편하게: 박스 주변을 넉넉히 잘라낸 그림도 함께 저장한다
        pad = 150
        crop = im.crop((max(0, x1 - pad), max(0, y1 - pad), min(im.width, x2 + pad), min(im.height, y2 + pad)))
        im.resize((960, 540)).save(os.path.join(cls_dir, f"{i:02d}_full.jpg"), quality=90)
        crop.save(os.path.join(cls_dir, f"{i:02d}_crop.jpg"), quality=90)
    print(f"{cls}: 후보 박스 {len(items)}개 중 {len(picked)}개 저장 → {cls_dir}")

print(f"\n완료. 이 폴더를 직접 열어 확인하세요: {OUT_DIR}")
