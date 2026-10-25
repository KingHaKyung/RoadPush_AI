# diag_source_labels.py
# 목적: 확인용 이미지(outputs/check)에 그려진 박스가 원본 AI Hub에서 어떤 라벨 이름이었는지
#       "라벨 이름과 개수만" 출력한다. (좌표·파일 내용은 출력하지 않음)
# 사용법: (obstacle_yolo 폴더에서) python scripts\diag_source_labels.py 07 10 11 12 14 15 18
import collections, csv, os, sys, zipfile
import xml.etree.ElementTree as ET
import yaml

BASE = r"C:\Furiosa\mini\RoadPush\obstacle_yolo"
DATA_DIR = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"
DATASET = "dataset_v2_10k"                                        # 확인할 데이터셋 (step5와 같게)
CHECK = os.path.join(BASE, "outputs", "check" if DATASET == "dataset" else f"check_{DATASET}")

with open(os.path.join(BASE, "configs", "class_map.yaml"), encoding="utf-8") as f:
    cmap = yaml.safe_load(f)
raw_to_cls = {raw: key.split("_", 1)[1] for key, raws in cmap["classes"].items() for raw in raws}

with open(os.path.join(BASE, "outputs", DATASET, "manifest.csv"), encoding="utf-8-sig") as f:
    manifest = {os.path.splitext(r["file"])[0]: r for r in csv.DictReader(f)}

wanted = set(sys.argv[1:])
for name in sorted(os.listdir(CHECK)):
    if wanted and name[:2] not in wanted:
        continue
    tail = os.path.splitext(name)[0].split("_", 2)[2]           # 저장 시 stem 끝 20자를 붙였음
    stem = next(s for s in manifest if s.endswith(tail))
    r = manifest[stem]
    with zipfile.ZipFile(os.path.join(DATA_DIR, r["zip"])) as z:
        xml_name = next(n for n in z.namelist() if n.startswith(r["folder"] + "/") and n.lower().endswith(".xml"))
        img_name = r["zip_path"].rsplit("/", 1)[1]
        img = next(i for i in ET.fromstring(z.read(xml_name)).iter("image") if i.get("name") == img_name)
        kept = collections.Counter(); dropped = collections.Counter()
        for b in img.findall("box"):
            lb = b.get("label")
            (kept if lb in raw_to_cls else dropped)[f"{lb}→{raw_to_cls.get(lb, '제외')}"] += 1
    print(f"{name[:2]}: 학습 라벨 {dict(kept)} | 제외된 원본 라벨 {dict(dropped)}")
