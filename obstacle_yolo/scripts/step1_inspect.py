# step1_inspect.py
# 사용법: python step1_inspect.py "C:\Furiosa\mini\RoadPush\bbox"
import sys, glob, collections
import xml.etree.ElementTree as ET

root = sys.argv[1]
xml_files = glob.glob(root + r"\**\*.xml", recursive=True)
print(f"XML 파일 {len(xml_files)}개\n")

box_cnt = collections.Counter()    # 클래스별 박스 개수
img_cnt = collections.Counter()    # 클래스별 '그 클래스가 1개 이상 있는 사진' 수
sizes = collections.Counter()      # 사진 해상도 분포
n_img, n_empty = 0, 0
small = collections.Counter()      # 아주 작은 박스(가로 or 세로 20px 미만) 개수

for xf in xml_files:
    for img in ET.parse(xf).getroot().iter("image"):
        n_img += 1
        sizes[(img.get("width"), img.get("height"))] += 1
        seen = set()
        boxes = img.findall("box")
        if not boxes:
            n_empty += 1
        for b in boxes:
            lb = b.get("label")
            box_cnt[lb] += 1
            seen.add(lb)
            w = float(b.get("xbr")) - float(b.get("xtl"))
            h = float(b.get("ybr")) - float(b.get("ytl"))
            if w < 20 or h < 20:
                small[lb] += 1
        img_cnt.update(seen)

print(f"사진 {n_img}장 / 박스가 하나도 없는 사진 {n_empty}장")
print(f"해상도: {dict(sizes.most_common(3))}\n")
print(f"{'클래스':20s} {'박스 수':>8s} {'등장 사진 수':>10s} {'작은 박스':>8s}")
for lb, n in box_cnt.most_common():
    print(f"{lb:20s} {n:8d} {img_cnt[lb]:10d} {small[lb]:8d}")