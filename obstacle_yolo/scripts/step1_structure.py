# step1_structure.py
# 목적: BBox 데이터의 "구조"를 파악한다.
#   (1) ZIP별 최상위 폴더 수, 확장자별 파일 수
#   (2) XML 하나의 태그 이름과 속성 "키" 목록  ← 값(좌표, 파일명, 라벨 내용)은 출력하지 않는다
#
# 데이터 규칙: 압축을 풀지 않고 ZIP 안을 직접 읽는다. 원본은 수정하지 않는다.
# 사용법: (obstacle_yolo 폴더에서) python scripts\step1_structure.py

import collections   # 개수 세기(Counter)에 쓰는 표준 라이브러리
import os
import zipfile        # ZIP 안의 파일 목록을 보거나 파일을 읽는 표준 라이브러리
import xml.etree.ElementTree as ET   # XML을 나무(tree) 구조로 읽는 표준 라이브러리

# BBox ZIP들이 있는 폴더 (원본 데이터 위치)
DATA_DIR = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"

# ------------------------------------------------------------
# (1) ZIP별 폴더 수와 확장자별 파일 수
# ------------------------------------------------------------
print("=== (1) ZIP 구조 ===")
first_xml = None   # 구조 확인에 쓸 XML 하나를 기억해 둔다

for zip_name in sorted(os.listdir(DATA_DIR)):
    if not zip_name.lower().endswith(".zip"):
        continue
    zip_path = os.path.join(DATA_DIR, zip_name)

    with zipfile.ZipFile(zip_path) as z:
        # infolist(): ZIP 안의 모든 항목(파일·폴더) 정보를 돌려준다. 실제 내용은 읽지 않는다.
        files = [info for info in z.infolist() if not info.is_dir()]

        # 경로의 첫 부분이 최상위 폴더 이름이다. 예) "Bbox_0691/xxx.jpg" → "Bbox_0691"
        top_folders = {info.filename.split("/")[0] for info in files}

        # 확장자별 개수. os.path.splitext("a/b.jpg") → ("a/b", ".jpg")
        ext_count = collections.Counter(os.path.splitext(info.filename)[1].lower() for info in files)

        print(f"{zip_name}: 최상위 폴더 {len(top_folders)}개, 확장자별 {dict(ext_count)}")

        if first_xml is None:
            xmls = [info.filename for info in files if info.filename.lower().endswith(".xml")]
            if xmls:
                first_xml = (zip_path, xmls[0])

# ------------------------------------------------------------
# (2) XML 구조: 태그 이름과 속성 키만
# ------------------------------------------------------------
print("\n=== (2) XML 구조 (태그 이름 → 속성 키, 등장 횟수) ===")
zip_path, xml_name = first_xml
with zipfile.ZipFile(zip_path) as z:
    root = ET.fromstring(z.read(xml_name))   # XML 전체를 읽어 나무 구조로 만든다 (화면에는 출력하지 않음)

# 부모 → 자식 관계를 "경로"로 표현한다. 예) annotations/image/box
tag_keys = collections.defaultdict(set)    # 경로별 속성 키 모음
tag_count = collections.Counter()          # 경로별 등장 횟수

def walk(element, path):
    """XML 요소를 하나씩 내려가며 태그 경로와 속성 '키'만 기록한다."""
    tag_count[path] += 1
    tag_keys[path].update(element.attrib.keys())   # .keys()만 쓴다 → 값은 보지 않는다
    for child in element:
        walk(child, path + "/" + child.tag)

walk(root, root.tag)

# meta 아래에는 라벨 목록 등 설정이 길게 들어 있어, 깊은 단계는 생략하고 보여준다
for path in sorted(tag_count, key=lambda p: (p.count("/"), p)):
    if path.count("/") > 3:
        continue
    keys = ", ".join(sorted(tag_keys[path])) or "(속성 없음)"
    print(f"{path:45s} 등장 {tag_count[path]:6d}회  속성 키: {keys}")
