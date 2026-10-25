import sys, zipfile, os
sys.stdout.reconfigure(encoding="utf-8")
D = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"
OUT = r"C:\Furiosa\mini\RoadPush\outputs\inspect_test\bbox_xml"
n = 0
for f in sorted(os.listdir(D)):
    with zipfile.ZipFile(os.path.join(D, f)) as z:
        for i in z.infolist():
            if i.filename.lower().endswith(".xml"):
                dst = os.path.join(OUT, os.path.splitext(f)[0], *i.filename.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(i) as s, open(dst, "wb") as d: d.write(s.read())
                n += 1
print(f"XML {n}개 → {OUT}")
