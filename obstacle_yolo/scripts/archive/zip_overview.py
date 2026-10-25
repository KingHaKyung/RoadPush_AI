import sys, zipfile, collections, os
sys.stdout.reconfigure(encoding="utf-8")
D = r"C:\Furiosa\mini\RoadPush\data\인도보행 영상\BBox"
for f in sorted(os.listdir(D)):
    with zipfile.ZipFile(os.path.join(D, f)) as z:
        inf = [i for i in z.infolist() if not i.is_dir()]
        ext = collections.Counter(os.path.splitext(i.filename)[1].lower() for i in inf)
        xml = [i for i in inf if i.filename.lower().endswith(".xml")]
        bad = [i.filename for i in inf if i.filename.startswith("/") or ".." in i.filename.split("/")]
        tops = sorted({i.filename.split("/")[0] for i in inf})
        print(f"{f}: {dict(ext)}, XML 합계 {sum(i.file_size for i in xml)/1e6:.1f}MB, 최상위 {tops[:2]}..{tops[-1:]} ({len(tops)}개), 위험경로 {len(bad)}")
        if xml: print("   예:", xml[0].filename)
