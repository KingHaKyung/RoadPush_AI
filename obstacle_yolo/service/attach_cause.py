# attach_cause.py
# 목적: 팀 파이프라인(RoadPush_AI) Phase 6이 찾은 "길을 막는 영역(blocker 마스크)"에
#       우리 모델의 "무엇인가(클래스)"와 "누가 처리하나(처리 유형 A~D)"를 붙인다.
#
#   팀 파이프라인이 판단: 막는가? (Phase 6 blocker 마스크, 7-B 통행 가능 폭)
#   우리 모델이 판단    : 무엇인가? (탐지 박스 + 클래스)
#   → blocker 마스크를 덩어리(연결 영역)로 나누고, 덩어리와 겹치는 박스만 원인으로 인정한다.
#     길을 막지 않는 탐지는 무시하고, 맞는 박스가 없는 덩어리는 "unclassified"(미분류)로 남긴다.
#
# 판정 기준 (둘 중 하나라도 만족하면 그 박스를 덩어리의 원인으로 인정)
#   1) 덩어리 면적의 30% 이상이 박스 안에 있다  ← blocker는 물체의 "바닥 부분"만이라 박스보다 훨씬 작다
#   2) 박스 면적의 10% 이상이 덩어리 위에 있다  ← 큰 덩어리 안의 작은 물체
#
# 팀 파일과의 관계 (RoadPush_AI 저장소 규칙)
#   - 팀 코드는 import하지 않고, Phase 6 / 7-B 결과 "파일"만 읽는다 (수정·덮어쓰기 금지).
#   - 결과는 새 폴더에 저장하고, 입력·모델의 SHA-256을 기록한다.
#
# 좌표계: Phase 6은 EXIF 방향 보정 후 긴 변을 700px로 줄인 "분석 크기"에서 마스크를 만든다.
#         우리 박스도 같은 크기로 바꿔서 마스크와 픽셀 단위로 비교한다.
#
# 화면에는 집계만 출력한다. 확인용 이미지는 vis/ 에 저장 (사용자가 직접 연다).
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경)
#   python service\attach_cause.py
#   python service\attach_cause.py --phase6 <Phase 6 출력 폴더> --phase7b <7-B results.json> --out <새 폴더>

import argparse
import collections
import hashlib
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

from predict import load_config   # 같은 service 폴더의 predict.py

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_ROOT = r"C:\Furiosa\mini\RoadPush_AI"
# 팀 기본 입력 (40장 canonical 세트). 7-B 결과는 docs/PHASE7B_PRODUCTION_LOGIC.md 9절의 canonical 파일
PHASE6_DIR = r"outputs\surface_generalization\phase6_production_selector_canonical40_v1"
MANIFEST = r"outputs\walkway_detection_ablation_v2\model_audit_40\audit_manifest.json"
PHASE7B_RESULTS = (r"experiments\reviews\phase7b\final\phase7b\outputs\surface_generalization"
                   r"\phase7_b_full160_v2_provisional_final\results.json")

# 선택된 길(보도/차도)별 Phase 6 blocker 마스크 파일
BLOCKER_FILES = {"sidewalk": os.path.join("sidewalk", "08_applied_blocker.png"),
                 "road": os.path.join("road", "09_road_blocker.png")}

ANALYSIS_MAX = 700            # 팀 파이프라인의 분석 크기 (legacy_policy MAX_SIDE)
MIN_COMPONENT_PX = 30         # 이보다 작은 덩어리는 잡음으로 보고 버린다
MIN_COMPONENT_IN_BOX = 0.30   # 기준 1
MIN_BOX_ON_BLOCKER = 0.10     # 기준 2
BOX_MARGIN = 3                # 박스를 사방으로 3px 넓혀서 비교 (바닥 경계가 박스 바로 아래 걸치는 경우)
SCHEMA = "roadpush_cause_attribution_v1"
POLICY = "phase6_blocker_overlap_v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def team_path(root, p):
    # 팀 파일의 경로는 상대 경로면 팀 저장소 기준으로 해석한다
    return p if os.path.isabs(p) else os.path.join(root, p)


def read_mask(path):
    # 한글 경로도 읽히도록 fromfile + imdecode 사용
    data = np.fromfile(path, dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if m is None:
        raise FileNotFoundError(path)
    return m > 0


def detect(model, path, cls_cfg, imgsz, device, min_conf):
    # 팀 파이프라인과 같은 방식: EXIF 방향 보정 → 박스를 분석 크기 좌표로 변환
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    W, H = img.size
    scale = min(1.0, ANALYSIS_MAX / max(W, H))
    size = (max(1, round(W * scale)), max(1, round(H * scale)))   # (가로, 세로)
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    r = model.predict(bgr, imgsz=imgsz, conf=min_conf, device=device, verbose=False)[0]
    dets = []
    for c, s, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        name = model.names[int(c)]
        info = cls_cfg[name]
        if s < info["conf"]:
            continue
        dets.append({"class": name, "name_ko": info["name_ko"], "type": info["type"],
                     "action": info["action"], "confidence": round(s, 3),
                     "box_analysis": [round(size[0] / W * xyxy[0], 1), round(size[1] / H * xyxy[1], 1),
                                      round(size[0] / W * xyxy[2], 1), round(size[1] / H * xyxy[3], 1)]})
    return dets, size, bgr


def attribute(blocker, dets):
    # blocker 마스크를 덩어리로 나누고, 덩어리마다 겹치는 박스를 원인으로 붙인다
    h, w = blocker.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blocker.astype(np.uint8), connectivity=8)
    comps, used = [], set()
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_PX:
            continue
        comp = labels == lab
        covered = np.zeros_like(comp)
        causes = []
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = d["box_analysis"]
            x0, y0 = max(0, int(x0) - BOX_MARGIN), max(0, int(y0) - BOX_MARGIN)
            x1, y1 = min(w, int(np.ceil(x1)) + BOX_MARGIN), min(h, int(np.ceil(y1)) + BOX_MARGIN)
            box_area = max(0, x1 - x0) * max(0, y1 - y0)
            if box_area == 0:
                continue
            inside = int(comp[y0:y1, x0:x1].sum())          # 덩어리 중 박스 안에 들어온 픽셀 수
            comp_share, box_share = inside / area, inside / box_area
            if comp_share < MIN_COMPONENT_IN_BOX and box_share < MIN_BOX_ON_BLOCKER:
                continue
            used.add(i)
            covered[y0:y1, x0:x1] |= comp[y0:y1, x0:x1]
            causes.append({**{k: d[k] for k in ("class", "name_ko", "type", "action", "confidence")},
                           "detection_index": i, "overlap_px": inside,
                           "component_in_box": round(comp_share, 3), "box_on_blocker": round(box_share, 3)})
        causes.sort(key=lambda c: (-c["overlap_px"], -c["confidence"]))
        comps.append({"component_id": len(comps), "area_px": area,
                      "bbox": [int(stats[lab, 0]), int(stats[lab, 1]),
                               int(stats[lab, 0] + stats[lab, 2]), int(stats[lab, 1] + stats[lab, 3])],
                      "primary_cause": causes[0]["class"] if causes else "unclassified",
                      "causes": causes, "unexplained_px": area - int(covered.sum()),
                      "_label": lab})
    n_cls = sum(1 for c in comps if c["causes"])
    if not comps:
        status = "no_blocker"
    elif n_cls == len(comps):
        status = "attributed"
    else:
        status = "partially_attributed" if n_cls else "unclassified_only"
    return status, comps, labels, used


def draw(bgr, size, labels, comps, dets, used):
    # 확인용 이미지: 원인 붙은 blocker = 빨강, 미분류 blocker = 노랑, 원인 박스 = 빨강, 길 안 막는 박스 = 회색
    vis = cv2.resize(bgr, size)
    for c in comps:
        color = np.array((0, 0, 255) if c["causes"] else (0, 255, 255), dtype=np.float32)
        m = labels == c["_label"]
        vis[m] = (vis[m] * 0.4 + color * 0.6).astype(np.uint8)
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = map(int, d["box_analysis"])
        col = (0, 0, 255) if i in used else (160, 160, 160)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, f'{d["type"]} {d["class"]} {d["confidence"]:.2f}', (x0, max(10, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-root", default=TEAM_ROOT, help="RoadPush_AI 폴더 (상대 경로 해석 기준)")
    ap.add_argument("--phase6", default=PHASE6_DIR, help="Phase 6 출력 폴더 (results.json, artifacts/)")
    ap.add_argument("--manifest", default=MANIFEST, help="사진 경로 목록 (Phase 6을 돌릴 때 쓴 것)")
    ap.add_argument("--phase7b", default=PHASE7B_RESULTS, help="7-B results.json (없으면 'none')")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "cause_" + time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} 이미 있음 — 덮어쓰지 않습니다 (새 이름으로 실행하세요)")

    root = args.team_root
    phase6 = team_path(root, args.phase6)
    manifest_path = team_path(root, args.manifest)
    p6_results = os.path.join(phase6, "results.json")
    with open(p6_results, encoding="utf-8") as f:
        p6_rows = json.load(f)
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    images = {r["sample_id"]: team_path(root, r["relative_path"])
              for r in (man["samples"] if isinstance(man, dict) else man) if "relative_path" in r}
    p7b_path = None if args.phase7b == "none" else team_path(root, args.phase7b)
    p7b = {}
    if p7b_path:
        with open(p7b_path, encoding="utf-8") as f:
            p7b = {r["sample_id"]: r for r in json.load(f)}

    cfg = load_config()                       # predict.py와 같은 설정 (모델 경로·임계값·처리 유형)
    cls_cfg = cfg["classes"]
    model = YOLO(cfg["model"]["path"])
    device = 0 if torch.cuda.is_available() else "cpu"
    min_conf = min(c["conf"] for c in cls_cfg.values())
    os.makedirs(os.path.join(args.out, "vis"))

    provenance = {"schema_version": SCHEMA, "policy": POLICY,
                  "model_version": cfg["model"]["version"], "model_sha256": sha256_file(cfg["model"]["path"]),
                  "phase6_dir": phase6, "phase6_results_sha256": sha256_file(p6_results),
                  "manifest_sha256": sha256_file(manifest_path),
                  "phase7b_results": p7b_path, "phase7b_results_sha256": sha256_file(p7b_path) if p7b_path else None,
                  "params": {"analysis_max_side": ANALYSIS_MAX, "min_component_px": MIN_COMPONENT_PX,
                             "min_component_in_box": MIN_COMPONENT_IN_BOX,
                             "min_box_on_blocker": MIN_BOX_ON_BLOCKER, "box_margin": BOX_MARGIN}}

    status_count, cause_count, auth_count = collections.Counter(), collections.Counter(), collections.Counter()
    out_rows = []
    for k, row in enumerate(p6_rows, 1):
        sid = row["sample_id"]
        surface = row["surface_selection"]["selected_surface"]
        p = p7b.get(sid)
        rec = {"sample_id": sid, "selected_surface": surface,
               "phase7b": {k2: p.get(k2) for k2 in ("passability_state", "minimum_passable_ratio",
                                                   "result_authority", "phase7a_status")} if p else None}
        if surface is None:
            status, comps, dets = "no_selected_surface", [], []    # Phase 6이 보도/차도를 정하지 못함
        else:
            mask_path = os.path.join(phase6, "artifacts", sid, BLOCKER_FILES[surface])
            blocker = read_mask(mask_path)
            dets, size, bgr = detect(model, images[sid], cls_cfg, cfg["model"]["imgsz"], device, min_conf)
            if blocker.shape != (size[1], size[0]):
                raise SystemExit(f"{sid}: 마스크 크기 {blocker.shape}와 분석 크기 {size[::-1]}가 다름")
            status, comps, labels, used = attribute(blocker, dets)
            vis = draw(bgr, size, labels, comps, dets, used)
            cv2.imencode(".jpg", vis)[1].tofile(os.path.join(args.out, "vis", sid + ".jpg"))
            rec["inputs"] = {"image_sha256": sha256_file(images[sid]),
                             "phase6_blocker": os.path.relpath(mask_path, phase6),
                             "phase6_blocker_sha256": sha256_file(mask_path)}
            for c in comps:
                c.pop("_label")
                cause_count[c["primary_cause"]] += 1
        types = sorted({c["type"] for comp in comps for c in comp["causes"]})
        # 자동 처리(민원 전달)는 원인이 있고 7-B 결과가 authoritative일 때만
        rec.update({"cause_status": status, "types": types,
                    "auto_routable": bool(types) and bool(p) and p.get("result_authority") == "authoritative",
                    "components": comps, "detections": dets})
        status_count[status] += 1
        auth_count[(status, (p or {}).get("result_authority"))] += 1
        out_rows.append(rec)
        print(f"[{k:02d}/{len(p6_rows):02d}] {sid}", flush=True)

    with open(os.path.join(args.out, "cause_results.json"), "w", encoding="utf-8") as f:
        json.dump({"provenance": provenance, "results": out_rows}, f, ensure_ascii=False, indent=2)

    # 집계 출력
    print(f"\n사진 {len(out_rows)}장")
    print("판정 상태:", dict(status_count))
    print("blocker 덩어리별 주 원인:", dict(cause_count))
    print("(판정 상태, 7-B 신뢰도):", {f"{a}/{b}": v for (a, b), v in sorted(auth_count.items(), key=str)})
    print("자동 처리 가능(원인 있음 + 7-B authoritative):", sum(r["auto_routable"] for r in out_rows), "건")
    print(f"\n결과: {args.out}")


if __name__ == "__main__":
    main()
