# predict.py
# 목적: 학습한 장애물 탐지 모델로 "새 사진"을 판정해서, 서비스에서 쓸 수 있는 JSON과 확인용 이미지를 만든다.
#
#   입력: 사진 1장 또는 사진이 든 폴더
#   출력: <out>/json/<사진이름>.json   ← 장애물 목록 (클래스, 처리 유형, 신뢰도, 박스 위치)
#         <out>/vis/<사진이름>.jpg     ← 박스를 그린 확인용 이미지
#         <out>/results.jsonl          ← 모든 사진 결과를 한 줄씩 모은 파일 (다른 프로그램이 읽기 쉬움)
#
# 화면에는 집계(사진 수, 클래스별 개수, 속도)만 출력한다.
# 사용법: (obstacle_yolo 폴더에서, roadpush-yolo 환경)
#   python service\predict.py 사진.jpg
#   python service\predict.py 사진폴더 --out service\outputs\내테스트

import argparse
import collections
import json
import os
import time

import cv2
import torch
import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 모델 경로가 상대 경로면 obstacle_yolo 폴더(service의 상위) 기준으로 바꾼다
    path = cfg["model"]["path"]
    if not os.path.isabs(path):
        cfg["model"]["path"] = os.path.normpath(os.path.join(HERE, "..", path))
    if not os.path.exists(cfg["model"]["path"]):
        raise SystemExit(f"모델 파일이 없습니다: {cfg['model']['path']} (Drive에서 받아 weights/ 에 넣으세요)")
    return cfg


def save_jpg(path, img):
    # cv2.imwrite는 Windows에서 한글 경로를 못 쓰므로, 메모리에서 jpg로 만든 뒤 파일로 저장한다
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="사진 파일 또는 폴더")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", time.strftime("%Y%m%d_%H%M%S")),
                    help="결과 폴더 (기본: service/outputs/날짜_시각)")
    ap.add_argument("--no-vis", action="store_true", help="확인용 이미지를 만들지 않음 (빠름)")
    args = ap.parse_args()

    cfg = load_config()
    cls_cfg = cfg["classes"]
    os.makedirs(os.path.join(args.out, "json"), exist_ok=True)
    if not args.no_vis:
        os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)

    model = YOLO(cfg["model"]["path"])
    names = model.names                                   # {0: 'vehicle', 1: 'construction', ...}
    min_conf = min(c["conf"] for c in cls_cfg.values())   # 모델에는 가장 낮은 임계값을 주고, 클래스별로 다시 거른다
    device = 0 if torch.cuda.is_available() else "cpu"    # GPU가 없는 서버에서도 돌아가게

    counts = collections.Counter()   # 클래스별 탐지 개수
    n_img, n_with_obstacle, t_total = 0, 0, 0.0
    jsonl = open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8")

    # stream=True: 사진을 한 장씩 처리해 메모리를 아낀다 (폴더에 사진이 많아도 OK)
    for r in model.predict(args.source, imgsz=cfg["model"]["imgsz"], conf=min_conf,
                           device=device, stream=True, verbose=False):
        n_img += 1
        t_total += sum(r.speed.values())                  # 전처리+추론+후처리 시간(ms)

        # 클래스별 임계값으로 다시 거르기
        keep = [i for i, (c, s) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()))
                if s >= cls_cfg[names[int(c)]]["conf"]]
        r = r[keep]

        obstacles = []
        for c, s, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            cname = names[int(c)]
            info = cls_cfg[cname]
            obstacles.append({
                "class": cname, "name_ko": info["name_ko"],
                "type": info["type"], "action": info["action"],
                "confidence": round(s, 3),
                "box_xyxy": [round(v, 1) for v in xyxy],  # 왼쪽 위(x1,y1)·오른쪽 아래(x2,y2) 픽셀 좌표
            })
            counts[cname] += 1
        n_with_obstacle += bool(obstacles)

        stem = os.path.splitext(os.path.basename(r.path))[0]
        result = {
            "image": os.path.basename(r.path),
            "width": r.orig_shape[1], "height": r.orig_shape[0],
            "model_version": cfg["model"]["version"],
            "obstacle_count": len(obstacles),
            "types": sorted({o["type"] for o in obstacles}),   # 이 사진에 필요한 처리 유형 목록
            "obstacles": obstacles,
        }
        with open(os.path.join(args.out, "json", stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
        if not args.no_vis:
            save_jpg(os.path.join(args.out, "vis", stem + ".jpg"), r.plot())

    jsonl.close()

    # 집계만 출력
    print(f"모델: {cfg['model']['version']}  장치: {'GPU' if device == 0 else 'CPU'}")
    print(f"사진 {n_img}장 처리, 장애물이 있는 사진 {n_with_obstacle}장")
    for cname, info in cls_cfg.items():
        print(f"  {info['type']} {info['name_ko']:8s} {counts[cname]:6d}개 (임계값 {info['conf']})")
    if n_img:
        print(f"평균 처리 시간 {t_total / n_img:.1f}ms/장")
    print(f"결과: {args.out}")


if __name__ == "__main__":
    main()
