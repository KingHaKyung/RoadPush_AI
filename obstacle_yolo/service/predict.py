"""Legacy obstacle CLI backed by the reusable integrated detector."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from roadpush_ai.obstacle import ObstacleDetector, ObstacleSettings


HERE = Path(__file__).resolve().parent
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def load_config():
    """Compatibility adapter retained for legacy attach_cause.py."""

    return ObstacleSettings.load().to_legacy_config()


def save_jpg(path, image):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buffer.tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="사진 파일 또는 폴더")
    parser.add_argument(
        "--out",
        default=str(HERE / "outputs" / time.strftime("%Y%m%d_%H%M%S")),
        help="결과 폴더 (기본: service/outputs/날짜_시각)",
    )
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    arguments = parser.parse_args()

    images = _input_images(Path(arguments.source).expanduser().resolve())
    output = Path(arguments.out).expanduser().resolve()
    (output / "json").mkdir(parents=True, exist_ok=True)
    if not arguments.no_vis:
        (output / "vis").mkdir(parents=True, exist_ok=True)

    settings = ObstacleSettings.load()
    detector = ObstacleDetector(settings=settings, requested_device=arguments.device)
    counts = collections.Counter()
    images_with_obstacle = 0
    total_time_ms = 0.0
    with (output / "results.jsonl").open("w", encoding="utf-8") as jsonl:
        for image_path in images:
            with Image.open(image_path) as opened:
                native_rgb = ImageOps.exif_transpose(opened).convert("RGB")
                rgb_array = np.asarray(native_rgb, dtype=np.uint8)
            result = detector.detect(rgb_array)
            record = result.to_legacy_dict(image_path.name)
            total_time_ms += result.inference_time_ms
            images_with_obstacle += bool(result.detections)
            counts.update(detection.class_name for detection in result.detections)
            (output / "json" / f"{image_path.stem}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not arguments.no_vis:
                bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                for detection in result.detections:
                    x0, y0, x1, y1 = map(int, detection.box_native_xyxy)
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.putText(
                        bgr,
                        f"{detection.type} {detection.class_name} {detection.confidence:.2f}",
                        (x0, max(10, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                save_jpg(output / "vis" / f"{image_path.stem}.jpg", bgr)

    print(f"모델: {settings.model_version}  장치: {detector.device}")
    print(f"사진 {len(images)}장 처리, 장애물이 있는 사진 {images_with_obstacle}장")
    for class_name, class_settings in settings.classes.items():
        print(
            f"  {class_settings.type} {class_settings.name_ko:8s} "
            f"{counts[class_name]:6d}개 (임계값 {class_settings.confidence})"
        )
    if images:
        print(f"평균 처리 시간 {total_time_ms / len(images):.1f}ms/장")
    print(f"결과: {output}")


def _input_images(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(source)
    images = tuple(
        path for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no supported images in {source}")
    return images


if __name__ == "__main__":
    main()
