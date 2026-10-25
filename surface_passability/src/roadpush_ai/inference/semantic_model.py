#!/usr/bin/env python3
"""Run one resumable raw semantic-segmentation benchmark condition.

Outputs are always restored to the EXIF-corrected source dimensions. The script
runs one model condition at a time so model-level checkpoints remain simple and
an interrupted long run can be resumed without repeating completed images.
"""

from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = get_settings().project_root
OUT = ROOT / "outputs" / "roi_connectivity_benchmark"
RAW = OUT / "raw"
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_KEYS = (
    "deeplab_author",
    "deeplab_letterbox",
    "mask2former_cityscapes_swin_t",
    "segformer_sidewalk_b0",
    "segformer_cityscapes_b0",
    "segformer_cityscapes_b2",
    "mask2former_mapillary_swin_l",
    "oneformer_cityscapes_swin_l",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(split: str, sample_id: str | None) -> list[dict]:
    with (OUT / "sample_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sample_id:
        rows = [row for row in rows if row["sample_id"] == sample_id]
        if not rows:
            raise ValueError(f"Unknown sample_id: {sample_id}")
    elif split != "all":
        rows = [row for row in rows if row["split"] == split]
    return rows


def atomic_save_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    Image.fromarray(array.astype(np.uint8)).save(temporary, optimize=True)
    temporary.replace(destination)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Runner:
    def __init__(self, model_key: str, requested_device: str):
        import torch

        self.torch = torch
        self.model_key = model_key
        if requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        self.model = None
        self.processor = None
        self.mode = None
        self.model_path = None
        self._load()

    def _load(self) -> None:
        torch = self.torch
        key = self.model_key
        if key.startswith("deeplab_"):
            sys.path.insert(0, str(next((ROOT / "work" / "selectstar").glob("segmentation-selectstar-*"))))
            from modules.models.deeplab_xception import DeepLabv3_plus
            from torchvision import transforms

            self.model_path = ROOT / "models" / "selectstar_deeplab"
            checkpoint_path = self.model_path / "model_iou_77.pth.tar"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=False, _print=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            del checkpoint
            self.processor = transforms.Compose(
                [
                    transforms.Resize((512, 1024), interpolation=Image.Resampling.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.mode = "letterbox512x1024" if key.endswith("letterbox") else "author_resize512x1024"
        elif key.startswith("segformer_"):
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            folder = {
                "segformer_sidewalk_b0": "segformer_sidewalk_alt",
                "segformer_cityscapes_b0": "segformer_cityscapes",
                "segformer_cityscapes_b2": "segformer_cityscapes_b2",
            }[key]
            self.model_path = ROOT / "models" / folder
            model = SegformerForSemanticSegmentation.from_pretrained(self.model_path, local_files_only=True)
            self.processor = SegformerImageProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.mode = "local_processor_default"
        elif key.startswith("mask2former_"):
            from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

            folder = {
                "mask2former_cityscapes_swin_t": "mask2former_cityscapes_tiny",
                "mask2former_mapillary_swin_l": "mask2former_swin_large_mapillary",
            }[key]
            self.model_path = ROOT / "models" / folder
            model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_path, local_files_only=True)
            self.processor = Mask2FormerImageProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.mode = "local_processor_default"
        elif key == "oneformer_cityscapes_swin_l":
            from transformers import (
                CLIPTokenizer,
                OneFormerForUniversalSegmentation,
                OneFormerImageProcessor,
                OneFormerProcessor,
            )

            self.model_path = ROOT / "models" / "oneformer_cityscapes_swin_large"
            config = json.loads((self.model_path / "preprocessor_config.json").read_text(encoding="utf-8"))
            config["repo_path"] = str(self.model_path)
            image_processor = OneFormerImageProcessor(**config)
            tokenizer = CLIPTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.processor = OneFormerProcessor(image_processor=image_processor, tokenizer=tokenizer)
            model = OneFormerForUniversalSegmentation.from_pretrained(self.model_path, local_files_only=True)
            self.mode = "semantic_task_local_taxonomy"
        else:
            raise ValueError(key)
        self.model = model.eval().to(self.device)

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict]:
        torch = self.torch
        width, height = image.size
        key = self.model_key
        started = time.perf_counter()
        with torch.inference_mode():
            if key.startswith("deeplab_"):
                if self.mode == "letterbox512x1024":
                    scale = min(1024 / width, 512 / height)
                    resized_width, resized_height = round(width * scale), round(height * scale)
                    pad_x, pad_y = (1024 - resized_width) // 2, (512 - resized_height) // 2
                    boxed = Image.new("RGB", (1024, 512), (128, 128, 128))
                    boxed.paste(
                        image.resize((resized_width, resized_height), Image.Resampling.BILINEAR),
                        (pad_x, pad_y),
                    )
                    tensor = self.processor(boxed).unsqueeze(0).to(self.device)
                else:
                    tensor = self.processor(image).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                small = output.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
                if self.mode == "letterbox512x1024":
                    small = small[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width]
                raw = np.asarray(Image.fromarray(small).resize((width, height), Image.Resampling.NEAREST))
                tensor_shape = list(tensor.shape)
            else:
                kwargs = {"images": image, "return_tensors": "pt"}
                if key == "oneformer_cityscapes_swin_l":
                    kwargs["task_inputs"] = ["semantic"]
                inputs = self.processor(**kwargs)
                tensor_shape = list(inputs["pixel_values"].shape)
                inputs = {name: value.to(self.device) if hasattr(value, "to") else value for name, value in inputs.items()}
                output = self.model(**inputs)
                raw = self.processor.post_process_semantic_segmentation(
                    output, target_sizes=[(height, width)]
                )[0].detach().cpu().numpy().astype(np.uint8)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        else:
            peak = None
        return raw, {
            "seconds": round(time.perf_counter() - started, 4),
            "tensor_shape": tensor_shape,
            "peak_cuda_bytes": peak,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--split", choices=("pilot", "holdout", "all"), default="pilot")
    parser.add_argument("--sample-id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    rows = read_rows(args.split, args.sample_id)
    destination_dir = RAW / args.model
    destination_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[dict, Image.Image]] = []
    for row in rows:
        source = ROOT / row["relative_path"]
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        destination = destination_dir / f"{row['sample_id']}.png"
        if destination.exists():
            with Image.open(destination) as existing:
                if existing.size != image.size:
                    raise RuntimeError(f"Existing output has wrong size: {destination}")
            print(f"SKIP {args.model} {row['sample_id']} existing", flush=True)
            continue
        pending.append((row, image))

    if not pending:
        print(f"DONE {args.model}: no pending images", flush=True)
        return

    runner = Runner(args.model, args.device)
    records_path = OUT / "inference_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else {}
    records.setdefault(args.model, {})
    print(f"LOAD {args.model} device={runner.device} pending={len(pending)}", flush=True)
    for row, image in pending:
        sample_id = row["sample_id"]
        source = ROOT / row["relative_path"]
        destination = destination_dir / f"{sample_id}.png"
        raw, timing = runner.predict(image)
        if raw.shape != (image.height, image.width):
            raise RuntimeError(f"Wrong result dimensions for {sample_id}: {raw.shape} vs {(image.height, image.width)}")
        atomic_save_png(raw, destination)
        record = {
            "sample_id": sample_id,
            "split": row["split"],
            "model_key": args.model,
            "model_path": str(runner.model_path.relative_to(ROOT)),
            "preprocessing": runner.mode,
            "device": str(runner.device),
            "torch_version": runner.torch.__version__,
            "source_path": row["relative_path"],
            "source_sha256": sha256(source),
            "source_size": [image.width, image.height],
            "output_path": str(destination.relative_to(ROOT)),
            "output_sha256": sha256(destination),
            "class_ids_present": [int(value) for value in np.unique(raw)],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **timing,
        }
        records[args.model][sample_id] = record
        write_json(records_path, records)
        write_json(destination.with_suffix(".meta.json"), record)
        print(f"OK {args.model} {sample_id} {timing['seconds']}s {timing['tensor_shape']}", flush=True)

    del runner
    gc.collect()
    print(f"DONE {args.model}: completed {len(pending)}", flush=True)


if __name__ == "__main__":
    main()
