from __future__ import annotations

from roadpush_ai.config.settings import get_settings

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from roadpush_ai import legacy_policy as legacy


ROOT = get_settings().project_root


@dataclass
class SemanticInputs:
    image: np.ndarray
    labels: np.ndarray
    cache_path: Path
    inference_run: bool
    inference: dict | None


def _atomic_save_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    Image.fromarray(array.astype(np.uint8)).save(temporary, optimize=True)
    temporary.replace(destination)


def _cpu_runner(model_holder: dict):
    if "mask2former_cpu" not in model_holder:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from roadpush_ai.inference.semantic_model import Runner

        runner = Runner("mask2former_mapillary_swin_l", "cpu")
        if runner.device.type != "cpu":
            raise RuntimeError(f"Semantic runner is not CPU-only: {runner.device}")
        model_holder["mask2former_cpu"] = runner
    return model_holder["mask2former_cpu"]


def load_or_infer_inputs(item: dict, model_holder: dict) -> SemanticInputs:
    """Load a cached Mapillary map or create it with local CPU inference.

    Cached inputs deliberately reproduce the frozen V2 loader. A cache miss is
    inferred at the EXIF-corrected source resolution, saved atomically, then
    resized with nearest-neighbor interpolation to the same analysis size.
    """

    image_path = ROOT / item["relative_path"]
    cache_path = ROOT / item["class_map_path"]
    with Image.open(image_path) as opened:
        source_image = ImageOps.exif_transpose(opened).convert("RGB")
    source_size = legacy.integrated.fit(*source_image.size)
    analysis_image = np.asarray(
        source_image.resize(source_size, Image.Resampling.LANCZOS)
    )

    native = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
    expected_native_shape = (source_image.height, source_image.width)
    cache_compatible = native is not None and native.shape[:2] == expected_native_shape
    inference_run = not cache_compatible
    inference = None
    if inference_run:
        runner = _cpu_runner(model_holder)
        native, inference = runner.predict(source_image)
        _atomic_save_png(native, cache_path)

    if native is None:
        raise RuntimeError(f"Semantic cache could not be loaded or created: {cache_path}")
    labels = cv2.resize(
        native.astype(np.uint8), source_size, interpolation=cv2.INTER_NEAREST
    )
    return SemanticInputs(
        image=analysis_image,
        labels=labels,
        cache_path=cache_path,
        inference_run=inference_run,
        inference=inference,
    )
