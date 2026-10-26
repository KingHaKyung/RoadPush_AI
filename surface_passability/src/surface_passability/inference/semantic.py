"""Accepted local Mask2Former semantic inference implementation."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import time
from typing import Any

import numpy as np
from PIL import Image


class Mask2FormerSemanticModel:
    def __init__(self, model_root: Path, requested_device: str = "cpu") -> None:
        import torch
        import transformers
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        self.torch = torch
        self.transformers_version = transformers.__version__
        if requested_device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(min(8, os.cpu_count() or 4))

        root = model_root.resolve()
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            root, local_files_only=True
        )
        self.model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(
                root, local_files_only=True
            )
            .eval()
            .to(self.device)
        )
        self._predict_lock = Lock()

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
        """Infer one native class map with bounded post-processing memory."""

        with self._predict_lock:
            return self._predict_locked(image)

    def _predict_locked(
        self, image: Image.Image
    ) -> tuple[np.ndarray, dict[str, Any]]:
        width, height = image.size
        started = time.perf_counter()
        with self.torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt")
            tensor_shape = list(inputs["pixel_values"].shape)
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
            output = self.model(**inputs)
            raw = _memory_bounded_semantic_map(
                self.torch,
                output,
                target_size=(height, width),
            )
            del output, inputs
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
            peak = int(self.torch.cuda.max_memory_allocated())
            self.torch.cuda.reset_peak_memory_stats()
        else:
            peak = None
        return raw, {
            "seconds": round(time.perf_counter() - started, 4),
            "tensor_shape": tensor_shape,
            "peak_cuda_bytes": peak,
            "device": str(self.device),
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers_version,
        }


SEMANTIC_CLASS_CHUNK_SIZE = 4
MASK2FORMER_PREPROCESSED_SIZE = (384, 384)


def _memory_bounded_semantic_map(
    torch: Any,
    output: Any,
    *,
    target_size: tuple[int, int],
    class_chunk_size: int = SEMANTIC_CLASS_CHUNK_SIZE,
) -> np.ndarray:
    """Reproduce the accepted HF native argmax without a full native tensor.

    Hugging Face first forms 65 class logits at 384x384, then bilinearly
    expands all classes to native resolution before ``argmax``.  Expanding all
    channels together needs almost 6 GiB for a 5712x4284 image.  Interpolation
    is channel-independent, so processing ordered class chunks and updating on
    strict improvement preserves the exact lowest-class-id tie behavior.
    """

    if class_chunk_size < 1:
        raise ValueError("class_chunk_size must be positive")
    class_queries_logits = output.class_queries_logits
    masks_queries_logits = torch.nn.functional.interpolate(
        output.masks_queries_logits,
        size=MASK2FORMER_PREPROCESSED_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()
    segmentation = torch.einsum(
        "bqc,bqhw->bchw", masks_classes, masks_probs
    )
    if int(segmentation.shape[0]) != 1:
        raise ValueError("single-image semantic inference requires batch size 1")
    segmentation = segmentation[0]
    height, width = target_size
    best_values = torch.full(
        (height, width),
        -torch.inf,
        dtype=segmentation.dtype,
        device=segmentation.device,
    )
    best_classes = torch.zeros(
        (height, width),
        dtype=torch.uint8,
        device=segmentation.device,
    )
    for start in range(0, int(segmentation.shape[0]), class_chunk_size):
        stop = min(start + class_chunk_size, int(segmentation.shape[0]))
        resized = torch.nn.functional.interpolate(
            segmentation[start:stop].unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        values, offsets = resized.max(dim=0)
        update = values > best_values
        best_values[update] = values[update]
        best_classes[update] = (offsets[update] + start).to(torch.uint8)
        del resized, values, offsets, update
    result = best_classes.cpu().numpy()
    del best_classes, best_values, segmentation, masks_probs
    del masks_classes, masks_queries_logits, class_queries_logits
    return result

