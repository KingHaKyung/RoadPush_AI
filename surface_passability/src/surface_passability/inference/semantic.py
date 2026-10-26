"""Accepted local Mask2Former semantic inference implementation."""

from __future__ import annotations

import os
from pathlib import Path
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

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
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
            raw = (
                self.processor.post_process_semantic_segmentation(
                    output, target_sizes=[(height, width)]
                )[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
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

