"""Reusable obstacle detector with verified lazy model loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from roadpush_ai.obstacle.contracts import (
    ObstacleDetection,
    ObstacleDetectionResult,
)
from roadpush_ai.obstacle.settings import ObstacleSettings


class ObstacleModel(Protocol):
    names: Any

    def predict(self, image: np.ndarray, **kwargs: object) -> Sequence[Any]: ...


class ObstacleModelProvider:
    """Verify and lazy-load one obstacle model for the worker lifetime."""

    def __init__(self, settings: ObstacleSettings) -> None:
        self.settings = settings
        self._model: ObstacleModel | None = None
        self._load_lock = Lock()

    def get_model(self) -> ObstacleModel:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    path = self.settings.model_path
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    actual = _sha256_file(path)
                    if actual != self.settings.model_sha256:
                        raise ValueError(
                            f"obstacle model hash mismatch: expected "
                            f"{self.settings.model_sha256}, got {actual}"
                        )
                    from ultralytics import YOLO

                    self._model = YOLO(str(path))
        return self._model


class ObstacleDetector:
    """Run the box detector on a previously EXIF-normalized RGB image."""

    def __init__(
        self,
        *,
        settings: ObstacleSettings | None = None,
        model_provider: ObstacleModelProvider | None = None,
        requested_device: str | int | None = None,
    ) -> None:
        self.settings = settings or ObstacleSettings.load()
        self.model_provider = model_provider or ObstacleModelProvider(self.settings)
        self.device = _resolve_device(requested_device)

    def detect(
        self,
        native_rgb: NDArray[np.uint8],
    ) -> ObstacleDetectionResult:
        if native_rgb.dtype != np.uint8:
            raise TypeError("native_rgb must use uint8")
        if native_rgb.ndim != 3 or native_rgb.shape[2] != 3:
            raise ValueError("native_rgb must have shape (height, width, 3)")
        model = self.model_provider.get_model()
        bgr = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2BGR)
        result = model.predict(
            bgr,
            imgsz=self.settings.image_size,
            conf=self.settings.minimum_confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[ObstacleDetection] = []
        boxes = result.boxes
        if boxes is not None:
            for class_id, score, xyxy in zip(
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            ):
                class_name = str(model.names[int(class_id)])
                class_settings = self.settings.classes.get(class_name)
                if class_settings is None:
                    raise ValueError(f"model returned unconfigured class {class_name!r}")
                if score < class_settings.confidence:
                    continue
                detections.append(ObstacleDetection(
                    detection_id=len(detections),
                    class_name=class_name,
                    name_ko=class_settings.name_ko,
                    type=class_settings.type,
                    action=class_settings.action,
                    confidence=round(float(score), 3),
                    box_native_xyxy=tuple(round(float(value), 1) for value in xyxy),
                ))
        speed = getattr(result, "speed", {}) or {}
        inference_time_ms = float(sum(float(value) for value in speed.values()))
        height, width = native_rgb.shape[:2]
        return ObstacleDetectionResult(
            width=width,
            height=height,
            model_version=self.settings.model_version,
            model_sha256=self.settings.model_sha256,
            device=str(self.device),
            inference_time_ms=inference_time_ms,
            detections=tuple(detections),
        )


def _resolve_device(requested: str | int | None) -> str | int:
    if requested is not None:
        if isinstance(requested, str) and not requested.strip():
            raise ValueError("requested_device must not be empty")
        if requested == "cuda":
            return 0
        return requested
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
