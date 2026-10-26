"""Typed obstacle-detection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ObstacleDetection:
    detection_id: int
    class_name: str
    name_ko: str
    type: str
    action: str
    confidence: float
    box_native_xyxy: tuple[float, float, float, float]

    def to_dict(
        self,
        *,
        box_analysis_xyxy: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "detection_id": self.detection_id,
            "class": self.class_name,
            "name_ko": self.name_ko,
            "type": self.type,
            "action": self.action,
            "confidence": self.confidence,
            "box_native_xyxy": list(self.box_native_xyxy),
        }
        if box_analysis_xyxy is not None:
            value["box_analysis_xyxy"] = list(box_analysis_xyxy)
        return value

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "name_ko": self.name_ko,
            "type": self.type,
            "action": self.action,
            "confidence": self.confidence,
            "box_xyxy": list(self.box_native_xyxy),
        }


@dataclass(frozen=True, slots=True)
class ObstacleDetectionResult:
    width: int
    height: int
    model_version: str
    model_sha256: str
    device: str
    inference_time_ms: float
    detections: tuple[ObstacleDetection, ...]

    def to_legacy_dict(self, image_name: str) -> dict[str, Any]:
        obstacles = [detection.to_legacy_dict() for detection in self.detections]
        return {
            "image": image_name,
            "width": self.width,
            "height": self.height,
            "model_version": self.model_version,
            "obstacle_count": len(obstacles),
            "types": sorted({item["type"] for item in obstacles}),
            "obstacles": obstacles,
        }
