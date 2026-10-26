"""Validated obstacle model and class settings."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ObstacleClassSettings:
    name_ko: str
    type: str
    action: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ObstacleSettings:
    model_path: Path
    model_sha256: str
    model_version: str
    image_size: int
    classes: dict[str, ObstacleClassSettings]

    @property
    def minimum_confidence(self) -> float:
        return min(value.confidence for value in self.classes.values())

    @classmethod
    def load(
        cls,
        *,
        config_path: Path | None = None,
        runtime_root: Path | None = None,
        model_root: Path | None = None,
    ) -> "ObstacleSettings":
        root = (runtime_root or _runtime_root()).resolve()
        config = (config_path or root / "configs" / "obstacle_detection.yaml").resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        with config.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("obstacle config must contain a mapping")
        model = _mapping(raw.get("model"), "model")
        configured_model_root = Path(
            os.environ.get("ROADPUSH_MODEL_ROOT", str(model_root or root / "models"))
        ).expanduser().resolve()
        relative_path = Path(str(model["relative_path"]))
        if relative_path.is_absolute():
            raise ValueError("model.relative_path must be relative to ROADPUSH_MODEL_ROOT")
        digest = str(model["sha256"]).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("model.sha256 must be a 64-character hexadecimal digest")
        image_size = int(model["imgsz"])
        if image_size < 1:
            raise ValueError("model.imgsz must be positive")
        raw_classes = _mapping(raw.get("classes"), "classes")
        classes: dict[str, ObstacleClassSettings] = {}
        for name, untyped in raw_classes.items():
            value = _mapping(untyped, f"classes.{name}")
            confidence = float(value["conf"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"classes.{name}.conf must be in [0, 1]")
            classes[str(name)] = ObstacleClassSettings(
                name_ko=str(value["name_ko"]),
                type=str(value["type"]),
                action=str(value["action"]),
                confidence=confidence,
            )
        if not classes:
            raise ValueError("at least one obstacle class is required")
        return cls(
            model_path=(configured_model_root / relative_path).resolve(),
            model_sha256=digest,
            model_version=str(model["version"]),
            image_size=image_size,
            classes=classes,
        )

    def to_legacy_config(self) -> dict[str, Any]:
        return {
            "model": {
                "path": str(self.model_path),
                "sha256": self.model_sha256,
                "version": self.model_version,
                "imgsz": self.image_size,
            },
            "classes": {
                name: {
                    "name_ko": value.name_ko,
                    "type": value.type,
                    "action": value.action,
                    "conf": value.confidence,
                }
                for name, value in self.classes.items()
            },
        }


def _runtime_root() -> Path:
    configured = os.environ.get("ROADPUSH_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[3]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping")
    return value
