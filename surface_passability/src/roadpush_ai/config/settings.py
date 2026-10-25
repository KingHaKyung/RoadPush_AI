"""Environment-backed runtime roots with no import-time I/O."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    data_root: Path
    model_root: Path
    cache_root: Path
    artifact_root: Path
    report_root: Path


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    installed_root = Path(__file__).resolve().parents[3]
    project = _path("ROADPUSH_AI_ROOT", installed_root)
    return RuntimeSettings(
        project_root=project,
        data_root=_path("ROADPUSH_AI_DATA_ROOT", project / "data"),
        model_root=_path("ROADPUSH_AI_MODEL_ROOT", project / "models"),
        cache_root=_path("ROADPUSH_AI_CACHE_ROOT", project / ".cache"),
        artifact_root=_path("ROADPUSH_AI_ARTIFACT_ROOT", project / "artifacts"),
        report_root=_path("ROADPUSH_AI_REPORT_ROOT", project / "reports"),
    )
