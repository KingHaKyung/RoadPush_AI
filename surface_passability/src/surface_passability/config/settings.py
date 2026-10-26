"""Environment-backed runtime roots without import-time filesystem I/O."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    runtime_root: Path
    data_root: Path
    model_root: Path
    cache_root: Path
    artifact_root: Path
    report_root: Path


def _configured_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _default_runtime_root() -> Path:
    # In the editable source tree this is ``RoadPush_AI``. Installed deployments
    # should set SURFACE_PASSABILITY_ROOT explicitly instead of depending on a
    # source-layout-relative location.
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    runtime_root = _configured_path(
        "SURFACE_PASSABILITY_ROOT", _default_runtime_root()
    )
    return RuntimeSettings(
        runtime_root=runtime_root,
        data_root=_configured_path(
            "SURFACE_PASSABILITY_DATA_ROOT", runtime_root / "data"
        ),
        model_root=_configured_path(
            "SURFACE_PASSABILITY_MODEL_ROOT",
            runtime_root / "models" / "surface_passability",
        ),
        cache_root=_configured_path(
            "SURFACE_PASSABILITY_CACHE_ROOT",
            runtime_root / ".cache" / "surface_passability",
        ),
        artifact_root=_configured_path(
            "SURFACE_PASSABILITY_ARTIFACT_ROOT",
            runtime_root / "artifacts" / "surface_passability",
        ),
        report_root=_configured_path(
            "SURFACE_PASSABILITY_REPORT_ROOT",
            runtime_root / "reports" / "surface_passability",
        ),
    )


def reset_settings_cache() -> None:
    """Clear cached settings for tests or an explicitly reconfigured process."""

    get_settings.cache_clear()

