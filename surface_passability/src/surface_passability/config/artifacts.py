"""Safe logical artifact references for relocatable deployments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal

from surface_passability.config.settings import RuntimeSettings, get_settings


ArtifactNamespace = Literal["model", "dataset", "cache", "artifact", "report"]


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    namespace: ArtifactNamespace
    artifact_id: str
    version: str
    relative_path: PurePosixPath
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if not self.version.strip():
            raise ValueError("version must not be empty")
        if self.sha256 is not None:
            normalized = self.sha256.lower()
            if len(normalized) != 64 or any(
                character not in "0123456789abcdef" for character in normalized
            ):
                raise ValueError("sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sha256", normalized)


class ArtifactLocator:
    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def resolve_read(self, ref: ArtifactRef) -> Path:
        path = self._safe_path(ref)
        if not path.is_file():
            raise FileNotFoundError(path)
        if ref.sha256 is not None and _sha256(path) != ref.sha256:
            raise ValueError(f"artifact hash mismatch: {ref.artifact_id}")
        return path

    def resolve_write(self, ref: ArtifactRef) -> Path:
        path = self._safe_path(ref)
        if path.exists() and path.is_dir():
            raise IsADirectoryError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_path(self, ref: ArtifactRef) -> Path:
        posix_parts = ref.relative_path.parts
        if (
            not posix_parts
            or ref.relative_path.name in ("", ".")
            or ref.relative_path.is_absolute()
            or ".." in posix_parts
            or any(":" in part or "\\" in part for part in posix_parts)
        ):
            raise ValueError(
                "artifact path must name a relative file and cannot contain .."
            )
        relative = Path(*posix_parts)
        root = self._root(ref.namespace).resolve()
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("artifact path escapes its namespace root")
        return candidate

    def _root(self, namespace: ArtifactNamespace) -> Path:
        roots = {
            "model": self.settings.model_root,
            "dataset": self.settings.data_root,
            "cache": self.settings.cache_root,
            "artifact": self.settings.artifact_root,
            "report": self.settings.report_root,
        }
        try:
            return roots[namespace]
        except KeyError as exc:
            raise ValueError(f"unknown artifact namespace: {namespace}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

