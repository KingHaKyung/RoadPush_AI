"""Safe logical artifact references for relocatable deployments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath

from .settings import RuntimeSettings, get_settings


@dataclass(frozen=True)
class ArtifactRef:
    namespace: str
    artifact_id: str
    version: str
    relative_path: PurePosixPath
    sha256: str | None = None


class ArtifactLocator:
    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def _root(self, namespace: str) -> Path:
        roots = {
            "model": self.settings.model_root,
            "dataset": self.settings.data_root,
            "cache": self.settings.cache_root,
            "artifact": self.settings.artifact_root,
            "report": self.settings.report_root,
            "phase6": self.settings.artifact_root / "phase6",
            "phase7a": self.settings.artifact_root / "phase7a",
            "evidence": self.settings.artifact_root / "evidence",
        }
        try:
            return roots[namespace].resolve()
        except KeyError as exc:
            raise ValueError(f"unknown artifact namespace: {namespace}") from exc

    def resolve_read(self, ref: ArtifactRef) -> Path:
        path = self._safe_path(ref)
        if not path.exists():
            raise FileNotFoundError(path)
        if ref.sha256 and path.is_file() and self._sha256(path) != ref.sha256.lower():
            raise ValueError(f"artifact hash mismatch: {ref.artifact_id}")
        return path

    def resolve_write(self, ref: ArtifactRef) -> Path:
        path = self._safe_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_path(self, ref: ArtifactRef) -> Path:
        relative = Path(*ref.relative_path.parts)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative and cannot contain ..")
        root = self._root(ref.namespace)
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("artifact path escapes its namespace root")
        return candidate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
