"""Presentation-only Korean font resolution.

Numeric pipelines must never import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


@dataclass(frozen=True)
class ResolvedFont:
    path: Path
    sha256: str
    source: str


class FontResolver:
    def __init__(self, bundled_root: Path | None = None) -> None:
        self.bundled_root = bundled_root

    def resolve_korean(self, *, bold: bool = False) -> ResolvedFont:
        configured = os.environ.get("ROADPUSH_AI_FONT_BOLD" if bold else "ROADPUSH_AI_FONT")
        candidates: list[tuple[str, Path]] = []
        if configured:
            candidates.append(("configured", Path(configured)))
        if self.bundled_root:
            name = "NotoSansKR-Bold.otf" if bold else "NotoSansKR-Regular.otf"
            candidates.append(("bundled", self.bundled_root / name))
        windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates.append(("windows-malgun", windows / ("malgunbd.ttf" if bold else "malgun.ttf")))
        candidates.extend(
            (
                ("system-noto", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")),
                ("system-noto", Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")),
            )
        )
        for source, path in candidates:
            if path.is_file():
                return ResolvedFont(path=path.resolve(), sha256=self._sha256(path), source=source)
        raise FileNotFoundError(
            "No Korean-capable review font found. Set ROADPUSH_AI_FONT and "
            "ROADPUSH_AI_FONT_BOLD or install the documented bundled font."
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

