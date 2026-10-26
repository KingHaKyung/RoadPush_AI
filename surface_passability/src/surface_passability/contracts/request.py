"""Validated input contract for one surface-passability analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Literal, TypeAlias


InputProfile: TypeAlias = Literal["image_upload", "cctv"]


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    request_id: str
    sample_id: str
    image_uri: str | Path
    image_content_sha256: str
    input_profile: InputProfile = "image_upload"
    surface_width_m: float | None = None
    agent_width_m: float | None = None
    safety_factor: float | None = None
    render_review: bool = False

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not str(self.image_uri).strip():
            raise ValueError("image_uri must not be empty")
        digest = self.image_content_sha256.lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "image_content_sha256 must be a 64-character hexadecimal digest"
            )
        object.__setattr__(self, "image_content_sha256", digest)
        if self.input_profile not in ("image_upload", "cctv"):
            raise ValueError(f"unsupported input_profile: {self.input_profile}")
        for name in ("surface_width_m", "agent_width_m"):
            self._normalize_optional_number(name, minimum=0.0, inclusive=False)
        self._normalize_optional_number(
            "safety_factor", minimum=1.0, inclusive=True
        )

    def _normalize_optional_number(
        self, name: str, *, minimum: float, inclusive: bool
    ) -> None:
        value = getattr(self, name)
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be finite")
        valid = normalized >= minimum if inclusive else normalized > minimum
        if not valid:
            operator = ">=" if inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
        object.__setattr__(self, name, normalized)

