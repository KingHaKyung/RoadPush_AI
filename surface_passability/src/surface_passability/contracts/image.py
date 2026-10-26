"""Prepared image contract shared by inference and downstream algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PreparedImage:
    request_id: str
    sample_id: str
    source_path: Path
    source_uri: str
    content_sha256: str
    native_rgb: NDArray[np.uint8]
    rgb: NDArray[np.uint8]
    original_size: tuple[int, int]
    analysis_size: tuple[int, int]
    exif_orientation_normalized: bool
    resize_filter: str

    def __post_init__(self) -> None:
        if self.native_rgb.dtype != np.uint8 or self.rgb.dtype != np.uint8:
            raise TypeError("PreparedImage RGB arrays must use uint8")
        expected_native_shape = (self.original_size[1], self.original_size[0], 3)
        if self.native_rgb.shape != expected_native_shape:
            raise ValueError(
                "PreparedImage.native_rgb shape "
                f"{self.native_rgb.shape!r} != {expected_native_shape!r}"
            )
        expected_shape = (self.analysis_size[1], self.analysis_size[0], 3)
        if self.rgb.shape != expected_shape:
            raise ValueError(
                f"PreparedImage.rgb shape {self.rgb.shape!r} != {expected_shape!r}"
            )

