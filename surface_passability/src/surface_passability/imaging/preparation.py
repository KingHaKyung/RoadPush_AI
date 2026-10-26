"""Decode, verify, orient, and resize one local input image."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from surface_passability.contracts.image import PreparedImage
from surface_passability.contracts.request import AnalysisRequest


DEFAULT_MAX_SIDE = 700


def analysis_size(
    width: int, height: int, *, max_side: int = DEFAULT_MAX_SIDE
) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if max_side < 1:
        raise ValueError("max_side must be positive")
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_image(
    request: AnalysisRequest, *, max_side: int = DEFAULT_MAX_SIDE
) -> PreparedImage:
    source_path = Path(request.image_uri).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    content_sha256 = _file_sha256(source_path)
    if content_sha256 != request.image_content_sha256:
        raise ValueError(
            f"image content hash mismatch for sample {request.sample_id}: "
            f"expected {request.image_content_sha256}, got {content_sha256}"
        )

    with Image.open(source_path) as opened:
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = normalized.size
        resized_size = analysis_size(*original_size, max_side=max_side)
        native_rgb = np.asarray(normalized, dtype=np.uint8)
        rgb = np.asarray(
            normalized.resize(resized_size, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

    return PreparedImage(
        request_id=request.request_id,
        sample_id=request.sample_id,
        source_path=source_path,
        source_uri=str(request.image_uri),
        content_sha256=content_sha256,
        native_rgb=native_rgb,
        rgb=rgb,
        original_size=original_size,
        analysis_size=resized_size,
        exif_orientation_normalized=True,
        resize_filter="PIL.Image.Resampling.LANCZOS",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

