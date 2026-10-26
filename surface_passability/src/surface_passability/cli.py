"""Single production command for surface-passability analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from surface_passability.contracts.request import AnalysisRequest
from surface_passability.dependencies import create_default_dependencies
from surface_passability.pipeline import analyze_photo


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadpush-surface-analyze",
        description="사진의 보행 surface, 경계, 통행 가능 폭을 분석합니다.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--request-id")
    parser.add_argument(
        "--input-profile", choices=("image_upload", "cctv"), default="image_upload"
    )
    parser.add_argument("--surface-width-m", type=float)
    parser.add_argument("--agent-width-m", type=float)
    parser.add_argument("--safety-factor", type=float)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    images = _input_images(arguments.image, arguments.input_dir)
    if len(images) > 1 and (arguments.sample_id or arguments.request_id):
        raise ValueError("batch input cannot use a shared sample-id or request-id")
    jobs = tuple(
        (
            image,
            arguments.sample_id or image.stem,
            arguments.request_id or arguments.sample_id or image.stem,
        )
        for image in images
    )
    sample_ids = [sample_id for _, sample_id, _ in jobs]
    if len(sample_ids) != len({sample_id.casefold() for sample_id in sample_ids}):
        raise ValueError("batch input contains duplicate sample ids")
    for _, sample_id, _ in jobs:
        _validate_output_identifier(sample_id)
    output_root = arguments.output.expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dependencies = create_default_dependencies(requested_device=arguments.device)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.staging.",
        dir=output_root.parent,
    ) as staging_name:
        staging = Path(staging_name)
        for image, sample_id, request_id in jobs:
            request = AnalysisRequest(
                request_id=request_id,
                sample_id=sample_id,
                image_uri=image,
                image_content_sha256=_file_sha256(image),
                input_profile=arguments.input_profile,
                surface_width_m=arguments.surface_width_m,
                agent_width_m=arguments.agent_width_m,
                safety_factor=arguments.safety_factor,
            )
            response = analyze_photo(request, dependencies=dependencies)
            _atomic_write_json(staging / f"{sample_id}.json", response.to_dict())
        output_root.mkdir(parents=True, exist_ok=True)
        for _, sample_id, _ in jobs:
            os.replace(
                staging / f"{sample_id}.json",
                output_root / f"{sample_id}.json",
            )
    return 0


def _validate_output_identifier(value: str) -> None:
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if (
        not value.strip()
        or value in (".", "..")
        or any(character in '<>:"/\\|?*' for character in value)
        or value.endswith((".", " "))
        or value.split(".", 1)[0].upper() in reserved
    ):
        raise ValueError("sample-id must be a portable file-name component")


def _input_images(image: Path | None, input_dir: Path | None) -> tuple[Path, ...]:
    if image is not None:
        resolved = image.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return (resolved,)
    assert input_dir is not None
    root = input_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    images = tuple(
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no supported images in {root}")
    return images


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
