"""Verified read-only compatibility evidence for historical occupancy results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from surface_passability.io.hashing import hash_mask_after_uint8_cast


@dataclass(frozen=True, slots=True)
class PersistedOccupancyEvidence:
    occupancy_mask: np.ndarray
    occupancy_pixels: int
    instance_records: tuple[dict[str, Any], ...]


class PersistedOccupancyEvidenceIndex:
    """Load narrowly bound legacy evidence without trusting a sample ID alone."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "surface_persisted_occupancy_v1":
            raise ValueError(f"unsupported occupancy evidence manifest: {manifest_path}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"occupancy evidence records are invalid: {manifest_path}")
        self._records: dict[str, dict[str, Any]] = {}
        for record in records:
            sample_id = str(record["sample_id"])
            if sample_id in self._records:
                raise ValueError(f"duplicate occupancy evidence sample: {sample_id}")
            relative = Path(str(record["mask_relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe occupancy evidence path: {relative}")
            mask_path = (self.root / relative).resolve()
            if not mask_path.is_relative_to(self.root) or not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            if _file_sha256(mask_path) != record["mask_file_sha256"]:
                raise ValueError(f"occupancy evidence file hash mismatch: {sample_id}")
            native = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if native is None:
                raise FileNotFoundError(mask_path)
            mask = np.asarray(native > 0, dtype=bool)
            if int(mask.sum()) != int(record["occupancy_pixels"]):
                raise ValueError(f"occupancy evidence pixel count mismatch: {sample_id}")
            mask.setflags(write=False)
            stored = dict(record)
            stored["occupancy_mask"] = mask
            self._records[sample_id] = stored

    def provide(
        self,
        *,
        sample_id: str,
        image_content_sha256: str,
        selected_surface: str | None,
        blocker_mask: np.ndarray | None,
        structural_mask: np.ndarray | None,
    ) -> PersistedOccupancyEvidence | None:
        record = self._records.get(sample_id)
        if record is None:
            return None
        if (
            record["image_content_sha256"] != image_content_sha256
            or record["selected_surface"] != selected_surface
            or blocker_mask is None
            or structural_mask is None
            or record["blocker_mask_sha256"]
            != hash_mask_after_uint8_cast(blocker_mask)
            or record["structural_mask_sha256"]
            != hash_mask_after_uint8_cast(structural_mask)
        ):
            return None
        return PersistedOccupancyEvidence(
            occupancy_mask=record["occupancy_mask"],
            occupancy_pixels=int(record["occupancy_pixels"]),
            instance_records=tuple(
                dict(value) for value in record.get("instance_records", ())
            ),
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
