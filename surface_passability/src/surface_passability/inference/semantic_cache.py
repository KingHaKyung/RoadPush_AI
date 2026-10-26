"""Verified semantic cache and cache-or-infer evidence provider."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from surface_passability.config.model_profiles import (
    file_sha256,
    verify_mask2former_mapillary_model,
)
from surface_passability.contracts.image import PreparedImage
from surface_passability.contracts.semantic import (
    SemanticEvidence,
    SemanticModelIdentity,
)
from surface_passability.inference.semantic import Mask2FormerSemanticModel


CACHE_SCHEMA_VERSION = "surface_passability_semantic_cache_v2"
CACHE_REUSE_POLICY = "device_independent_canonical_evidence_v1"
POSTPROCESSING_IDENTITY = (
    "Mask2FormerImageProcessor.post_process_semantic_segmentation:"
    "target_native_size:uint8:v1"
)


class SemanticCacheValidationError(ValueError):
    """The cache exists but cannot be trusted for the current request/model."""


class SemanticPredictor(Protocol):
    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]: ...


class SemanticEvidenceProvider:
    def __init__(
        self,
        *,
        model_root: Path,
        cache_root: Path,
        requested_device: str = "cpu",
        model_identity: SemanticModelIdentity | None = None,
        predictor_factory: Callable[[Path, str], SemanticPredictor]
        | None = None,
    ) -> None:
        self.model_root = model_root.resolve()
        self.cache_root = cache_root.resolve()
        self.requested_device = requested_device
        self.model_identity = model_identity or verify_mask2former_mapillary_model(
            self.model_root
        )
        self._predictor_factory = predictor_factory or Mask2FormerSemanticModel
        self._predictor: SemanticPredictor | None = None
        self._predictor_lock = Lock()
        self.inference_count = 0

    def provide(
        self,
        prepared: PreparedImage,
        *,
        frozen_cache_path: Path | None = None,
        frozen_cache_sha256: str | None = None,
    ) -> SemanticEvidence:
        frozen_digest = _normalize_optional_sha256(frozen_cache_sha256)
        if frozen_cache_path is None and frozen_digest is not None:
            raise ValueError(
                "frozen_cache_sha256 requires an explicit frozen_cache_path"
            )
        if frozen_cache_path is not None and frozen_digest is None:
            raise ValueError(
                "frozen_cache_path requires frozen_cache_sha256"
            )

        cache_key = semantic_cache_key(prepared, self.model_identity)
        if frozen_cache_path is not None:
            destination = frozen_cache_path.resolve()
            if not destination.is_file():
                raise FileNotFoundError(
                    "frozen semantic cache must already exist and is read-only: "
                    f"{destination}"
                )
        else:
            destination = _writable_cache_destination(
                self.cache_root,
                profile_id=self.model_identity.profile_id,
                cache_key=cache_key,
            )
        metadata_path = destination.with_suffix(".meta.json")
        cache_hit = destination.is_file()
        inference_run = False
        inference_metadata: dict[str, Any] | None = None
        producer_provenance: dict[str, Any]

        if cache_hit:
            native, cache_file_hash = _read_native_map(destination)
            _validate_native_shape(native, prepared)
            if frozen_digest is not None and cache_file_hash != frozen_digest:
                raise SemanticCacheValidationError(
                    "frozen semantic cache hash mismatch: "
                    f"expected {frozen_digest}, got {cache_file_hash}"
                )
            if metadata_path.is_file():
                metadata = _read_metadata(metadata_path)
                if metadata.get("schema_version") == CACHE_SCHEMA_VERSION:
                    producer_provenance = _validate_metadata(
                        metadata,
                        prepared=prepared,
                        model=self.model_identity,
                        cache_key=cache_key,
                        cache_file_sha256=cache_file_hash,
                        native_array_sha256=array_sha256(native),
                    )
                elif frozen_digest is not None:
                    _validate_frozen_metadata(
                        metadata,
                        prepared=prepared,
                        cache_file_sha256=cache_file_hash,
                        frozen_cache_sha256=frozen_digest,
                    )
                    producer_provenance = _legacy_producer_provenance(metadata)
                else:
                    raise SemanticCacheValidationError(
                        f"unsupported semantic cache metadata schema: {metadata_path}"
                    )
            elif frozen_digest is None:
                raise SemanticCacheValidationError(
                    f"semantic cache metadata is missing: {metadata_path}"
                )
            else:
                producer_provenance = _unknown_legacy_provenance()
        else:
            if metadata_path.exists():
                raise SemanticCacheValidationError(
                    f"semantic cache image is missing but metadata exists: {metadata_path}"
                )
            predictor = self._get_predictor()
            native, inference_metadata = predictor.predict(
                Image.fromarray(prepared.native_rgb, mode="RGB")
            )
            native = np.asarray(native, dtype=np.uint8)
            _validate_native_shape(native, prepared)
            producer_provenance = _normalize_producer_provenance(
                inference_metadata,
                requested_device=self.requested_device,
                model=self.model_identity,
            )
            _atomic_save_png(native, destination)
            cache_file_hash = file_sha256(destination)
            metadata = _metadata(
                prepared=prepared,
                model=self.model_identity,
                cache_key=cache_key,
                cache_file_sha256=cache_file_hash,
                native=native,
                producer_provenance=producer_provenance,
            )
            _atomic_write_json(metadata_path, metadata)
            self.inference_count += 1
            inference_run = True

        analysis = _resize_nearest(native, prepared.analysis_size)
        return SemanticEvidence(
            class_map=analysis,
            native_size=prepared.original_size,
            analysis_size=prepared.analysis_size,
            native_array_sha256=array_sha256(native),
            analysis_array_sha256=array_sha256(analysis),
            cache_key=cache_key,
            cache_path=destination,
            cache_file_sha256=cache_file_hash,
            cache_hit=cache_hit,
            inference_run=inference_run,
            inference_metadata=inference_metadata,
            cache_reuse_policy=CACHE_REUSE_POLICY,
            requested_device=self.requested_device,
            producer_provenance=producer_provenance,
            model=self.model_identity,
        )

    def _get_predictor(self) -> SemanticPredictor:
        if self._predictor is None:
            with self._predictor_lock:
                if self._predictor is None:
                    self._predictor = self._predictor_factory(
                        self.model_root, self.requested_device
                    )
        return self._predictor


def semantic_cache_key(
    prepared: PreparedImage, model: SemanticModelIdentity
) -> str:
    payload = {
        "source_sha256": prepared.content_sha256,
        "profile_id": model.profile_id,
        "model_tree_sha256": model.model_tree_sha256,
        "class_mapping_sha256": model.class_mapping_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_optional_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("frozen_cache_sha256 must be a 64-character hex digest")
    return normalized


def _writable_cache_destination(
    cache_root: Path, *, profile_id: str, cache_key: str
) -> Path:
    root = cache_root.resolve()
    candidate = (root / profile_id / f"{cache_key}.png").resolve()
    if candidate == root or root not in candidate.parents:
        raise SemanticCacheValidationError(
            "writable semantic cache path escapes cache_root"
        )
    return candidate


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _read_native_map(path: Path) -> tuple[np.ndarray, str]:
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            native = np.asarray(opened)
    except (OSError, UnidentifiedImageError) as exc:
        raise SemanticCacheValidationError(
            f"semantic cache is corrupt: {path}"
        ) from exc
    if native.ndim != 2 or native.dtype != np.uint8:
        raise SemanticCacheValidationError(
            f"semantic cache must be a uint8 single-channel PNG: {path}"
        )
    return native, file_sha256(path)


def _validate_native_shape(native: np.ndarray, prepared: PreparedImage) -> None:
    expected = (prepared.original_size[1], prepared.original_size[0])
    if native.shape != expected:
        raise SemanticCacheValidationError(
            f"semantic cache shape {native.shape!r} != source shape {expected!r}"
        )


def _metadata(
    *,
    prepared: PreparedImage,
    model: SemanticModelIdentity,
    cache_key: str,
    cache_file_sha256: str,
    native: np.ndarray,
    producer_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "source_sha256": prepared.content_sha256,
        "native_size": list(prepared.original_size),
        "native_array_sha256": array_sha256(native),
        "cache_file_sha256": cache_file_sha256,
        "cache_reuse_policy": CACHE_REUSE_POLICY,
        "producer_provenance": producer_provenance,
        "model": asdict(model),
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticCacheValidationError(
            f"semantic cache metadata is corrupt: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SemanticCacheValidationError(
            f"semantic cache metadata must be an object: {path}"
        )
    return value


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    prepared: PreparedImage,
    model: SemanticModelIdentity,
    cache_key: str,
    cache_file_sha256: str,
    native_array_sha256: str,
) -> dict[str, Any]:
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "source_sha256": prepared.content_sha256,
        "native_size": list(prepared.original_size),
        "native_array_sha256": native_array_sha256,
        "cache_file_sha256": cache_file_sha256,
        "cache_reuse_policy": CACHE_REUSE_POLICY,
        "model": asdict(model),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise SemanticCacheValidationError(
            "semantic cache metadata is stale: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return _validate_v2_producer_provenance(
        metadata.get("producer_provenance"), model=model
    )


def _validate_v2_producer_provenance(
    producer: Any, *, model: SemanticModelIdentity
) -> dict[str, Any]:
    if not isinstance(producer, dict):
        raise SemanticCacheValidationError(
            "semantic cache producer provenance is missing or invalid"
        )
    required = {
        "requested_device",
        "resolved_device",
        "torch_version",
        "transformers_version",
        "preprocessing_identity",
        "postprocessing_identity",
        "tensor_shape",
        "peak_cuda_bytes",
        "seconds",
    }
    missing = sorted(required.difference(producer))
    if missing:
        raise SemanticCacheValidationError(
            "semantic cache producer provenance is incomplete: "
            + ", ".join(missing)
        )
    expected_producer = {
        "preprocessing_identity": f"{model.profile_id}:{model.preprocessor_sha256}",
        "postprocessing_identity": POSTPROCESSING_IDENTITY,
    }
    provenance_mismatches = {
        key: {"expected": value, "actual": producer.get(key)}
        for key, value in expected_producer.items()
        if producer.get(key) != value
    }
    for key in (
        "requested_device",
        "resolved_device",
        "torch_version",
        "transformers_version",
    ):
        if not isinstance(producer.get(key), str) or not producer[key].strip():
            provenance_mismatches[key] = {
                "expected": "non-empty string",
                "actual": producer.get(key),
            }
    if provenance_mismatches:
        raise SemanticCacheValidationError(
            "semantic cache producer provenance is stale: "
            + json.dumps(
                provenance_mismatches, ensure_ascii=False, sort_keys=True
            )
        )
    tensor_shape = producer["tensor_shape"]
    if (
        not isinstance(tensor_shape, list)
        or not tensor_shape
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in tensor_shape
        )
    ):
        raise SemanticCacheValidationError(
            "semantic cache producer tensor_shape must be a non-empty list "
            "of positive integers"
        )
    peak_cuda_bytes = producer["peak_cuda_bytes"]
    if peak_cuda_bytes is not None and (
        isinstance(peak_cuda_bytes, bool)
        or not isinstance(peak_cuda_bytes, int)
        or peak_cuda_bytes < 0
    ):
        raise SemanticCacheValidationError(
            "semantic cache producer peak_cuda_bytes must be None or a "
            "non-negative integer"
        )
    seconds = producer["seconds"]
    if seconds is not None and (
        isinstance(seconds, bool)
        or not isinstance(seconds, Real)
        or not math.isfinite(float(seconds))
        or seconds < 0
    ):
        raise SemanticCacheValidationError(
            "semantic cache producer seconds must be None or a finite "
            "non-negative real number"
        )
    return producer


def _normalize_producer_provenance(
    inference_metadata: dict[str, Any],
    *,
    requested_device: str,
    model: SemanticModelIdentity,
) -> dict[str, Any]:
    producer: dict[str, Any] = {
        "requested_device": requested_device,
        "preprocessing_identity": (
            f"{model.profile_id}:{model.preprocessor_sha256}"
        ),
        "postprocessing_identity": POSTPROCESSING_IDENTITY,
    }
    source_keys = {
        "resolved_device": "device",
        "torch_version": "torch_version",
        "transformers_version": "transformers_version",
        "tensor_shape": "tensor_shape",
        "peak_cuda_bytes": "peak_cuda_bytes",
        "seconds": "seconds",
    }
    for target, source in source_keys.items():
        if source in inference_metadata:
            producer[target] = inference_metadata[source]
    return _validate_v2_producer_provenance(producer, model=model)


def _legacy_producer_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    device = metadata.get("device")
    return {
        "source": "legacy_metadata",
        "requested_device": device,
        "resolved_device": device,
        "torch_version": metadata.get("torch_version"),
        "transformers_version": metadata.get("transformers_version"),
        "preprocessing_identity": metadata.get("preprocessing"),
        "postprocessing_identity": "legacy_native_restore",
        "tensor_shape": metadata.get("tensor_shape"),
        "peak_cuda_bytes": metadata.get("peak_cuda_bytes"),
        "seconds": metadata.get("seconds"),
    }


def _unknown_legacy_provenance() -> dict[str, Any]:
    return {
        "source": "legacy_hash_only",
        "requested_device": None,
        "resolved_device": None,
        "torch_version": None,
        "transformers_version": None,
        "preprocessing_identity": None,
        "postprocessing_identity": None,
        "tensor_shape": None,
        "peak_cuda_bytes": None,
        "seconds": None,
    }


def _validate_frozen_metadata(
    metadata: dict[str, Any],
    *,
    prepared: PreparedImage,
    cache_file_sha256: str,
    frozen_cache_sha256: str,
) -> None:
    expected = {
        "model_key": "mask2former_mapillary_swin_l",
        "preprocessing": "local_processor_default",
        "source_sha256": prepared.content_sha256,
        "source_size": list(prepared.original_size),
        "output_sha256": frozen_cache_sha256.lower(),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if cache_file_sha256 != frozen_cache_sha256.lower():
        mismatches["cache_file_sha256"] = {
            "expected": frozen_cache_sha256.lower(),
            "actual": cache_file_sha256,
        }
    if mismatches:
        raise SemanticCacheValidationError(
            "frozen semantic cache metadata is stale: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _resize_nearest(
    native: np.ndarray, analysis_size: tuple[int, int]
) -> np.ndarray:
    import cv2

    return cv2.resize(
        native.astype(np.uint8),
        analysis_size,
        interpolation=cv2.INTER_NEAREST,
    )


def _atomic_save_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(array.astype(np.uint8)).save(temporary, optimize=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

