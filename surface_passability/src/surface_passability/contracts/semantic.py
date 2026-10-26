"""Typed semantic-model identity and evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SemanticModelIdentity:
    profile_id: str
    upstream_model_id: str
    model_tree_sha256: str
    weights_sha256: str
    config_sha256: str
    preprocessor_sha256: str
    class_mapping_sha256: str
    class_count: int


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    class_map: NDArray[np.uint8]
    native_size: tuple[int, int]
    analysis_size: tuple[int, int]
    native_array_sha256: str
    analysis_array_sha256: str
    cache_key: str
    cache_path: Path
    cache_file_sha256: str
    cache_hit: bool
    inference_run: bool
    inference_metadata: dict[str, Any] | None
    cache_reuse_policy: str
    requested_device: str
    producer_provenance: dict[str, Any]
    model: SemanticModelIdentity

    def __post_init__(self) -> None:
        if self.class_map.dtype != np.uint8:
            raise TypeError("SemanticEvidence.class_map must use uint8")
        expected_shape = (self.analysis_size[1], self.analysis_size[0])
        if self.class_map.shape != expected_shape:
            raise ValueError(
                f"semantic class-map shape {self.class_map.shape!r} "
                f"!= {expected_shape!r}"
            )

