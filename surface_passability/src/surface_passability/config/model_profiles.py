"""Accepted semantic model identity and on-disk verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from surface_passability.contracts.semantic import SemanticModelIdentity


MASK2FORMER_MAPILLARY_PROFILE_ID = "mask2former_mapillary_swin_large"
MASK2FORMER_MAPILLARY_UPSTREAM_ID = (
    "facebook/mask2former-swin-large-mapillary-vistas-semantic"
)
MASK2FORMER_MAPILLARY_TREE_SHA256 = (
    "54eabd2be34b27ebc5c051ce6c7606ee1acc66ae045f0b48b859f687d90eabb1"
)
MASK2FORMER_MAPILLARY_WEIGHTS_SHA256 = (
    "72721afb6894c0b1239b9259c428996d9ca0aecbebc4efb4a2deaae0403e2ea7"
)
MASK2FORMER_MAPILLARY_CONFIG_SHA256 = (
    "b40bec8d82de3b53ef967dce0496c37de520b69466f1a8d2f6c035700cbae9f3"
)
MASK2FORMER_MAPILLARY_PREPROCESSOR_SHA256 = (
    "a76f8cc9f21f1e4152d8824338d9231d823c2426eedf06b77fa6790889206987"
)
MASK2FORMER_MAPILLARY_CLASS_MAPPING_SHA256 = (
    "a7d797bc565b7366659fe82f6d442d6cf004b6c4cada1cfe499064432ae68a93"
)
MASK2FORMER_MAPILLARY_CLASS_COUNT = 65


def accepted_mask2former_mapillary_identity() -> SemanticModelIdentity:
    return SemanticModelIdentity(
        profile_id=MASK2FORMER_MAPILLARY_PROFILE_ID,
        upstream_model_id=MASK2FORMER_MAPILLARY_UPSTREAM_ID,
        model_tree_sha256=MASK2FORMER_MAPILLARY_TREE_SHA256,
        weights_sha256=MASK2FORMER_MAPILLARY_WEIGHTS_SHA256,
        config_sha256=MASK2FORMER_MAPILLARY_CONFIG_SHA256,
        preprocessor_sha256=MASK2FORMER_MAPILLARY_PREPROCESSOR_SHA256,
        class_mapping_sha256=MASK2FORMER_MAPILLARY_CLASS_MAPPING_SHA256,
        class_count=MASK2FORMER_MAPILLARY_CLASS_COUNT,
    )


def verify_mask2former_mapillary_model(
    model_root: Path,
) -> SemanticModelIdentity:
    root = model_root.resolve()
    identity = accepted_mask2former_mapillary_identity()
    required = {
        "model.safetensors": identity.weights_sha256,
        "config.json": identity.config_sha256,
        "preprocessor_config.json": identity.preprocessor_sha256,
    }
    for filename, expected in required.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"semantic model asset hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mapping = {
        "id2label": config.get("id2label"),
        "label2id": config.get("label2id"),
    }
    mapping_bytes = json.dumps(
        mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
    if mapping_hash != identity.class_mapping_sha256:
        raise ValueError("semantic class mapping identity mismatch")
    if len(config.get("id2label", {})) != identity.class_count:
        raise ValueError("semantic class count mismatch")

    tree_hash = tree_sha256(root, known_hashes=required)
    if tree_hash != identity.model_tree_sha256:
        raise ValueError(
            "semantic model tree identity mismatch: "
            f"expected {identity.model_tree_sha256}, got {tree_hash}"
        )
    return identity


def load_mask2former_mapillary_class_names(
    model_root: Path,
    *,
    verified_identity: SemanticModelIdentity | None = None,
) -> dict[int, str]:
    """Load the accepted class mapping after verifying its owning model."""

    root = model_root.resolve()
    identity = verified_identity or verify_mask2former_mapillary_model(root)
    if identity != accepted_mask2former_mapillary_identity():
        raise ValueError("semantic model identity is not the accepted profile")
    config_path = root / "config.json"
    if file_sha256(config_path) != identity.config_sha256:
        raise ValueError("semantic model config identity changed after verification")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_mapping = config.get("id2label")
    if not isinstance(raw_mapping, dict) or len(raw_mapping) != identity.class_count:
        raise ValueError("semantic class mapping is missing or incomplete")
    mapping = {int(key): str(value) for key, value in raw_mapping.items()}
    mapping_identity = {
        "id2label": config.get("id2label"),
        "label2id": config.get("label2id"),
    }
    mapping_hash = hashlib.sha256(
        json.dumps(
            mapping_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if mapping_hash != identity.class_mapping_sha256:
        raise ValueError("semantic class mapping identity changed after verification")
    return mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(
    root: Path,
    *,
    known_hashes: dict[str, str] | None = None,
) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        file_hash = (
            known_hashes.get(relative)
            if known_hashes is not None
            else None
        )
        digest.update((file_hash or file_sha256(path)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()

