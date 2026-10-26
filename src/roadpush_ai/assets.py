"""Verify required model files before service startup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    from surface_passability.config.model_profiles import (
        tree_sha256,
        verify_mask2former_mapillary_model,
    )

    root = Path(os.environ.get("ROADPUSH_ROOT", Path(__file__).resolve().parents[2])).resolve()
    model_root = Path(os.environ.get("ROADPUSH_MODEL_ROOT", root / "models")).resolve()
    manifest = json.loads((root / "models" / "model_assets.json").read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        path = model_root / asset["relative_path"]
        if not path.exists():
            raise FileNotFoundError(path)
        expected = asset.get("sha256")
        if expected:
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"model hash mismatch for {asset['asset_id']}: {actual} != {expected}"
                )
        expected_tree = asset.get("tree_sha256")
        if expected_tree:
            actual_tree = tree_sha256(path)
            if actual_tree != expected_tree:
                raise ValueError(
                    f"model tree hash mismatch for {asset['asset_id']}: "
                    f"{actual_tree} != {expected_tree}"
                )
        print(f"OK {asset['asset_id']}: {path}")

    verify_mask2former_mapillary_model(model_root / "surface_passability" / "semantic")
    print("All required model assets verified.")
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
