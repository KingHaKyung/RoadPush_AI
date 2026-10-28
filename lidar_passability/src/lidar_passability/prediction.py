"""Opt-in adapter to a separately supplied, hash-pinned Linux GPU runtime.

No Torch import, installation, download, GPU allocation or process launch occurs
at import time or during plan(). Frozen runtime/model assets remain outside Git.
"""

from dataclasses import dataclass
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from .contracts import ContractError, require_hash, sha256


@dataclass(frozen=True)
class FrozenRuntime:
    python: Path
    source_root: Path
    checkpoint: Path
    expected_checkpoint_sha256: str
    profile: str = "small_b_cu124"

    def receipt(self):
        if self.profile not in ("small_b_cu124", "team271_cu126"):
            raise ContractError("Unknown frozen runtime profile")
        return json.loads(
            files("lidar_passability")
            .joinpath("profiles", self.profile + ".json")
            .read_text(encoding="utf8")
        )

    def plan(self, pcd_path, output, *, expected_pcd_sha256, max_seconds=120):
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 120
        ):
            raise ContractError(
                "Explicit inference limit must be finite and in (0,120] seconds"
            )
        pcd = Path(pcd_path).resolve()
        output = Path(output).resolve()
        root = Path(self.source_root).resolve()
        if not Path(self.python).is_file():
            raise ContractError("Existing runtime Python required")
        if output.exists():
            raise FileExistsError("Existing inference artifacts are preserved")
        if sha256(pcd) != require_hash(expected_pcd_sha256):
            raise ContractError("PCD hash mismatch")
        if sha256(self.checkpoint) != require_hash(self.expected_checkpoint_sha256):
            raise ContractError("Checkpoint hash mismatch")
        receipt = self.receipt()
        fixed = receipt.get("checkpoint_sha256")
        if fixed and fixed != self.expected_checkpoint_sha256:
            raise ContractError(
                "Checkpoint does not belong to selected development profile"
            )
        for relative, expected in receipt["source_sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256(path) != expected:
                raise ContractError("Frozen runtime source mismatch: " + relative)
        command = [
            str(Path(self.python).resolve()),
            str(root / "litept_external/predict.py"),
            "--pcd",
            str(pcd),
            "--checkpoint",
            str(Path(self.checkpoint).resolve()),
            "--output",
            str(output),
            "--coverage",
            "rotating5cm",
            "--pointrope",
            "cuda",
            "--max-seconds",
            str(max_seconds),
            "--execute-gpu",
        ]
        return dict(
            command=command,
            cwd=str(root),
            output=str(output),
            max_seconds=max_seconds,
            source_receipt_sha256=sha256_resource(self.profile),
            execute=False,
        )

    def export(
        self, pcd_path, output, *, expected_pcd_sha256, max_seconds=120, execute=False
    ):
        plan = self.plan(
            pcd_path,
            output,
            expected_pcd_sha256=expected_pcd_sha256,
            max_seconds=max_seconds,
        )
        if execute is not True:
            return plan
        if sys.platform != "linux":
            raise ContractError("GPU execution requires the existing Linux runtime")
        busy = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        if busy:
            raise ContractError(
                "GPU has an active compute process; no other work is interrupted"
            )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(
            PYTHONNOUSERSITE="1",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            TMPDIR="/tmp",
        )
        # Capture a bounded diagnostic log; the direct frozen predictor owns no server.
        completed = subprocess.run(
            plan["command"],
            cwd=plan["cwd"],
            env=env,
            timeout=max_seconds + 5,
            check=True,
            capture_output=True,
            text=True,
        )
        out = Path(plan["output"])
        report = json.loads((out / "result.json").read_text(encoding="utf8"))
        if (
            report.get("status") != "inference_export_complete"
            or report.get("pcd_sha256") != expected_pcd_sha256
            or report.get("checkpoint_sha256") != self.expected_checkpoint_sha256
        ):
            raise ContractError("Incomplete or mismatched prediction export")
        return dict(
            prediction_path=out / "prediction.npz",
            report_path=out / "result.json",
            source_receipt_sha256=plan["source_receipt_sha256"],
            process_log=completed.stdout,
        )


def sha256_resource(profile):
    import hashlib

    return hashlib.sha256(
        files("lidar_passability").joinpath("profiles", profile + ".json").read_bytes()
    ).hexdigest()
