# Dependency profiles

Use `setup.bat` from the repository root. It prompts for one of two profiles
and creates a separate Python 3.12 environment so CPU/GPU wheels cannot leak
across profiles:

- CPU only: `torch-cpu.txt` from the official PyTorch CPU index.
- NVIDIA notebook GPU: `torch-gpu-cu128.txt` from the CUDA 12.8 index.

The resulting environments are `.venv-cpu` and `.venv-gpu-cu128`.

After the selected torch profile, setup installs `inference-common.txt` and
`review.txt` from the normal Python package index. Keeping the torch index on a
separate command prevents it from becoming the index for unrelated packages.

The GPU profile targets both the RTX 4060 Laptop GPU (Ada) and RTX 5070 Ti
Laptop GPU (Blackwell). Blackwell first requires CUDA 12.8 support. The wheel
ships its CUDA runtime, so a separate CUDA Toolkit installation is not
required; a sufficiently current notebook NVIDIA/OEM driver is required.
CUDA 12.8 GA corresponds to Windows driver 570.65 or newer. Setup verifies
`nvidia-smi`, `torch.cuda.is_available()`, CUDA runtime 12.8, and the detected
GPU name.

The direct dependency pins match the accepted development baseline. Before the
release commit, run clean installs on CPU, RTX 4060 Laptop, and RTX 5070 Ti
Laptop environments, then generate OS/Python-specific transitive lock files.

Do not put model weights in requirements. Record source, license, version, and
SHA-256 in `manifests/model_assets_manifest.json`.
