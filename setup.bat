@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo RoadPush_AI environment setup
echo   [1] CPU only
echo   [2] NVIDIA GPU - CUDA 12.8 wheel
echo       Supported targets: RTX 4060 Laptop GPU / RTX 5070 Ti Laptop GPU
echo.
choice /C 12 /N /M "Select 1 or 2: "
if errorlevel 2 goto gpu
if errorlevel 1 goto cpu
goto invalid

:cpu
set "PROFILE=CPU"
set "TORCH_REQ=surface_passability\requirements\torch-cpu.txt"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "VENV=.venv-cpu"
goto create_env

:gpu
set "PROFILE=GPU-CUDA-12.8"
set "TORCH_REQ=surface_passability\requirements\torch-gpu-cu128.txt"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "VENV=.venv-gpu-cu128"
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo ERROR: nvidia-smi was not found. Install a current NVIDIA/OEM notebook driver first.
  exit /b 1
)
echo NVIDIA driver and GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo CUDA 12.8 GA requires a compatible current driver; Windows 570.65 or newer is recommended.
goto create_env

:create_env
if exist "%VENV%\Scripts\python.exe" goto env_ready
where py >nul 2>nul
if errorlevel 1 goto create_with_uv
py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>nul
if errorlevel 1 goto create_with_uv
py -3.12 -m venv "%VENV%"
if errorlevel 1 (
  echo ERROR: Failed to create %VENV% with the Python 3.12 launcher.
  exit /b 1
)
goto env_ready

:create_with_uv
where uv >nul 2>nul
if errorlevel 1 (
  echo ERROR: Neither Python 3.12 through 'py' nor uv was found.
  echo Install 64-bit CPython 3.12 or uv, then retry.
  exit /b 1
)
uv venv --seed --python 3.12 "%VENV%"
if errorlevel 1 (
  echo ERROR: Failed to create %VENV% with uv-managed Python 3.12.
  exit /b 1
)

:env_ready

set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install -r "%TORCH_REQ%" --index-url "%TORCH_INDEX%"
if errorlevel 1 exit /b 1
"%PY%" -m pip install -r "surface_passability\requirements\inference-common.txt"
if errorlevel 1 exit /b 1
"%PY%" -m pip install -e "surface_passability"
if errorlevel 1 exit /b 1

"%PY%" -c "import cv2,numpy,PIL,scipy,torch,torchvision,transformers,ultralytics,surface_passability; print('surface_passability=',surface_passability.__file__); print('torch=',torch.__version__); print('torchvision=',torchvision.__version__); print('cuda_runtime=',torch.version.cuda)"
if errorlevel 1 exit /b 1

if "%PROFILE%"=="CPU" goto verify_cpu
goto verify_gpu

:verify_cpu
"%PY%" -c "import torch; assert torch.version.cuda is None, 'CPU profile installed a CUDA torch wheel'; print('CPU profile verified')"
if errorlevel 1 exit /b 1
goto success

:verify_gpu
"%PY%" -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU profile verified:',torch.cuda.get_device_name(0))"
if errorlevel 1 (
  echo ERROR: CUDA 12.8 PyTorch could not use the notebook GPU. Update the NVIDIA/OEM driver and retry.
  exit /b 1
)
goto success

:invalid
echo ERROR: Invalid selection.
exit /b 1

:success
echo.
echo Setup complete: %PROFILE%
echo Activate with: %VENV%\Scripts\activate.bat
exit /b 0
