# step0_env_check.py
# 목적: YOLO 학습에 필요한 환경(Python, GPU, PyTorch, CUDA, ultralytics)이 준비됐는지 확인한다.
# 데이터 파일은 전혀 읽지 않는다.
import platform
import shutil
import subprocess
import sys

print("Python 실행 파일:", sys.executable)
print("Python 버전:", platform.python_version())

# 1) NVIDIA GPU와 드라이버: nvidia-smi 명령이 있으면 GPU 이름, 메모리, 드라이버 버전을 묻는다
if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True)
    print("GPU:", out.stdout.strip())
else:
    print("GPU: nvidia-smi 없음 (NVIDIA GPU가 없거나 드라이버 미설치)")

# 2) PyTorch: 설치돼 있으면 버전과 GPU 사용 가능 여부를 출력한다
try:
    import torch
    print("PyTorch 버전:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("PyTorch가 보는 GPU:", torch.cuda.get_device_name(0), "/ CUDA", torch.version.cuda)
except ImportError:
    print("PyTorch: 설치 안 됨")

# 3) ultralytics(YOLO): 설치 여부만 확인한다
try:
    import ultralytics
    print("ultralytics 버전:", ultralytics.__version__)
except ImportError:
    print("ultralytics: 설치 안 됨")
