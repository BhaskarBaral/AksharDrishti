<#
Enables GPU acceleration for this project's OCR engines (EasyOCR, PaddleOCR)
by reinstalling torch/torchvision with a CUDA build and installing
paddlepaddle-gpu, then installing paddleocr.

Idempotent: skips the torch reinstall if it's already CUDA-enabled.
Safe to re-run. Falls back to CPU paddlepaddle if the GPU wheel isn't
available for this machine's CUDA tag rather than failing outright.

Usage: .\setup_gpu.ps1   (run from the demo/ folder)
#>

$ErrorActionPreference = "Stop"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv found at $venvPython -- create it first: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# CUDA 12.6 wheels are backward-compatible with newer drivers (NVIDIA's
# forward compatibility guarantee) and have the widest wheel availability
# for both torch and paddlepaddle-gpu on Windows -- safer bet than chasing
# the exact driver-reported CUDA version.
$cudaTag = "cu126"

Write-Host "== Checking current torch CUDA status ==" -ForegroundColor Cyan
$torchStatus = & $venvPython -c "
try:
    import torch
    print('CUDA' if torch.cuda.is_available() else 'CPU')
except ImportError:
    print('MISSING')
" 2>$null

if ($torchStatus -eq "CUDA") {
    Write-Host "torch already CUDA-enabled -- skipping reinstall." -ForegroundColor Green
} else {
    Write-Host "Reinstalling torch/torchvision with CUDA ($cudaTag) build..." -ForegroundColor Cyan
    & $venvPython -m pip uninstall -y torch torchvision
    & $venvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cudaTag"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "torch CUDA install failed -- falling back to CPU torch." -ForegroundColor Yellow
        & $venvPython -m pip install torch torchvision
    }
}

Write-Host "== Installing paddlepaddle-gpu ($cudaTag) ==" -ForegroundColor Cyan
& $venvPython -m pip install "paddlepaddle-gpu==3.2.2" -i "https://www.paddlepaddle.org.cn/packages/stable/$cudaTag/"

$paddleGpuInstallFailed = ($LASTEXITCODE -ne 0)

Write-Host "== Installing paddleocr ==" -ForegroundColor Cyan
& $venvPython -m pip install paddleocr

# The GPU wheel can install cleanly and still fail at import time -- pip-installed
# nvidia-cudnn-cu12 DLLs commonly hit "WinError 127" on Windows without the MSVC
# redistributable, or a driver/toolkit mismatch. Detect that here instead of
# leaving a broken paddle install behind: paddleocr is unusable either way until
# `import paddle` itself succeeds, not just GPU init.
$paddleImportOk = & $venvPython -c "
try:
    import paddle
    print('OK')
except Exception as e:
    print('FAIL')
" 2>$null

if ($paddleGpuInstallFailed -or $paddleImportOk -ne "OK") {
    Write-Host "paddlepaddle-gpu is broken on this machine (install failure or DLL load failure at import time)." -ForegroundColor Yellow
    Write-Host "Falling back to CPU paddlepaddle -- PaddleOCR will still work, just without GPU accel." -ForegroundColor Yellow
    & $venvPython -m pip uninstall -y paddlepaddle-gpu nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 `
        nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12 2>$null
    # Pinned to 3.2.2, not latest: paddlepaddle 3.3.0/3.3.1 have a CPU (oneDNN/PIR)
    # regression that raises NotImplementedError on every predict() call, GPU or
    # CPU device -- https://github.com/PaddlePaddle/Paddle/issues/77340
    & $venvPython -m pip install "paddlepaddle==3.2.2"
}

Write-Host "== Smoke test ==" -ForegroundColor Cyan
& $venvPython -c "
import torch
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

try:
    import paddle
    print('paddle:', paddle.__version__, '| compiled with CUDA:', paddle.is_compiled_with_cuda())
except Exception as e:
    print('paddle import failed:', e)
"
