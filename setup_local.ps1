# Local NVIDIA GPU setup for Unlimited-OCR (Windows).
# Run in PowerShell from the repo root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_local.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "== Unlimited-OCR local setup =="
Write-Host "`n[1/5] Checking NVIDIA driver..."
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed. Install Game Ready/Studio drivers, then retry."
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$pythonExe = $null
$pythonArgs = @()
if ($pyLauncher) {
    $pythonExe = $pyLauncher.Source
    $pythonArgs = @("-3.12")
} elseif (Get-Command python3.12 -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python3.12).Source
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
} else {
    throw "Python 3.12 is required. Install it from https://www.python.org/downloads/"
}

Write-Host "[2/5] Using $pythonExe $($pythonArgs -join ' ')"
& $pythonExe @pythonArgs --version

if (-not (Test-Path ".venv")) {
    Write-Host "[3/5] Creating .venv ..."
    & $pythonExe @pythonArgs -m venv .venv
} else {
    Write-Host "[3/5] Reusing .venv"
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install -U pip
Write-Host "[4/5] Installing PyTorch with CUDA..."
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
Write-Host "[5/5] Installing Unlimited-OCR Python deps..."
& $venvPython -m pip install -r requirements-local.txt

Write-Host "`nSetup finished. Next commands:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python infer_local.py --check-gpu"
Write-Host "  python infer_local.py --pdf .\Unlimited-OCR.pdf --output_dir .\outputs --max_pages 1"
Write-Host "`nFirst run downloads baidu/Unlimited-OCR from Hugging Face (several GB)."
