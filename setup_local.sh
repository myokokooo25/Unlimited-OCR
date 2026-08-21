#!/usr/bin/env bash
# Local NVIDIA GPU setup for Unlimited-OCR (Linux).
set -euo pipefail
cd "$(dirname "$0")"

echo "== Unlimited-OCR local setup =="
echo "[1/5] Checking NVIDIA driver..."
nvidia-smi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
echo "[2/5] Using $PYTHON_BIN"

if [[ ! -d .venv ]]; then
  echo "[3/5] Creating .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "[3/5] Reusing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
echo "[4/5] Installing PyTorch with CUDA..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "[5/5] Installing Unlimited-OCR Python deps..."
python -m pip install -r requirements-local.txt

echo
echo "Setup finished. Next commands:"
echo "  source .venv/bin/activate"
echo "  python infer_local.py --check-gpu"
echo "  python infer_local.py --pdf ./Unlimited-OCR.pdf --output_dir ./outputs --max_pages 1"
echo
echo "First run downloads baidu/Unlimited-OCR from Hugging Face (several GB)."
