#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LAUNCH=0
SKIP_MODEL=0

for arg in "$@"; do
  case "$arg" in
    --launch)
      LAUNCH=1
      ;;
    --skip-model)
      SKIP_MODEL=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/colab_bootstrap.sh [--launch] [--skip-model]

Sets up RC Stable Audio Tools on a fresh Colab GPU runtime:
  - verifies CUDA is available
  - installs cloud-compatible dependencies
  - installs Basic Pitch MIDI extras without triggering TensorFlow conflicts
  - installs this repo in editable mode
  - downloads RoyalCities/Foundation-1 unless --skip-model is passed
  - launches Gradio when --launch is passed
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

echo "==> Checking CUDA runtime"
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected. In Colab, use Runtime > Change runtime type > T4 GPU, then restart.")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "==> Installing base cloud dependencies"
python -m pip install -U "setuptools<81" wheel
python -m pip install -r requirements-cloud.txt -c constraints-cloud.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo "==> Installing Basic Pitch MIDI backend"
python -m pip install "basic-pitch==0.4.0" --no-deps
python -m pip install "onnxruntime>=1.18,<1.21" "mir_eval>=0.8,<0.9" "resampy==0.4.2"

echo "==> Installing RC Stable Audio Tools editable package"
python -m pip install -e . --no-build-isolation --no-deps

echo "==> Verifying imports"
python - <<'PY'
import torch
import aeiou
import gradio
import stable_audio_tools
from basic_pitch.inference import predict

print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("imports ok")
PY

if [[ "$SKIP_MODEL" -eq 0 ]]; then
  echo "==> Downloading Foundation-1 model"
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RoyalCities/Foundation-1",
    local_dir="models/RoyalCities-Foundation-1",
)
PY
else
  echo "==> Skipping model download"
fi

if [[ "$LAUNCH" -eq 1 ]]; then
  echo "==> Launching Gradio share link"
  exec python run_gradio.py --share
fi

echo "==> Setup complete. Launch with: python run_gradio.py --share"
