#!/usr/bin/env bash
# one-time environment setup on the Chameleon A30 node for the publication-grade generality sweep.
# creates a venv, installs the CUDA torch + diffusion/eval stack, caches models on the big local disk.
set -eu
export HF_HOME="$HOME/hf_cache" HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"
cd "$HOME"

if [ ! -d gvenv ]; then python3 -m venv gvenv; fi
# shellcheck disable=SC1091
source gvenv/bin/activate
pip install -q --upgrade pip wheel
# torch cuda build is the default linux wheel; pin diffusers to the version used locally
pip install -q torch torchvision
pip install -q "diffusers==0.40.0" transformers accelerate cleanfid opencv-python-headless \
  matplotlib huggingface_hub scikit-image safetensors

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
PY
echo "NODE_SETUP_DONE"
