#!/usr/bin/env bash
# Reproduce the e2ecl conda environment from scratch.
# Run this from the repo root (e2e-cl-planner/).
set -euo pipefail

ENV_NAME="e2ecl"

echo "Creating conda env '${ENV_NAME}' with Python 3.11..."
conda create -n "${ENV_NAME}" python=3.11 -y

echo "Cloning MetaDrive and ScenarioNet from source..."
git clone --depth 1 https://github.com/metadriverse/metadrive.git
conda run -n "${ENV_NAME}" pip install -e ./metadrive

git clone --depth 1 https://github.com/metadriverse/scenarionet.git
conda run -n "${ENV_NAME}" pip install -e ./scenarionet

echo "Installing PyTorch (CUDA 12.8 wheel) and viz deps..."
conda run -n "${ENV_NAME}" pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

conda run -n "${ENV_NAME}" pip install \
    imageio imageio-ffmpeg matplotlib pyyaml numpy pillow

echo "Pulling bundled scenario assets (nuScenes + Waymo mini splits)..."
conda run -n "${ENV_NAME}" python -m metadrive.pull_asset

echo "Verifying GPU visibility..."
conda run -n "${ENV_NAME}" python -c "
import torch
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Done. Activate with: conda activate ${ENV_NAME}"
