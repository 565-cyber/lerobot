#!/bin/bash
# =============================================================================
# LeIsaac Simulation Environment Setup Script
# =============================================================================
#
# This script sets up the LeIsaac simulation environment for LeRobot.
# It performs the following steps:
#   1. Creates a conda environment with required dependencies
#   2. Installs Isaac Sim, Isaac Lab, LeIsaac, and LeRobot
#   3. Fixes the isaaclab package for proper module imports
#   4. Downloads the LeIsaac assets from HuggingFace Hub
#
# Usage:
#   bash setup_leisaac.sh
#
# After setup, test with:
#   conda activate leisaac_envhub
#   python examples/leisaac/test_env.py
# =============================================================================

set -e

ENV_NAME="leisaac_envhub"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.8.1"

echo "============================================"
echo " LeIsaac Simulation Environment Setup"
echo "============================================"
echo ""
echo "This will create/update the '${ENV_NAME}' conda environment."
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Step 1: Create/verify conda environment ──
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/6] Environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[1/6] Creating conda environment '${ENV_NAME}'..."
    conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
fi

conda activate ${ENV_NAME}

# ── Step 2: Install CUDA toolkit ──
echo "[2/6] Installing CUDA toolkit ${CUDA_VERSION}..."
conda install -y -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit 2>/dev/null || true

# ── Step 3: Install PyTorch ──
echo "[3/6] Installing PyTorch..."
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# ── Step 4: Install LeIsaac + IsaacLab ──
echo "[4/6] Installing LeIsaac with IsaacLab support..."
pip install 'leisaac[isaaclab] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac' --extra-index-url https://pypi.nvidia.com

# ── Step 5: Install LeRobot ──
echo "[5/6] Installing LeRobot..."
pip install lerobot==0.4.1

# ── Step 6: Fix numpy version ──
echo "[6/6] Fixing numpy version..."
pip install numpy==1.26.0

# ── Post-install: Fix isaaclab package structure ──
echo ""
echo "============================================"
echo " Fixing Isaac Lab package structure..."
echo "============================================"

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
ISAACLAB_SOURCE="${SITE_PKG}/isaaclab/source/isaaclab/isaaclab"
ISAACLAB_TASKS_SOURCE="${SITE_PKG}/isaaclab/source/isaaclab_tasks"
ISAACLAB_PKG="${SITE_PKG}/isaaclab"

echo "Site packages: ${SITE_PKG}"

# Create symlinks for isaaclab source submodules
if [ -d "${ISAACLAB_SOURCE}" ]; then
    echo "Creating symlinks for isaaclab submodules..."
    for dir in envs scene sim actuators managers controllers devices utils assets sensors; do
        if [ -d "${ISAACLAB_SOURCE}/${dir}" ] && [ ! -e "${ISAACLAB_PKG}/${dir}" ]; then
            ln -sf "${ISAACLAB_SOURCE}/${dir}" "${ISAACLAB_PKG}/${dir}"
            echo "  Linked: isaaclab/${dir}"
        fi
    done
else
    echo "WARNING: isaaclab source not found at ${ISAACLAB_SOURCE}"
fi

# Install isaaclab_tasks as editable
if [ -d "${ISAACLAB_TASKS_SOURCE}" ]; then
    echo "Installing isaaclab_tasks as editable..."
    pip install -e "${ISAACLAB_TASKS_SOURCE}"
else
    echo "WARNING: isaaclab_tasks source not found at ${ISAACLAB_TASKS_SOURCE}"
fi

# ── Download assets from HuggingFace ──
echo ""
echo "============================================"
echo " Downloading LeIsaac assets (this may take a few minutes)..."
echo "============================================"

# Use HF mirror for China (uncomment if needed)
# export HF_ENDPOINT=https://hf-mirror.com

python -c "
from huggingface_hub import snapshot_download
import os
print('Downloading assets...')
path = snapshot_download(
    'LightwheelAI/leisaac_env',
    cache_dir=os.path.expanduser('~/.cache/huggingface')
)
print(f'Assets downloaded to: {path}')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Environment: ${ENV_NAME}"
echo ""
echo "To activate:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "EULA environment variables (add to ~/.bashrc for persistence):"
echo "  export ACCEPT_EULA=Y"
echo "  export PRIVACY_CONSENT=Y"
echo "  export OMNI_KIT_ACCEPT_EULA=YES"
echo ""
echo "If in China, also set HF mirror:"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo ""
echo "Test your setup:"
echo "  python examples/leisaac/test_env.py"
echo ""
