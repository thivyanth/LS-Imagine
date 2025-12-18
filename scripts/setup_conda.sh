# Save as: scripts/setup_conda.sh
set -euo pipefail

ENV_NAME=${ENV_NAME:-ls}
PY_VER=${PY_VER:-3.9}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# Ensure 'conda' shell functions are available
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Conda env '$ENV_NAME' already exists. Skipping creation."
else
  echo "[INFO] Creating conda env '$ENV_NAME' with python=$PY_VER"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

echo "[INFO] Activating env '$ENV_NAME'"
conda activate "$ENV_NAME"

echo "[INFO] Installing PyTorch (CUDA 12.1)"
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Pinning build tooling for legacy gym (0.21.0)"
pip install --upgrade pip==23.2.1
pip install "setuptools<60" "wheel<0.41.0"

echo "[INFO] Pre-installing gym==0.21.0 to avoid build metadata issues"
pip install gym==0.21.0 || true

echo "[INFO] Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

echo "[INFO] Attempting to install MineDojo (optional; see docs if it fails)"
pip install minedojo || echo "[WARN] minedojo install encountered issues. See docs/minedojo_installation.md"

echo "[INFO] Creating weights directory if missing"
mkdir -p weights

cat <<EON
[NEXT STEPS]
- Install Java JDK 1.8 (see docs/minedojo_installation.md)
- Download MineCLIP weights to ./weights/mineclip_attn.pth
- Configure wandb in ./config.yaml if desired
EON

echo "[DONE] Environment setup complete. Use: 'conda activate $ENV_NAME'"
# ```

# Run it:

# ```bash
# chmod +x scripts/setup_conda.sh
# ./scripts/setup_conda.sh
# ```






# apt-get update && apt-get install -y xvfb
# cd /workspace/ && git clone https://github.com/verityw/MixinGradle-dcfaf61
