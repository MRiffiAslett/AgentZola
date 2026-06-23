#!/usr/bin/env bash
# =============================================================================
# One-time setup: create the Python 3.10 venv for the 20230507 TF wheel.
# Run this once from the cluster login node before submitting jobs with
# WHITEFOX_WHEEL_VERSION=20230507.
#
# Usage:
#   bash WhiteFox/slurm/tfxla/setup_venv_20230507.sh
#
# If python3.10 is not in PATH, prepend the standalone build:
#   PATH="/vol/bitbucket/mtr25/python/bin:$PATH" \
#     bash WhiteFox/slurm/tfxla/setup_venv_20230507.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv-cp310"
WHEEL="/vol/bitbucket/mtr25/tfbuild/wheels/tensorflow_cpu-2.14.0+selfbuilt.20230507-cp310-cp310-linux_x86_64.whl"
echo "[$(date)] PROJECT_ROOT : $PROJECT_ROOT"
echo "[$(date)] VENV_DIR     : $VENV_DIR"
echo "[$(date)] WHEEL        : $WHEEL"

if [ ! -f "$WHEEL" ]; then
  echo "ERROR: wheel not found: $WHEEL" >&2
  exit 1
fi

# Require Python 3.10
PY310="$(command -v python3.10 2>/dev/null || true)"
if [ -z "$PY310" ]; then
  echo "ERROR: python3.10 not found in PATH" >&2
  exit 1
fi
echo "[$(date)] Python 3.10  : $PY310 ($($PY310 --version))"

# Create venv
echo "[$(date)] Creating venv at $VENV_DIR"
"$PY310" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel

# Install dependencies directly (mirrors pyproject-20230507.toml, no poetry needed).
# vllm and its transitive deps first (brings in torch, numpy 2.x, protobuf 7.x, etc.)
pip install \
  "pydantic>=2.12.5,<3.0" \
  "tomli>=2.3.0,<3.0" \
  "vllm>=0.12.0,<0.13.0" \
  "astunparse>=1.6.3" \
  "psutil>=6.1.1"

# Pin TF 2.14.0 runtime deps — must come AFTER vllm so these overrides win.
# TF 2.14 was compiled against NumPy 1.x and protobuf < 5.0; newer versions
# either fail at import or corrupt the protobuf MessageFactory at init time.
pip install \
  "numpy>=1.24,<2.0" \
  "protobuf>=3.20.3,<5.0" \
  "keras>=2.13.1,<2.14" \
  "tensorflow-estimator>=2.13.0,<2.14" \
  "gast<=0.4.0" \
  "wrapt<1.15" \
  absl-py \
  libclang

# TF wheel: pip >= 23 rejects this wheel because the dist-info inside is named
# 'tensorflow' while the filename says 'tensorflow_cpu'. Bypass by unzipping
# the wheel (a zip archive) directly into site-packages.
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
unzip -o "$WHEEL" -d "$SITE"

echo "[$(date)] venv-cp310 setup complete."
echo "[$(date)] Python: $(python --version)"
echo "[$(date)] TF: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
