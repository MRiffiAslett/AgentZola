#!/usr/bin/env bash
# =============================================================================
# One-time setup: create the Python 3.10 venv for the 20230507 TF wheel.
# Run this once from the cluster login node before submitting jobs with
# WHITEFOX_WHEEL_VERSION=20230507.
#
# Usage:
#   bash WhiteFox/slurm/tfxla/setup_venv_20230507.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv-cp310"
WHEEL="/vol/bitbucket/mtr25/tfbuild/wheels/tensorflow_cpu-2.14.0+selfbuilt.20230507-cp310-cp310-linux_x86_64.whl"
PYPROJECT="$PROJECT_ROOT/pyproject-20230507.toml"

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

# Install dependencies from the 20230507 pyproject using poetry
# We swap pyproject.toml temporarily so poetry uses the right spec.
cp "$PROJECT_ROOT/pyproject.toml" "$PROJECT_ROOT/pyproject.toml.bak"
cp "$PYPROJECT" "$PROJECT_ROOT/pyproject.toml"

trap 'mv "$PROJECT_ROOT/pyproject.toml.bak" "$PROJECT_ROOT/pyproject.toml"' EXIT

poetry config virtualenvs.create false
poetry install --no-interaction

# Force-reinstall the TF wheel last to ensure it takes precedence.
pip install --force-reinstall --no-deps "$WHEEL"

echo "[$(date)] venv-cp310 setup complete."
echo "[$(date)] Python: $(python --version)"
echo "[$(date)] TF: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
