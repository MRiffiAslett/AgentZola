#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv-cp310"
WHEEL="/vol/bitbucket/<user>/tfbuild/wheels/tensorflow_cpu-2.14.0+selfbuilt.20230507-cp310-cp310-linux_x86_64.whl"
echo "[$(date)] PROJECT_ROOT : $PROJECT_ROOT"
echo "[$(date)] VENV_DIR     : $VENV_DIR"
echo "[$(date)] WHEEL        : $WHEEL"

if [ ! -f "$WHEEL" ]; then
  echo "ERROR: wheel not found: $WHEEL" >&2
  exit 1
fi

PY310="$(command -v python3.10 2>/dev/null || true)"
if [ -z "$PY310" ]; then
  echo "ERROR: python3.10 not found in PATH" >&2
  exit 1
fi
echo "[$(date)] Python 3.10  : $PY310 ($($PY310 --version))"

echo "[$(date)] Creating venv at $VENV_DIR"
"$PY310" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel

pip install \
  "pydantic>=2.12.5,<3.0" \
  "tomli>=2.3.0,<3.0" \
  "vllm>=0.12.0,<0.13.0" \
  "astunparse>=1.6.3" \
  "psutil>=6.1.1"

pip install \
  "numpy>=1.24,<2.0" \
  "protobuf>=3.20.3,<5.0" \
  "keras>=2.13.1,<2.14" \
  "tensorflow-estimator>=2.13.0,<2.14" \
  "gast<=0.4.0" \
  "wrapt<1.15" \
  absl-py \
  libclang

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
unzip -o "$WHEEL" -d "$SITE"

echo "[$(date)] venv-cp310 setup complete."
echo "[$(date)] Python: $(python --version)"
echo "[$(date)] TF: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
