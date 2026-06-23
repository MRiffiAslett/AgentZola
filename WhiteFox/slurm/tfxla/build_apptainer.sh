#!/usr/bin/env bash
# =============================================================================
# WhiteFox TF/XLA fuzzer — Apptainer image builder.
#
# Run this ONCE from the cluster login node to produce a .sif image that
# tfxla_a100_apptainer.sh will submit to the GPU nodes.
#
# Prerequisites:
#   1. apptainer is in your PATH (run: apptainer --version)
#   2. fakeroot is configured for your account (run: apptainer build --fakeroot
#      and check for "fakeroot: not configured" — if so, ask CSG to enable it)
#   3. The custom TF wheel exists at WHITEFOX_TF_WHEEL_PATH (see REPRODUCIBILITY.md)
#   4. An HF token in $HF_TOKEN or ~/.hf_token (only needed at run time, not build)
#
# Usage:
#   # wheel is already at the default path on Imperial's cluster:
#   bash build_apptainer.sh
#
#   # or point at a different wheel:
#   export WHITEFOX_TF_WHEEL_PATH=/path/to/tensorflow_cpu-...whl
#   bash build_apptainer.sh
#
# Optional knobs:
#   WHITEFOX_SIF_PATH          where to write the .sif  (default: see below)
#   WHITEFOX_APPTAINER_SANDBOX if "1", build a directory sandbox instead of
#                               a .sif (useful if --fakeroot is not available)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHITEFOX_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # .../AgentZola/WhiteFox

if [ ! -f "$WHITEFOX_DIR/pyproject.toml" ]; then
  echo "ERROR: expected WhiteFox project at $WHITEFOX_DIR (no pyproject.toml)" >&2
  exit 1
fi

# ---- Configuration ---------------------------------------------------------
WHEEL_PATH="${WHITEFOX_TF_WHEEL_PATH:-/vol/bitbucket/mtr25/tfbuild/wheels/tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806-cp312-cp312-linux_x86_64.whl}"
SIF_PATH="${WHITEFOX_SIF_PATH:-/vol/bitbucket/mtr25/whitefox-tfxla.sif}"
USE_SANDBOX="${WHITEFOX_APPTAINER_SANDBOX:-0}"

# ---- Sanity checks ---------------------------------------------------------
if ! command -v apptainer >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: apptainer not found in PATH.

On Imperial's GPU cluster this is usually pre-installed. Try:
  module load apptainer    # if modules are in use
  which apptainer
MSG
  exit 1
fi

if [ ! -f "$WHEEL_PATH" ]; then
  cat >&2 <<MSG
ERROR: TF wheel not found at: $WHEEL_PATH

Set WHITEFOX_TF_WHEEL_PATH to the correct path, or see
WhiteFox/slurm/tfxla/REPRODUCIBILITY.md for how to obtain the wheel.
MSG
  exit 1
fi

echo "[$(date)] apptainer version: $(apptainer --version)"
echo "[$(date)] Wheel: $WHEEL_PATH ($(du -h "$WHEEL_PATH" | cut -f1))"
echo "[$(date)] Output: $SIF_PATH"
echo

# ---- Stage build context ---------------------------------------------------
STAGE_DIR="$(mktemp -d -t whitefox-apptainer-XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "[$(date)] Staging build context at $STAGE_DIR"

# Copy WhiteFox source (rsync preferred for speed + excludes).
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'logging/' \
    --exclude 'output/' \
    --exclude 'hf_cache/' \
    --exclude 'wf_profraw_*/' \
    --exclude 'default.profraw' \
    --exclude 'slurm/output_*' \
    --exclude '*.pyc' \
    "$WHITEFOX_DIR/" "$STAGE_DIR/whitefox/"
else
  cp -a "$WHITEFOX_DIR" "$STAGE_DIR/whitefox"
  find "$STAGE_DIR/whitefox" \( \
       -name '__pycache__' -o -name '.git' -o -name '.venv' \
    -o -name 'logging' -o -name 'output' -o -name 'hf_cache' \
    -o -name 'wf_profraw_*' \
    \) -prune -exec rm -rf {} +
  find "$STAGE_DIR/whitefox" -name '*.pyc' -delete
fi

# Drop the existing poetry.lock — it contains the Imperial-specific wheel path
# and will fail validation inside the image. We re-lock after the sed patch.
rm -f "$STAGE_DIR/whitefox/poetry.lock"

# ---- Stage TF wheel --------------------------------------------------------
echo "[$(date)] Copying TF wheel to build context"
cp "$WHEEL_PATH" "$STAGE_DIR/tensorflow.whl"

# ---- Write Apptainer definition file ---------------------------------------
# Paths in %files are relative to the directory containing the .def file,
# which is STAGE_DIR — so "tensorflow.whl" and "whitefox" resolve correctly.
cat > "$STAGE_DIR/whitefox.def" <<'DEF_EOF'
Bootstrap: docker
From: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

%labels
    Maintainer WhiteFox
    Description WhiteFox TF/XLA fuzzer with LLVM coverage instrumentation

%files
    tensorflow.whl /opt/wheels/tensorflow.whl
    whitefox /workspace/WhiteFox

%post
    export DEBIAN_FRONTEND=noninteractive
    export PIP_NO_CACHE_DIR=1

    # ---- System deps + Python 3.12 -----------------------------------------
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg xz-utils \
        build-essential git tini && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.12 - && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    apt-get purge -y software-properties-common && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir "poetry>=2.0,<3.0"

    # ---- LLVM 17 (llvm-profdata, llvm-cov) ---------------------------------
    LLVM_VER=17.0.6
    STRIP="clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-22.04"
    mkdir -p /opt/llvm17/bin
    curl -fSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/${STRIP}.tar.xz" \
      | tar -xJf - -C /opt/llvm17 --strip-components=1 \
          "${STRIP}/bin/llvm-profdata" "${STRIP}/bin/llvm-cov"

    # ---- WhiteFox deps -----------------------------------------------------
    cd /workspace/WhiteFox
    # Repoint the path-dep at the in-image wheel location, then re-lock.
    sed -i \
      -e 's|path = "/vol/bitbucket/[^"]*\.whl"|path = "/opt/wheels/tensorflow.whl"|' \
      pyproject.toml
    rm -f poetry.lock
    poetry config virtualenvs.create false
    poetry lock
    poetry install --no-interaction --no-root
    pip install --force-reinstall --no-deps /opt/wheels/tensorflow.whl

    # ---- Runtime directories -----------------------------------------------
    mkdir -p /whitefox-data \
             /workspace/WhiteFox/logging \
             /workspace/WhiteFox/output \
             /workspace/WhiteFox/hf_cache

%environment
    export PATH=/opt/llvm17/bin:${PATH}
    export WHITEFOX_LLVM_DIR=/opt/llvm17/bin
    export PYTHONPATH=/workspace/WhiteFox
    export WHITEFOX_PROFRAW_DIR=/whitefox-data
    export WHITEFOX_PROFRAW_POOL_SIZE=8
    export WHITEFOX_COVERAGE_MERGE_EVERY_ITERS=100
    export WHITEFOX_LLVM_PROFDATA_JOBS=0
    export WHITEFOX_MERGE_BATCH_SIZE=8
    export WHITEFOX_PARALLEL_TEST_WORKERS=4
    export WHITEFOX_TEST_MEM_LIMIT_GB=6
    export WHITEFOX_EARLY_STOP_ITERS=20
    export WHITEFOX_LOGGING_DIR=/workspace/WhiteFox/logging/batch_apptainer
    export HF_HOME=/workspace/WhiteFox/hf_cache
    export HUGGINGFACE_HUB_CACHE=/workspace/WhiteFox/hf_cache
    export VLLM_CACHE_DIR=/workspace/WhiteFox/hf_cache
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TF_XLA_FLAGS=--tf_xla_auto_jit=2
    export XLA_FLAGS=--xla_dump_to=/tmp/xla_dump

%runscript
    exec python -m generation.main "$@"
DEF_EOF

# ---- Build -----------------------------------------------------------------
if [ "$USE_SANDBOX" = "1" ]; then
  # Sandbox mode: produces a directory instead of a .sif.
  # Use this if --fakeroot is not enabled on your account.
  # The run script handles both .sif and sandbox directories transparently.
  SANDBOX_PATH="${SIF_PATH%.sif}_sandbox"
  echo "[$(date)] Building sandbox (fakeroot not required): $SANDBOX_PATH"
  apptainer build --sandbox "$SANDBOX_PATH" "$STAGE_DIR/whitefox.def"
  echo "[$(date)] Done. Sandbox: $SANDBOX_PATH"
  echo
  echo "Set WHITEFOX_SIF_PATH=$SANDBOX_PATH when submitting:"
  echo "  WHITEFOX_SIF_PATH=$SANDBOX_PATH sbatch WhiteFox/slurm/tfxla/tfxla_a100_apptainer.sh"
else
  echo "[$(date)] Building .sif (requires --fakeroot). If this fails with"
  echo "           'fakeroot: not configured', re-run with:"
  echo "           WHITEFOX_APPTAINER_SANDBOX=1 bash build_apptainer.sh"
  echo
  apptainer build --fakeroot "$SIF_PATH" "$STAGE_DIR/whitefox.def"
  echo "[$(date)] Done: $SIF_PATH ($(du -h "$SIF_PATH" | cut -f1))"
  echo
  echo "Submit the fuzzing job with:"
  echo "  sbatch WhiteFox/slurm/tfxla/tfxla_a100_apptainer.sh"
  echo "or a quick single-opt test:"
  echo "  sbatch WhiteFox/slurm/tfxla/tfxla_a100_apptainer.sh --only-opt HloDce"
fi
