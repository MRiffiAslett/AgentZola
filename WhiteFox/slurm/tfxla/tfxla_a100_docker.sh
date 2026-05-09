#!/usr/bin/env bash
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/tfxla/out/whitefox-docker_%j.out
# (the SBATCH directive above is a safety net: this script is NOT meant to be
# submitted via `sbatch` — Imperial's compute nodes don't run a Docker daemon.
# If someone does sbatch it by accident, at least the failure log lands in
# slurm/tfxla/out/ instead of polluting slurm/tfxla/. The script also detects
# SLURM_JOB_ID below and exits early with a clear message.)
# =============================================================================
# WhiteFox TF/XLA fuzzer — standalone Docker reproducibility wrapper.
#
# Mirrors WhiteFox/slurm/tfxla/tfxla_a100_array.sh, but containerised so any
# reproducer with Docker + an NVIDIA GPU can run the experiment without the
# Imperial /vol/bitbucket filesystem.
#
# IMPORTANT — differences vs the SLURM script:
#   * No SLURM array sharding. The production SLURM script splits 50
#     optimizations into 3 array tasks purely to get more *CPU RAM* (≈190 GB
#     per node × 3 nodes); the fuzzer itself only uses one GPU per task for
#     vLLM. Reproducers therefore need ONE suitable GPU, not three. This
#     wrapper runs all 50 optimizations sequentially in one container; on a
#     machine with ≥64 GB RAM that works fine, just slower.
#   * No /data node-local scratch. The profraw merge pool is a tmpfs inside
#     the container at /whitefox-data.
#   * No bind to /vol. CUDA, LLVM 17 and the custom TF wheel are fetched at
#     image-build time (the wheel from a URL the author hosts; LLVM 17 from
#     the public LLVM release page).
#
# See WhiteFox/slurm/tfxla/REPRODUCIBILITY.md for the full reproducibility
# story (TF wheel hosting, HF token, GPU requirement, etc.).
#
# Usage:
#   # 1. obtain the custom TF wheel from the project authors and place it at
#   #    a known absolute path on the host, then point the script at it:
#   export WHITEFOX_TF_WHEEL_PATH=/abs/path/to/tensorflow_cpu-...whl
#
#   # 2. provide an HF token (one of):
#   export HF_TOKEN=hf_xxx          # or place token in ~/.hf_token
#
#   # 3. run:
#   ./tfxla_a100_docker.sh                              # all 50 optimizations
#   ./tfxla_a100_docker.sh --only-opt HloDce,HloCse     # subset
#
# Optional knobs (env vars):
#   WHITEFOX_DOCKER_IMAGE       image tag (default: whitefox-tfxla:latest)
#   WHITEFOX_RESULTS_DIR        host dir for results (default: ./whitefox-results)
#   WHITEFOX_TF_WHEEL_SHA256    optional sha256 to verify the local wheel
#   WHITEFOX_DOCKER_NO_BUILD    if set to 1, skip docker build (use existing image)
#   WHITEFOX_DOCKER_SHELL       if set to 1, drop into bash inside the container
# =============================================================================

set -euo pipefail

# ---- Refuse to run under sbatch --------------------------------------------
# Imperial's GPU compute nodes don't expose a Docker daemon (they use Apptainer
# / Singularity instead), so this script will always fail there. Bail early
# with a clear message rather than silently producing a useless out file.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cat >&2 <<'MSG'
ERROR: tfxla_a100_docker.sh is not a SLURM script and must NOT be submitted
       with sbatch. Run it directly with bash on a host that has Docker +
       NVIDIA Container Toolkit. For the cluster, use tfxla_a100_array.sh.
MSG
  exit 2
fi

# ---- Resolve script + project paths ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHITEFOX_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # .../AgentZola/WhiteFox

if [ ! -f "$WHITEFOX_DIR/pyproject.toml" ]; then
  echo "ERROR: expected WhiteFox project at $WHITEFOX_DIR (no pyproject.toml found)" >&2
  exit 1
fi

# ---- Configuration ---------------------------------------------------------
IMAGE_TAG="${WHITEFOX_DOCKER_IMAGE:-whitefox-tfxla:latest}"
RESULTS_DIR="${WHITEFOX_RESULTS_DIR:-$PWD/whitefox-results}"
WHEEL_PATH="${WHITEFOX_TF_WHEEL_PATH:-}"
WHEEL_SHA256="${WHITEFOX_TF_WHEEL_SHA256:-}"

if [ -z "$WHEEL_PATH" ]; then
  cat >&2 <<'MSG'
ERROR: WHITEFOX_TF_WHEEL_PATH is not set.

Obtain the custom TensorFlow wheel from the project authors (see
WhiteFox/slurm/tfxla/REPRODUCIBILITY.md), place it at an absolute path on
this host, then export:

  export WHITEFOX_TF_WHEEL_PATH=/abs/path/to/tensorflow_cpu-...whl

The wheel is the self-built TensorFlow with LLVM source-based coverage
instrumentation; stock upstream tensorflow will NOT work for coverage runs.
MSG
  exit 1
fi

# ---- GPU sanity check (warn-only) ------------------------------------------
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia\|nvidia-container'; then
    GPU_FLAGS=(--gpus all)
  else
    echo "WARNING: nvidia-smi works but Docker has no NVIDIA runtime configured." >&2
    echo "         Install nvidia-container-toolkit or vLLM/StarCoder will fail." >&2
    GPU_FLAGS=(--gpus all)   # try anyway; docker will reject if truly unsupported
  fi
else
  echo "WARNING: no NVIDIA GPU detected on this host. vLLM will fail at model load." >&2
  echo "         Continuing anyway so image build / dry runs still work." >&2
fi

# ---- HF token --------------------------------------------------------------
HF_TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_ARGS+=(-e "HF_TOKEN=$HF_TOKEN" -e "HUGGINGFACE_HUB_TOKEN=$HF_TOKEN")
elif [ -f "$HOME/.hf_token" ]; then
  _tok="$(tr -d '[:space:]' < "$HOME/.hf_token")"
  HF_TOKEN_ARGS+=(-e "HF_TOKEN=$_tok" -e "HUGGINGFACE_HUB_TOKEN=$_tok")
else
  echo "WARNING: no HF_TOKEN env var and no ~/.hf_token. StarCoder is gated;" >&2
  echo "         the run will fail at vLLM model download without a token." >&2
fi

# ---- Stage build context ---------------------------------------------------
STAGE_DIR="$(mktemp -d -t whitefox-docker-XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "[$(date)] Staging build context at $STAGE_DIR"

# Copy WhiteFox source minus runtime/output directories. We use rsync if
# available (fast + flexible excludes), otherwise fall back to cp + find.
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
  rm -f "$STAGE_DIR/whitefox/default.profraw" 2>/dev/null || true
fi

# Drop existing poetry.lock — its path-dep is the Imperial-only wheel path
# and would fail validation. We re-lock inside the image after the sed patch.
rm -f "$STAGE_DIR/whitefox/poetry.lock"

# ---- Stage the TF wheel ----------------------------------------------------
WHEEL_DST="$STAGE_DIR/tensorflow.whl"
if [ ! -f "$WHEEL_PATH" ]; then
  echo "ERROR: WHITEFOX_TF_WHEEL_PATH=$WHEEL_PATH does not exist" >&2
  exit 1
fi
echo "[$(date)] Copying local TF wheel: $WHEEL_PATH"
cp "$WHEEL_PATH" "$WHEEL_DST"

if [ -n "$WHEEL_SHA256" ]; then
  echo "[$(date)] Verifying wheel sha256"
  echo "$WHEEL_SHA256  $WHEEL_DST" | sha256sum -c -
fi

# ---- Inline Dockerfile -----------------------------------------------------
cat > "$STAGE_DIR/Dockerfile" <<'DOCKERFILE_EOF'
# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- System deps + Python 3.12 (deadsnakes) + pip ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
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

RUN pip install --no-cache-dir "poetry>=2.0,<3.0"

# --- LLVM 17 (llvm-profdata, llvm-cov) for coverage merging ------------------
ARG LLVM_VER=17.0.6
RUN STRIP="clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-22.04" && \
    mkdir -p /opt/llvm17/bin && \
    curl -fSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/${STRIP}.tar.xz" \
      | tar -xJf - -C /opt/llvm17 --strip-components=1 \
        "${STRIP}/bin/llvm-profdata" "${STRIP}/bin/llvm-cov"
ENV PATH=/opt/llvm17/bin:${PATH} \
    WHITEFOX_LLVM_DIR=/opt/llvm17/bin

# --- Custom TF wheel ---------------------------------------------------------
COPY tensorflow.whl /opt/wheels/tensorflow.whl

# --- WhiteFox source ---------------------------------------------------------
WORKDIR /workspace/WhiteFox
COPY whitefox/ /workspace/WhiteFox/

# Repoint the path-dep at the in-image wheel location, then install deps.
# We delete any leftover lock and let poetry re-resolve.
RUN sed -i \
      -e 's|path = "/vol/bitbucket/[^"]*\.whl"|path = "/opt/wheels/tensorflow.whl"|' \
      pyproject.toml && \
    rm -f poetry.lock && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-interaction --no-root && \
    pip install --force-reinstall --no-deps /opt/wheels/tensorflow.whl

# --- Runtime defaults (mirror tfxla_a100_array.sh) ---------------------------
RUN mkdir -p /whitefox-data /workspace/WhiteFox/logging \
             /workspace/WhiteFox/output /workspace/WhiteFox/hf_cache

ENV PYTHONPATH=/workspace/WhiteFox \
    WHITEFOX_PROFRAW_DIR=/whitefox-data \
    WHITEFOX_PROFRAW_POOL_SIZE=8 \
    WHITEFOX_COVERAGE_MERGE_EVERY_ITERS=100 \
    WHITEFOX_LLVM_PROFDATA_JOBS=0 \
    WHITEFOX_MERGE_BATCH_SIZE=8 \
    WHITEFOX_PARALLEL_TEST_WORKERS=4 \
    WHITEFOX_TEST_MEM_LIMIT_GB=6 \
    WHITEFOX_EARLY_STOP_ITERS=20 \
    WHITEFOX_LOGGING_DIR=/workspace/WhiteFox/logging/batch_docker \
    HF_HOME=/workspace/WhiteFox/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/WhiteFox/hf_cache \
    VLLM_CACHE_DIR=/workspace/WhiteFox/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TF_XLA_FLAGS=--tf_xla_auto_jit=2 \
    XLA_FLAGS=--xla_dump_to=/tmp/xla_dump

ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "generation.main"]
CMD ["--sut", "xla", "--config", "xilo_xla/config/generator.toml"]
DOCKERFILE_EOF

# ---- Build -----------------------------------------------------------------
if [ "${WHITEFOX_DOCKER_NO_BUILD:-0}" = "1" ]; then
  echo "[$(date)] Skipping docker build (WHITEFOX_DOCKER_NO_BUILD=1); reusing $IMAGE_TAG"
else
  echo "[$(date)] Building image: $IMAGE_TAG"
  docker build -t "$IMAGE_TAG" "$STAGE_DIR"
fi

# ---- Prepare host result dirs ----------------------------------------------
mkdir -p "$RESULTS_DIR/logging" "$RESULTS_DIR/output" "$RESULTS_DIR/hf_cache"

# ---- Run -------------------------------------------------------------------
TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

RUN_CMD=(
  docker run --rm
  "${TTY_ARGS[@]}"
  "${GPU_FLAGS[@]}"
  "${HF_TOKEN_ARGS[@]}"
  --shm-size=8g
  --tmpfs /whitefox-data:rw,size=8g
  -v "$RESULTS_DIR/logging:/workspace/WhiteFox/logging"
  -v "$RESULTS_DIR/output:/workspace/WhiteFox/output"
  -v "$RESULTS_DIR/hf_cache:/workspace/WhiteFox/hf_cache"
)

if [ "${WHITEFOX_DOCKER_SHELL:-0}" = "1" ]; then
  echo "[$(date)] Dropping into a shell inside $IMAGE_TAG"
  exec "${RUN_CMD[@]}" --entrypoint /bin/bash "$IMAGE_TAG"
fi

echo "[$(date)] Running fuzzer (results -> $RESULTS_DIR)"
echo "[$(date)] Image: $IMAGE_TAG"
echo "[$(date)] Extra args: $*"
exec "${RUN_CMD[@]}" "$IMAGE_TAG" "$@"
