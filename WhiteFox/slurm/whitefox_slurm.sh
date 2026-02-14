#!/bin/bash
#SBATCH --job-name=whitefox_tfxla
#SBATCH --partition=a40
#SBATCH --gres=gpu:1                  # Request 1 GPU (required for LLM inference)
#SBATCH --cpus-per-task=4             # CPUs for parallel test execution
#SBATCH --mem=32G                     # Memory allocation (64GB to avoid OOM)
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%j.out

set -euo pipefail

echo "[$(date)] Starting WhiteFox job on node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo

# ===================== 1. Config / CLI args =====================

# First argument: config path (relative to PROJECT_ROOT). Default: generator.toml
CONFIG_PATH="${1:-xilo_xla/config/generator.toml}"

# Second argument (optional): comma-separated list of optimizations for --only-opt
ONLY_OPT="${2:-}"

echo "Using config: $CONFIG_PATH"
if [ -n "$ONLY_OPT" ]; then
  echo "Restricting to optimizations: $ONLY_OPT"
else
  echo "Running all optimizations from config."
fi
echo

# ===================== 2. Environment setup =====================

# Ensure Poetry in PATH (installed under ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Optional: set up CUDA toolchain if available
if [ -f /vol/cuda/12.0.0/setup.sh ]; then
  . /vol/cuda/12.0.0/setup.sh
fi

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/slurm/output_tf" "$PROJECT_ROOT/hf_cache"

# Hugging Face token for gated models (stored in ~/.hf_token, not in git)
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# HF / vLLM cache directory
HF_CACHE_DIR="$PROJECT_ROOT/hf_cache"
mkdir -p "$HF_CACHE_DIR"

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export VLLM_CACHE_DIR="$HF_CACHE_DIR"

# Output directory for WhiteFox logs/artifacts
mkdir -p "$PROJECT_ROOT/output"

# Optional XLA / TF instrumentation
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Disable tokenizers parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Better CUDA allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make project importable
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Working directory: $(pwd)"
echo "PATH inside job: $PATH"
echo "HF cache dir: $HF_CACHE_DIR"
echo -n "poetry in job: "
which poetry || echo "NOT FOUND"
poetry --version || { echo "Poetry not available, aborting."; exit 1; }
echo

# Ensure config file exists
if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# ===================== 3. Run WhiteFox =====================

CMD=(poetry run python -m generation.main --config "$CONFIG_PATH")

if [ -n "$ONLY_OPT" ]; then
  CMD+=(--only-opt "$ONLY_OPT")
fi

echo "Running command:"
printf '  %q ' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "[$(date)] WhiteFox job finished."

