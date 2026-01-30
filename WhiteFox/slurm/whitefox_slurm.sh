#!/bin/bash
#SBATCH --job-name=whitefox_vllm_gpu_xla_cpu
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%j.out

set -euo pipefail

echo "[$(date)] Starting WhiteFox job on node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
CONFIG_PATH="xilo_xla/config/generator.toml"
FULL_CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"

# Make sure Poetry is visible inside the batch environment
export PATH="$HOME/.local/bin:$PATH"

# Optional: set up CUDA toolchain if available
if [ -f /vol/cuda/12.0.0/setup.sh ]; then
  . /vol/cuda/12.0.0/setup.sh
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/slurm/output_tf" "$PROJECT_ROOT/hf_cache"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Make HF cache writable (fixes /JawTitan permission issues)
HF_CACHE_DIR="$PROJECT_ROOT/hf_cache"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export VLLM_CACHE_DIR="$HF_CACHE_DIR"

# Disable tokenizers parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Better CUDA allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional XLA / TF instrumentation
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Keep vLLM on GPU (default), and force TF/XLA side to CPU via code (see oracle.py change)
export WHITEFOX_TF_CPU_ONLY="1"

# (Optional but recommended on T4): force an attention backend explicitly
# export VLLM_ATTENTION_BACKEND="FLASHINFER"

echo "Working directory: $(pwd)"
echo "PATH inside job: $PATH"
echo "HF cache dir: $HF_CACHE_DIR"
echo -n "poetry in job: "
which poetry || echo "NOT FOUND"
poetry --version || { echo "Poetry not available, aborting."; exit 1; }
echo

# Check if config file exists
if [ ! -f "$FULL_CONFIG_PATH" ]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "Using config: $CONFIG_PATH"
echo "Running all optimizations from config."
echo

poetry run python -m generation.main --config "$CONFIG_PATH"

echo "[$(date)] WhiteFox job finished."

