#!/bin/bash
#SBATCH --job-name=whitefox_tfxla
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%j.out

set -euo pipefail

echo "[$(date)] Starting WhiteFox TF-XLA job on node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo

CONFIG_PATH="${1:-xilo_xla/config/generator.toml}"
ONLY_OPT="${2:-}"

export PATH="$HOME/.local/bin:$PATH"

if [ -f /vol/cuda/12.0.0/setup.sh ]; then
  . /vol/cuda/12.0.0/setup.sh
fi

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/slurm/output_tf" "$PROJECT_ROOT/hf_cache"

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

HF_CACHE_DIR="$PROJECT_ROOT/hf_cache"
mkdir -p "$HF_CACHE_DIR"

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export VLLM_CACHE_DIR="$HF_CACHE_DIR"

mkdir -p "$PROJECT_ROOT/output"

export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

poetry --version || exit 1

if [ ! -f "$CONFIG_PATH" ]; then
  exit 1
fi

CMD=(poetry run python -m generation.main --sut xla --config "$CONFIG_PATH")

if [ -n "$ONLY_OPT" ]; then
  CMD+=(--only-opt "$ONLY_OPT")
fi

"${CMD[@]}"
