#!/bin/bash
#SBATCH --job-name=whitefox_batch
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%j.out

set -euo pipefail

# Arguments: $1 = comma-separated optimization list, $2 = batch name
OPTIMIZATIONS="${1:-}"
BATCH_NAME="${2:-unknown}"

if [ -z "$OPTIMIZATIONS" ]; then
    echo "ERROR: No optimizations specified!"
    exit 1
fi

echo "[$(date)] Starting WhiteFox batch: $BATCH_NAME"
echo "Node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "Optimizations: $OPTIMIZATIONS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
CONFIG_PATH="xilo_xla/config/generator.toml"

# Make sure Poetry is visible inside the batch environment
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/slurm/output_tf" "$PROJECT_ROOT/hf_cache"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Make HF cache writable (fixes /JawTitan permission issues)
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

# Keep vLLM on GPU (default), and force TF/XLA side to CPU via code (see oracle.py change)
export WHITEFOX_TF_CPU_ONLY="1"

POETRY_ENV="$(poetry env info --path)"
source "$POETRY_ENV/bin/activate"

# Run WhiteFox with only specified optimizations
python generation/main.py --config "$CONFIG_PATH" --only-opt "$OPTIMIZATIONS"

echo "[$(date)] WhiteFox batch $BATCH_NAME finished."

