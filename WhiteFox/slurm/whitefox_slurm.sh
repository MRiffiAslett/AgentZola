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

# (Optional but recommended on T4): force an attention backend explicitly
# export VLLM_ATTENTION_BACKEND="FLASHINFER"

POETRY_ENV="$(poetry env info --path)"
source "$POETRY_ENV/bin/activate"

python generation/main.py --config "$CONFIG_PATH"

echo "[$(date)] WhiteFox job finished."

