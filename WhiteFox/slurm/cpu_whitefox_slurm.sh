#!/bin/bash
#SBATCH --job-name=whitefox_cpu
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%j.out

set -euo pipefail

echo "[$(date)] Starting WhiteFox CPU-only job on node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo

# ---------- Environment ----------
export PATH="$HOME/.local/bin:$PATH"

# 🔒 Force CPU-only TensorFlow (this is the key line)
export CUDA_VISIBLE_DEVICES=""

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/output"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------- Activate Poetry ----------
POETRY_ENV=$(poetry env info --path)
source "$POETRY_ENV/bin/activate"

# ---------- Run ----------
python generation/main.py --config xilo_xla/config/generator.toml

echo "[$(date)] WhiteFox job finished."

