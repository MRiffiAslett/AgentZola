#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --partition=t4
#SBATCH --mail-user=${USER}
#SBATCH --job-name=whitefox_tfxla
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%j.out

set -euo pipefail

echo "[$(date)] Starting WhiteFox job on node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo

# ---------- 1. Config ----------
CONFIG_PATH="xilo_xla/config/generator.toml"

# ---------- 2. Environment ----------
export PATH="$HOME/.local/bin:$PATH"

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
cd "$PROJECT_ROOT"

# Ensure output directory exists
mkdir -p "$PROJECT_ROOT/output"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------- 3. Run ----------
CMD=(poetry run python -m generation.main --config "$CONFIG_PATH")

echo "PATH inside job: $PATH"
echo -n "poetry in job: "

echo "Running WhiteFox with config: $CONFIG_PATH"
"${CMD[@]}"

echo "[$(date)] WhiteFox job finished."

