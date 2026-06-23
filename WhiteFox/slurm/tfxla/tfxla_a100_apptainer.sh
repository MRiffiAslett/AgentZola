#!/bin/bash
#SBATCH --job-name=wf_apptainer
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=190G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/tfxla/out/whitefox-apptainer_%j.out
# =============================================================================
# WhiteFox TF/XLA fuzzer — Apptainer SLURM launcher.
#
# Runs the pre-built Apptainer image (.sif) produced by build_apptainer.sh.
# Build the image ONCE on the login node before submitting this script.
#
# Usage:
#   # full run (all 49 optimizations, sequential):
#   sbatch tfxla_a100_apptainer.sh
#
#   # quick test — single cheap optimization:
#   sbatch tfxla_a100_apptainer.sh --only-opt HloDce
#
#   # subset:
#   sbatch tfxla_a100_apptainer.sh --only-opt HloDce,HloCse,HloDce
#
# Optional env knobs (pass via --export=ALL,VAR=val):
#   WHITEFOX_SIF_PATH          path to the .sif or sandbox dir
#                               (default: /vol/bitbucket/mtr25/whitefox-tfxla.sif)
#   WHITEFOX_RESULTS_DIR       host dir for logging/, output/, hf_cache/
#                               (default: /vol/bitbucket/mtr25/AgentZola/WhiteFox)
#   WHITEFOX_EARLY_STOP_ITERS  0 = no early stop, 20 = default
# =============================================================================

set -euo pipefail

SIF_PATH="${WHITEFOX_SIF_PATH:-/vol/bitbucket/mtr25/whitefox-tfxla.sif}"
RESULTS_DIR="${WHITEFOX_RESULTS_DIR:-/vol/bitbucket/mtr25/AgentZola/WhiteFox}"

# ---- Sanity checks ---------------------------------------------------------
if [ ! -e "$SIF_PATH" ]; then
  cat >&2 <<MSG
ERROR: Apptainer image not found at: $SIF_PATH

Build it first from the login node:
  bash WhiteFox/slurm/tfxla/build_apptainer.sh

or point at an existing image:
  sbatch --export=ALL,WHITEFOX_SIF_PATH=/path/to/image.sif tfxla_a100_apptainer.sh
MSG
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH on compute node $(hostname)" >&2
  exit 1
fi

# ---- Job info --------------------------------------------------------------
echo "[$(date)] WhiteFox TF/XLA — Apptainer run"
echo "[$(date)] Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"
echo "[$(date)] SIF: $SIF_PATH"
echo "[$(date)] Results dir: $RESULTS_DIR"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "[$(date)] Extra args: $*"
echo

# ---- HF token --------------------------------------------------------------
HF_ENV_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_ENV_ARGS+=(--env "HF_TOKEN=$HF_TOKEN" --env "HUGGINGFACE_HUB_TOKEN=$HF_TOKEN")
elif [ -f "$HOME/.hf_token" ]; then
  _tok="$(tr -d '[:space:]' < "$HOME/.hf_token")"
  HF_ENV_ARGS+=(--env "HF_TOKEN=$_tok" --env "HUGGINGFACE_HUB_TOKEN=$_tok")
else
  echo "WARNING: no HF token found (\$HF_TOKEN or ~/.hf_token)." >&2
  echo "         vLLM will fail at StarCoder model download." >&2
fi

# ---- Prepare result directories on the host --------------------------------
mkdir -p "$RESULTS_DIR/logging" "$RESULTS_DIR/output" "$RESULTS_DIR/hf_cache"

# ---- Run -------------------------------------------------------------------
# --nv          passes NVIDIA GPU device + drivers into the container
# --bind        mounts host result dirs over the image's placeholders
# --env         injects per-run overrides without rebuilding the image
apptainer run --nv \
  --bind "$RESULTS_DIR/logging:/workspace/WhiteFox/logging" \
  --bind "$RESULTS_DIR/output:/workspace/WhiteFox/output" \
  --bind "$RESULTS_DIR/hf_cache:/workspace/WhiteFox/hf_cache" \
  --bind /tmp:/tmp \
  "${HF_ENV_ARGS[@]}" \
  --env "WHITEFOX_EARLY_STOP_ITERS=${WHITEFOX_EARLY_STOP_ITERS:-20}" \
  "$SIF_PATH" \
  --sut xla --config xilo_xla/config/generator.toml \
  "$@"
