# Source this from WhiteFox Slurm job scripts after set -euo pipefail.
#
# Slurm copies the batch script to /var/spool/slurm/.../slurm_script — do not locate
# this file via dirname(BASH_SOURCE) of the job script. Job scripts must set
# WHITEFOX_SLURM_ROOT and source "${WHITEFOX_SLURM_ROOT}/container_launch.sh".
#
# Enable container execution (job scripts default WHITEFOX_USE_CONTAINER=0; set 1 to opt in):
#   WHITEFOX_USE_CONTAINER=1
#   WHITEFOX_APPTAINER_IMAGE=docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
# Or point to a local image built on the login node, e.g.:
#   WHITEFOX_APPTAINER_IMAGE=/vol/bitbucket/$USER/containers/whitefox.sif
#
# Disable and run on the cluster host (previous behaviour):
#   WHITEFOX_USE_CONTAINER=0
#
# Extra bind mounts (space-separated -B args), e.g.:
#   WHITEFOX_APPTAINER_EXTRA_BINDS="-B /scratch/$USER:/scratch/$USER"
#
# Some sites use Slurm's native container support instead (Pyxis/Enroot); see comments
# in the job scripts for #SBATCH --container-image examples.

# Args: path Slurm uses for the running script (often spool copy), real basename (e.g. tfxla_slurm.sh), then job args.
whitefox_maybe_reexec_container() {
  local spool_script="$1"
  local known_basename="$2"
  shift 2

  local script_path="$spool_script"
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/$known_basename" ]; then
    script_path="$SLURM_SUBMIT_DIR/$known_basename"
  fi

  if [ "${WHITEFOX_USE_CONTAINER:-0}" != "1" ]; then
    return 0
  fi
  if [ -n "${WHITEFOX_IN_CONTAINER:-}" ]; then
    return 0
  fi

  local img="${WHITEFOX_APPTAINER_IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  local runner=""
  if command -v apptainer >/dev/null 2>&1; then
    runner=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    runner=singularity
  else
    echo "ERROR: WHITEFOX_USE_CONTAINER=1 but neither apptainer nor singularity is in PATH." >&2
    echo "Install Apptainer/Singularity on the cluster, set WHITEFOX_USE_CONTAINER=0, or use Slurm's --container-image if your site provides it." >&2
    exit 1
  fi

  echo "[$(date)] Re-executing inside container via ${runner}: ${img}"
  # shellcheck disable=SC2086
  exec "$runner" exec --nv \
    -B /vol:/vol \
    -B "${HOME}:${HOME}" \
    -B /tmp:/tmp \
    ${WHITEFOX_APPTAINER_EXTRA_BINDS:-} \
    "$img" \
    env WHITEFOX_IN_CONTAINER=1 bash "$script_path" "$@"
}
