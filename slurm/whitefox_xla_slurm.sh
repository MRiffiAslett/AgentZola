#!/bin/bash
#
# WhiteFox XLA Compiler Fuzzing - SLURM Submission Script
# 
# This script submits a WhiteFox fuzzing job to the SLURM cluster.
# WhiteFox is a white-box compiler fuzzing tool that uses Large Language Models
# (LLMs) to generate test programs that target specific compiler optimizations.
#
# Usage (from project root /vol/bitbucket/mtr25/AgentZola/):
#   sbatch slurm/whitefox_slurm.sh
#   sbatch slurm/whitefox_slurm.sh "xilo_xla/config/generator.toml" "HloConstantFolding,HloCse"
#
# Usage (from slurm directory):
#   cd /vol/bitbucket/mtr25/AgentZola/slurm
#   sbatch whitefox_slurm.sh
#   sbatch whitefox_slurm.sh "xilo_xla/config/generator.toml" "HloConstantFolding,HloCse"
#
# Arguments:
#   $1: Path to TOML configuration file (relative to WhiteFox/ or absolute)
#   $2: Optional comma-separated list of optimization names to target
#

# =============================================================================
# SLURM DIRECTIVES - Resource allocation and job configuration
# =============================================================================

#SBATCH --job-name=whitefox_tfxla     # Job name visible in queue
#SBATCH --partition=t4                # Request T4 GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU (required for LLM inference)
#SBATCH --cpus-per-task=8             # CPUs for parallel test execution
#SBATCH --mem=32G                     # Memory allocation (adjust for model size)
#SBATCH --time=02:00:00               # Time limit (2 hours, adjust as needed)
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/output_xla/whitefox_%j.out  # SLURM stdout/stderr (%j = job ID)
#SBATCH --mail-type=END,FAIL          # Email notifications on job completion/failure
#SBATCH --mail-user=${USER}@ic.ac.uk  # Email address for notifications

# =============================================================================
# SCRIPT SETTINGS - Error handling and initial logging
# =============================================================================

# set -e: Exit immediately if any command fails
# set -u: Treat unset variables as errors
# set -o pipefail: Return exit status of failed command in pipeline
set -euo pipefail

echo "[$(date)] Starting WhiteFox job on node: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo

# =============================================================================
# SECTION 1: CONFIGURATION FILE AND OPTIMIZATION FILTER
# =============================================================================
# Parse command-line arguments to determine which configuration file to use
# and optionally filter which optimizations to process.
# =============================================================================

# Default configuration file path (relative to WhiteFox/ directory)
# This TOML file contains all WhiteFox settings: model config, paths, generation params, etc.
DEFAULT_CONFIG="xilo_xla/config/generator.toml"

# Parse command-line arguments:
#   $1: Configuration file path (optional, defaults to DEFAULT_CONFIG)
#   $2: Comma-separated list of optimization names (optional)
#       If provided, only these optimizations will be processed instead of all 44
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
ONLY_OPT="${2:-}"   # Empty string if not provided

echo "Configuration:"
echo "  Config file: $CONFIG_PATH"
if [ -n "$ONLY_OPT" ]; then
  echo "  Filtering optimizations: $ONLY_OPT"
  echo "  (Only these will be processed, others will be skipped)"
else
  echo "  Processing all optimizations from config file"
  echo "  (All 44 optimizations in generator.toml will be processed)"
fi
echo

# =============================================================================
# SECTION 2: ENVIRONMENT SETUP
# =============================================================================
# Configure the runtime environment: Poetry, CUDA, Python paths, and XLA flags.
# This ensures all dependencies and tools are available and properly configured.
# =============================================================================

# --- Poetry Setup ---
# Poetry is the Python dependency manager used by this project.
# Ensure Poetry is in PATH so we can run 'poetry run' commands.
# Poetry is typically installed in ~/.local/bin when installed via pip/installer
export PATH="$HOME/.local/bin:$PATH"

# --- CUDA Setup (Optional but Recommended) ---
# CUDA is required for GPU acceleration (LLM inference, TensorFlow/XLA operations).
# Load CUDA environment if available on this cluster.
# This sets up CUDA_PATH, LD_LIBRARY_PATH, and other necessary CUDA variables.
if [ -f /vol/cuda/12.0.0/setup.sh ]; then
  echo "Loading CUDA 12.0.0 environment..."
  . /vol/cuda/12.0.0/setup.sh
fi

# --- Project Root and Working Directory ---
# CRITICAL: The code structure requires running from the AgentZola project root,
# not from the WhiteFox subdirectory. This is because:
#   - generator.py imports "AgentZola.WhiteFox.models.generation"
#   - The code dynamically adds project_root (AgentZola/) to sys.path
#   - All relative paths in config files are resolved from project root
PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola"
cd "$PROJECT_ROOT"
echo "Changed to project root: $PROJECT_ROOT"

# --- Create Output Directory ---
# WhiteFox generates several types of output:
#   - Generated test programs (Python files)
#   - Execution logs (stdout/stderr from test runs)
#   - Bug reports (JSON files when oracles detect issues)
#   - Bandit state file (JSON persistence of fuzzing progress)
# All output goes into output_xla/ subdirectory
OUTPUT_DIR="$PROJECT_ROOT/WhiteFox/output_xla"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# --- XLA/TensorFlow Instrumentation Flags ---
# These environment variables configure TensorFlow/XLA to:
#   XLA_FLAGS: Dump XLA HLO graphs to /tmp/xla_dump/ for debugging
#   TF_XLA_FLAGS: Enable XLA auto-clustering (jit_compile=True behavior)
# These are useful for understanding what optimizations are being triggered
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# --- Python Path Configuration ---
# Ensure the project root is in PYTHONPATH so imports resolve correctly.
# This is a safety measure; the code also adds it programmatically, but having
# it in the environment ensures consistency.
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# --- Verify Python Environment ---
# Print information about the Python interpreter and paths to aid debugging.
# This confirms we're using Poetry's virtual environment with correct dependencies.
echo
echo "Environment check:"
echo "  Working directory: $(pwd)"
echo "  Python interpreter:"
poetry run python -c "import sys; print(f'    {sys.executable}')"
echo "  Python path (first 3 entries):"
poetry run python -c "import sys; [print(f'    {p}') for p in sys.path[:3]]"
echo

# =============================================================================
# SECTION 3: RUN WHITEFOX FUZZING
# =============================================================================
# Execute the WhiteFox main module with the specified configuration.
# The main.py script will:
#   1. Load the TOML config file
#   2. Initialize the LLM (VLLM) with model settings
#   3. Load optimization specifications from optimizations_dir
#   4. For each optimization (or filtered subset):
#      - Generate test programs using LLM
#      - Execute tests in subprocess
#      - Check if optimization was triggered
#      - Run oracles to detect bugs
#      - Update bandit state (Thompson Sampling)
#      - Save results and persist state
# =============================================================================

# --- Resolve Configuration File Path ---
# The config path can be provided as:
#   - Absolute path: /full/path/to/config.toml
#   - Relative path: xilo_xla/config/generator.toml (relative to WhiteFox/)
# This code normalizes it to an absolute path for reliability.
if [[ "$CONFIG_PATH" = /* ]]; then
  # Already an absolute path, use as-is
  FULL_CONFIG_PATH="$CONFIG_PATH"
else
  # Relative path, resolve relative to WhiteFox/ directory
  FULL_CONFIG_PATH="$PROJECT_ROOT/WhiteFox/$CONFIG_PATH"
fi

# --- Validate Configuration File Exists ---
# Fail early if the config file is missing rather than getting cryptic errors
# later during initialization.
if [ ! -f "$FULL_CONFIG_PATH" ]; then
  echo "ERROR: Configuration file not found: $FULL_CONFIG_PATH" >&2
  echo "Please check the path and try again." >&2
  exit 1
fi
echo "Using configuration file: $FULL_CONFIG_PATH"

# --- Build Command Array ---
# Construct the command to run WhiteFox. Using an array makes it easy to:
#   - Add conditional arguments (--only-opt)
#   - Properly quote paths with spaces
#   - Print the exact command before execution
# The module path 'WhiteFox.generation.main' matches the package structure:
#   AgentZola/WhiteFox/generation/main.py
CMD=(poetry run python -m WhiteFox.generation.main --config "$FULL_CONFIG_PATH")

# --- Add Optional Optimization Filter ---
# If --only-opt was provided, add it to the command.
# This limits processing to specific optimizations (useful for testing/debugging).
if [ -n "$ONLY_OPT" ]; then
  CMD+=(--only-opt "$ONLY_OPT")
fi

# --- Print and Execute Command ---
# Print the exact command being run for transparency and debugging.
# The printf with %q properly escapes special characters in the output.
echo
echo "Executing WhiteFox fuzzing:"
printf '  %q ' "${CMD[@]}"
echo
echo

# Execute the command. The script will exit with the command's exit code
# due to 'set -e', so SLURM will mark the job as FAILED if WhiteFox fails.
"${CMD[@]}"

# --- Job Completion ---
# If we reach here, the job completed successfully.
echo
echo "[$(date)] WhiteFox job finished successfully."
echo "Check output directory for results: $OUTPUT_DIR"

