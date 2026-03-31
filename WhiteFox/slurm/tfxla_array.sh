#!/bin/bash
#SBATCH --job-name=wf_tfxla
#SBATCH --partition=a40
#SBATCH --nodelist=gpuvm33
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%A_%a.out

# ---- Array: 6 tasks, optimizations distributed dynamically -----------------
# To change the split, just update the --array range below (and keep N_TASKS
# in sync).  E.g. --array=0-7 + N_TASKS=8 gives 6 opts/task for 48 total.
#SBATCH --array=0-5
N_TASKS=6

set -euo pipefail

ALL_OPTS=(
    "AllGatherBroadcastReorder"
    "AllGatherCombiner"
    "AllGatherDecomposer"
    "AllReduceCombiner"
    "AllReduceFolder"
    "AllReduceReassociate"
    "AllReduceSimplifier"
    "AsyncCollectiveCreator"
    "BatchDotSimplification"
    "Bfloat16ConversionFolding"
    "BroadcastCanonicalizer"
    "ChangeOpDataType"
    "CollectivesScheduleLinearizer"
    "ConcatForwarding"
    "ConditionalCanonicalizer"
    "ConvertAsyncCollectivesToSync"
    "ConvertMover"
    "Defuser"
    "DotDecomposer"
    "DotMerger"
    "DynamicIndexSplitter"
    "HloConstantFolding"
    "HloCse"
    "HloDce"
    "HloElementTypeConverter"
    "IdentityConvertRemoving"
    "IdentityReshapeRemoving"
    "LoopScheduleLinearizer"
    "MapInliner"
    "ReduceScatterCombiner"
    "ReduceScatterDecomposer"
    "ReduceScatterReassociate"
    "ReshapeBroadcastForwarding"
    "ReshapeReshapeForwarding"
    "ShardingRemover"
    "SimplifyFpConversions"
    "SliceConcatForwarding"
    "SliceSinker"
    "SortSimplifier"
    "StochasticConvertDecomposer"
    "TopkRewriter"
    "TransposeFolding"
    "TreeReductionRewriter"
    "TupleSimplifier"
    "WhileLoopConstantSinking"
    "WhileLoopExpensiveInvariantCodeMotion"
    "WhileLoopInvariantCodeMotion"
    "WhileLoopTripCountAnnotator"
    "ZeroSizedHloElimination"
)

TOTAL=${#ALL_OPTS[@]}
PER_TASK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
COUNT=$PER_TASK
if (( START + COUNT > TOTAL )); then
    COUNT=$(( TOTAL - START ))
fi
if (( COUNT <= 0 )); then
    echo "[$(date)] Array task $SLURM_ARRAY_TASK_ID: nothing to do (START=$START >= TOTAL=$TOTAL)"
    exit 0
fi
MY_BATCH=("${ALL_OPTS[@]:$START:$COUNT}")
OPT_CSV=$(IFS=,; echo "${MY_BATCH[*]}")
BATCH_LABEL="batch${SLURM_ARRAY_TASK_ID}"

export WHITEFOX_COVERAGE_MERGE_EVERY_ITERS="${WHITEFOX_COVERAGE_MERGE_EVERY_ITERS:-25}"
export WHITEFOX_LLVM_PROFDATA_JOBS="${WHITEFOX_LLVM_PROFDATA_JOBS:-6}"
export WHITEFOX_PARALLEL_TEST_WORKERS="${WHITEFOX_PARALLEL_TEST_WORKERS:-4}"
export WHITEFOX_USE_CONTAINER="${WHITEFOX_USE_CONTAINER:-0}"
# Per-subprocess virtual memory cap (GB). Prevents a single pathological test
# from OOM-killing the entire SLURM job. 8 GB × 4 workers = 32 GB headroom.
export WHITEFOX_TEST_MEM_LIMIT_GB="${WHITEFOX_TEST_MEM_LIMIT_GB:-8}"

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
export WHITEFOX_LOGGING_DIR="$PROJECT_ROOT/logging/$BATCH_LABEL"
mkdir -p "$WHITEFOX_LOGGING_DIR"

WHITEFOX_SLURM_ROOT="${WHITEFOX_SLURM_ROOT:-$PROJECT_ROOT/slurm}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/container_launch.sh" ]; then
  WHITEFOX_SLURM_ROOT="$SLURM_SUBMIT_DIR"
fi
source "${WHITEFOX_SLURM_ROOT}/container_launch.sh"
whitefox_maybe_reexec_container "${BASH_SOURCE[0]}" "tfxla_array.sh" "$@"

echo "[$(date)] Array task $SLURM_ARRAY_TASK_ID ($BATCH_LABEL)"
echo "[$(date)] Optimizations (${#MY_BATCH[@]}): $OPT_CSV"
echo "[$(date)] Logging to: $WHITEFOX_LOGGING_DIR"
echo "[$(date)] Node: $(hostname), Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo

CONFIG_PATH="${1:-xilo_xla/config/generator.toml}"

# ---- LLVM toolchain --------------------------------------------------------
LLVM17_BIN="/vol/bitbucket/mtr25/tfbuild/llvm17/bin"
if [ -x "$LLVM17_BIN/llvm-profdata" ]; then
  export WHITEFOX_LLVM_DIR="$LLVM17_BIN"
fi
for _llvm_dir in /vol/bitbucket/mtr25/tfbuild/tmp/bazel_root_*/*/external/llvm_linux_x86_64/bin; do
  [ -d "$_llvm_dir" ] && export PATH="$_llvm_dir:$PATH"
done
export PATH="$LLVM17_BIN:$HOME/.local/bin:$PATH"

if [ -z "${WHITEFOX_IN_CONTAINER:-}" ] && [ -f /vol/cuda/12.0.0/setup.sh ]; then
  . /vol/cuda/12.0.0/setup.sh
fi

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

rm -rf /tmp/wf_profraw_* /tmp/xla_dump 2>/dev/null || true

export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

TF_WHEEL="/vol/bitbucket/mtr25/tfbuild/wheels/tensorflow_cpu-V2.20.0.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl"

poetry --version || exit 1

LOCKFILE="$PROJECT_ROOT/.install.lock"
(
  flock -x 200
  echo "[$(date)] [$BATCH_LABEL] Acquired install lock"
  poetry lock
  poetry install --no-interaction
  echo "[$(date)] Force-reinstalling TensorFlow wheel"
  poetry run pip install --force-reinstall --no-deps "$TF_WHEEL"
  echo "[$(date)] [$BATCH_LABEL] Install done, releasing lock"
) 200>"$LOCKFILE"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "[$(date)] Starting $BATCH_LABEL: --only-opt $OPT_CSV"
poetry run python -m generation.main --sut xla --config "$CONFIG_PATH" --only-opt "$OPT_CSV"
echo "[$(date)] Finished $BATCH_LABEL (exit $?)"
