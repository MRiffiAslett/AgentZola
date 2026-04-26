#!/bin/bash
#SBATCH --job-name=wf_tfxla
#SBATCH --partition=a100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=600G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_%A_%a.out

# ---- Array: 3 tasks, ~17 opts each.  With early-stop disabled every opt runs
# the full 100 iterations (1000 tests); 17 opts × ~90 min ≈ 25.5 h per task,
# within the 72 h wall-time limit.
#SBATCH --array=0-2
N_TASKS=3

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

# ---- Coverage: %Nm merge pool on node-local /data --------------------------
# Instead of one 950 MB profraw per subprocess on NFS (the old bottleneck),
# LLVM's %Nm merge pool keeps N shared files on local ext4.  Subprocesses
# lock → merge counters → unlock in-place.  After 1000 tests, only N files
# need merging instead of 1000.  ~500x less merge I/O.
LOCAL_COV_DIR="/data/whitefox_cov_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$LOCAL_COV_DIR"
export WHITEFOX_PROFRAW_DIR="$LOCAL_COV_DIR"
export WHITEFOX_PROFRAW_POOL_SIZE="${WHITEFOX_PROFRAW_POOL_SIZE:-8}"

# With pool mode, intermediate merges are cheap (8 files, local disk).
# Merge every 100 iterations = once per optimization boundary.
export WHITEFOX_COVERAGE_MERGE_EVERY_ITERS="${WHITEFOX_COVERAGE_MERGE_EVERY_ITERS:-100}"
export WHITEFOX_LLVM_PROFDATA_JOBS="${WHITEFOX_LLVM_PROFDATA_JOBS:-0}"
export WHITEFOX_MERGE_BATCH_SIZE="${WHITEFOX_MERGE_BATCH_SIZE:-8}"

# ---- Test execution --------------------------------------------------------
# 18 CPUs available.  vLLM uses ~2-3 CPU threads; leave the rest for workers.
# With early-stop disabled every opt runs 1000 tests; longer runs accumulate
# more memory (TF/XLA subprocesses, profraw, caches).  8 workers caused OOM
# at ~iteration 59 under 190 GB.  4 workers × 10 GB RLIMIT_AS = 40 GB virtual,
# leaving ample headroom for vLLM + coverage merge.
export WHITEFOX_PARALLEL_TEST_WORKERS="${WHITEFOX_PARALLEL_TEST_WORKERS:-4}"
export WHITEFOX_USE_CONTAINER="${WHITEFOX_USE_CONTAINER:-0}"
export WHITEFOX_TEST_MEM_LIMIT_GB="${WHITEFOX_TEST_MEM_LIMIT_GB:-6}"
export WHITEFOX_EARLY_STOP_ITERS="${WHITEFOX_EARLY_STOP_ITERS:-20}"

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
export WHITEFOX_LOGGING_DIR="$PROJECT_ROOT/logging/$BATCH_LABEL"
mkdir -p "$WHITEFOX_LOGGING_DIR"
echo "${SLURM_ARRAY_JOB_ID}" > "$WHITEFOX_LOGGING_DIR/.job_id"

WHITEFOX_SLURM_ROOT="${WHITEFOX_SLURM_ROOT:-$PROJECT_ROOT/slurm}"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/container_launch.sh" ]; then
  WHITEFOX_SLURM_ROOT="$SLURM_SUBMIT_DIR"
fi
source "${WHITEFOX_SLURM_ROOT}/container_launch.sh"
whitefox_maybe_reexec_container "${BASH_SOURCE[0]}" "tfxla_a100_array.sh" "$@"

echo "[$(date)] Array task $SLURM_ARRAY_TASK_ID ($BATCH_LABEL)"
echo "[$(date)] Optimizations (${#MY_BATCH[@]}): $OPT_CSV"
echo "[$(date)] Logging to: $WHITEFOX_LOGGING_DIR"
echo "[$(date)] Local coverage: $LOCAL_COV_DIR (pool_size=$WHITEFOX_PROFRAW_POOL_SIZE)"
echo "[$(date)] Node: $(hostname), Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Resources: $(nproc) CPUs, $(free -h | awk '/^Mem:/{print $2}') RAM"
echo "[$(date)] /data storage:"
df -h /data 2>/dev/null || echo "  /data not available"
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

XLA_DUMP_DIR="/tmp/xla_dump_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
rm -rf "$XLA_DUMP_DIR" 2>/dev/null || true
rm -rf /tmp/wf_profraw_* 2>/dev/null || true

export XLA_FLAGS="--xla_dump_to=$XLA_DUMP_DIR"
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
GEN_EXIT=$?
echo "[$(date)] Finished $BATCH_LABEL (exit $GEN_EXIT)"

# ---- Copy coverage artifacts from local /data to NFS -----------------------
echo "[$(date)] Copying coverage artifacts to NFS …"
if [ -f "$WHITEFOX_LOGGING_DIR/coverage/merged.profdata" ]; then
  echo "[$(date)] merged.profdata: $(du -h "$WHITEFOX_LOGGING_DIR/coverage/merged.profdata" | cut -f1)"
fi
# Clean up node-local coverage scratch.
rm -rf "$LOCAL_COV_DIR" 2>/dev/null || true
echo "[$(date)] Local coverage dir cleaned up"

# ---- Aggregate batch summaries into a single combined report ---------------
echo "[$(date)] Aggregating batch summaries …"
poetry run python -c "
from pathlib import Path
from collections import defaultdict

logging_root = Path('$PROJECT_ROOT') / 'logging'
combined_summary = logging_root / 'run_summary_combined.log'
combined_coverage = logging_root / 'coverage_combined.log'
my_job_id = '$SLURM_ARRAY_JOB_ID'

def batch_belongs_to_current_run(batch_dir):
    jf = batch_dir / '.job_id'
    if not jf.exists():
        return False
    return jf.read_text().strip() == my_job_id

batch_dirs = [d for d in sorted(logging_root.glob('batch*')) if d.is_dir() and batch_belongs_to_current_run(d)]
skipped = [d.name for d in sorted(logging_root.glob('batch*')) if d.is_dir() and not batch_belongs_to_current_run(d)]
if skipped:
    print(f'Skipping stale batches (wrong job_id): {skipped}')

totals = defaultdict(lambda: defaultdict(int))
header_line = ''

for batch_dir in batch_dirs:
    summary = batch_dir / 'run_summary_detailed.log'
    if not summary.exists():
        continue
    for line in summary.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith(('=', '-', 'Optimization', 'WHITEFOX')):
            if line.startswith('Optimization'):
                header_line = line
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            continue
        if parts[0] == 'TOTAL':
            continue
        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            continue
        for i, v in enumerate(values):
            totals[parts[0]][i] += v

if totals:
    with open(combined_summary, 'w') as f:
        sep = '=' * 95
        dash = '-' * 95
        f.write(f'{sep}\nWHITEFOX COMBINED RUN SUMMARY (all batches)\n{sep}\n\n')
        if header_line:
            f.write(header_line + '\n')
        f.write(dash + '\n\n')
        grand = defaultdict(int)
        ncols = max(len(v) for v in totals.values())
        for opt_name in sorted(totals):
            vals = totals[opt_name]
            f.write(f'{opt_name:40s}')
            for i in range(ncols):
                v = vals.get(i, 0)
                grand[i] += v
                f.write(f' | {v:7d}')
            f.write('\n')
        f.write(dash + '\n')
        f.write(f'{\"TOTAL\":40s}')
        for i in range(len(grand)):
            f.write(f' | {grand[i]:7d}')
        f.write('\n' + sep + '\n')
    print(f'Combined run summary: {combined_summary}')
else:
    print('No batch summaries found to aggregate.')

cov_lines = []
for batch_dir in batch_dirs:
    cov_file = batch_dir / 'coverage_report.log'
    if not cov_file.exists():
        continue
    text = cov_file.read_text().strip()
    if 'UNAVAILABLE' in text and len(text) < 200:
        cov_lines.append(f'--- {batch_dir.name}: coverage unavailable ---')
    else:
        cov_lines.append(f'--- {batch_dir.name} ---')
        cov_lines.append(text)
    cov_lines.append('')

if cov_lines:
    sep = '=' * 60
    with open(combined_coverage, 'w') as f:
        f.write(f'{sep}\nCOMBINED COVERAGE REPORT (all batches)\n{sep}\n\n')
        f.write('\n'.join(cov_lines) + '\n')
    print(f'Combined coverage: {combined_coverage}')
" 2>&1 || echo "[$(date)] Aggregation failed (non-fatal)"

exit $GEN_EXIT
