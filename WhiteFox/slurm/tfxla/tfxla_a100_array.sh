#!/bin/bash
#SBATCH --job-name=wf_tfxla
#SBATCH --partition=a100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=190G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/tfxla/out/whitefox_%A_%a.out

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

# Smoke-test escape hatch: when WHITEFOX_OPT_OVERRIDE is set in the
# environment (e.g. `sbatch --array=0 --time=02:00:00 \
#   --export=ALL,WHITEFOX_OPT_OVERRIDE=AllReduceCombiner \
#   WhiteFox/slurm/tfxla/tfxla_a100_array.sh`) the array task runs only
# that comma-separated subset and writes its outputs to a sibling
# logging dir, so we can validate a fix on a single problematic opt in
# ~1 hour instead of waiting 15 h for the full array to finish.
if [ -n "${WHITEFOX_OPT_OVERRIDE:-}" ]; then
    OPT_CSV="$WHITEFOX_OPT_OVERRIDE"
    BATCH_LABEL="batch${SLURM_ARRAY_TASK_ID}_smoke"
    echo "[$(date)] WHITEFOX_OPT_OVERRIDE active: only running '$OPT_CSV'"
    echo "[$(date)] Logging to $BATCH_LABEL (separate from regular batch dir)"
fi

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
export WHITEFOX_TEST_MEM_LIMIT_GB="${WHITEFOX_TEST_MEM_LIMIT_GB:-6}"
export WHITEFOX_EARLY_STOP_ITERS="${WHITEFOX_EARLY_STOP_ITERS:-20}"

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
export WHITEFOX_LOGGING_DIR="$PROJECT_ROOT/logging/$BATCH_LABEL"
mkdir -p "$WHITEFOX_LOGGING_DIR"
echo "${SLURM_ARRAY_JOB_ID}" > "$WHITEFOX_LOGGING_DIR/.job_id"

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

if [ -f /vol/cuda/12.0.0/setup.sh ]; then
  . /vol/cuda/12.0.0/setup.sh
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/slurm/tfxla/out" "$PROJECT_ROOT/hf_cache"

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

# ---- Background cgroup memory tracer ---------------------------------------
# psutil-based profiler only sees the parent process; the cgroup OOM is driven
# by children + tmpfs + page cache that the parent never sees. Sample the
# cgroup's own counters every 2 s into cgroup_mem.tsv so we can see the cliff.
CG_REL=$(awk -F'::' 'NR==1 {print $2}' /proc/self/cgroup 2>/dev/null)
CG_DIR="/sys/fs/cgroup${CG_REL}"
TRACE_FILE="$WHITEFOX_LOGGING_DIR/cgroup_mem.tsv"
XLA_DUMP_GLOB="/tmp/xla_dump_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
(
  # Loosen strict-mode in the tracer: a single failing `du` (e.g. before the
  # XLA dump dir exists) must not kill the monitor. Bash inherits set -e and
  # pipefail into subshells; that bug caused cgroup_mem.tsv to stay empty in
  # the previous run.
  set +e
  set +o pipefail
  printf 'ts\tmem_current\tmem_peak\tswap_current\toom\toom_kill\ttmp_kb\ttop5_pid_rss_cmd\n' > "$TRACE_FILE"
  while true; do
    CUR=$(cat "$CG_DIR/memory.current" 2>/dev/null)
    PEAK=$(cat "$CG_DIR/memory.peak" 2>/dev/null)
    SWAP=$(cat "$CG_DIR/memory.swap.current" 2>/dev/null)
    OOM=$(awk '/^oom /{print $2}' "$CG_DIR/memory.events" 2>/dev/null)
    OOMK=$(awk '/^oom_kill /{print $2}' "$CG_DIR/memory.events" 2>/dev/null)
    TMP_KB=$(du -sk "$XLA_DUMP_GLOB" 2>/dev/null | awk '{print $1}')
    TOP=$(ps -eo pid,rss,comm --no-headers --sort=-rss 2>/dev/null \
            | head -5 | awk '{printf "%s/%sK/%s;", $1, $2, $3}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%s)" "${CUR:-0}" "${PEAK:-0}" "${SWAP:-0}" \
      "${OOM:-0}" "${OOMK:-0}" "${TMP_KB:-0}" "$TOP" >> "$TRACE_FILE"
    sleep 2
  done
) &
TRACER_PID=$!

echo "[$(date)] Starting $BATCH_LABEL: --only-opt $OPT_CSV"
echo "[$(date)] cgroup trace: $TRACE_FILE (PID=$TRACER_PID)"

# Don't let set -e short-circuit past the postmortem when python is SIGKILL'd.
GEN_EXIT=0
poetry run python -m generation.main --sut xla --config "$CONFIG_PATH" --only-opt "$OPT_CSV" \
  || GEN_EXIT=$?

kill "$TRACER_PID" 2>/dev/null || true
echo "[$(date)] Finished $BATCH_LABEL (exit $GEN_EXIT)"

# ---- OOM postmortem: capture cgroup memory state on non-zero exit ----------
# We don't yet know what's actually consuming memory before the OOM kill.
# On any non-zero exit, dump cgroup memory.{current,peak,events,stat} plus
# recent kernel oom-kill lines to logging/oom_postmortem.log so the next
# failure is debuggable without needing to re-attach a live srun.
if [ "$GEN_EXIT" != "0" ]; then
  OOM_LOG="$WHITEFOX_LOGGING_DIR/oom_postmortem.log"
  CG_REL=$(awk -F'::' 'NR==1 {print $2}' /proc/self/cgroup 2>/dev/null)
  CG_DIR="/sys/fs/cgroup${CG_REL}"
  {
    echo "=== OOM POSTMORTEM $(date -Iseconds) ==="
    echo "exit_code=$GEN_EXIT  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  node=$(hostname)"
    echo "cgroup=$CG_DIR"
    for f in memory.current memory.peak memory.swap.current memory.swap.peak \
             memory.events memory.events.local memory.stat; do
      if [ -r "$CG_DIR/$f" ]; then
        echo "--- $f ---"
        cat "$CG_DIR/$f"
      fi
    done
    echo "--- top RSS users in this job (ps) ---"
    ps -eo pid,ppid,rss,vsz,comm,args --sort=-rss 2>/dev/null | head -30
    echo "--- recent dmesg oom lines ---"
    dmesg 2>/dev/null | grep -iE "oom|killed process|memory cgroup" | tail -40 \
      || journalctl -k --since "10 min ago" 2>/dev/null \
         | grep -iE "oom|killed process|memory cgroup" | tail -40 \
      || echo "(no kernel log access)"
    echo "--- /tmp usage ---"
    df -h /tmp 2>/dev/null
    du -sh /tmp/xla_dump_* 2>/dev/null
    echo "--- last 20 cgroup_mem.tsv samples ---"
    tail -20 "$TRACE_FILE" 2>/dev/null || echo "(no trace file)"
  } > "$OOM_LOG" 2>&1
  echo "[$(date)] OOM postmortem written to $OOM_LOG"
fi

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
