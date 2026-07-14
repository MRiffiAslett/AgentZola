#!/bin/bash
#SBATCH --job-name=wf_tfxla
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/tfxla/out/whitefox_%A_%a.out

#SBATCH --array=0-2
N_TASKS=3

set -euo pipefail

# ===========================================================================
# RUN CONFIGURATION — edit the three variables below to switch experiments.
#
#   MODEL:   bigcode/starcoder              (7 B,  float16,  8 192-token ctx)
#            Qwen/Qwen2.5-Coder-14B         (14 B, bfloat16, 32 768-token ctx)
#            Qwen/Qwen2.5-Coder-14B-Instruct(14 B, bfloat16, 32 768-token ctx)
#
#   WHEEL:   20250806 → tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806
#            20230507 → tensorflow_cpu-2.14.0+selfbuilt.20230507
#
#   PROMPTS: 20250806 | 20230507
#
# Model-specific vLLM params (dtype, max_model_len, stop tokens, …) are
# resolved automatically from the _MODEL_REGISTRY in generation/generator.py —
# no other file needs editing when you switch MODEL here.
# ===========================================================================
WHITEFOX_MODEL="bigcode/starcoder"
WHITEFOX_WHEEL_VERSION="20250806"
WHITEFOX_PROMPTS_VERSION="20250806"
export WHITEFOX_MODEL WHITEFOX_WHEEL_VERSION WHITEFOX_PROMPTS_VERSION
# ===========================================================================

_WHEEL_DIR="/vol/bitbucket/mtr25/tfbuild/wheels"
case "$WHITEFOX_WHEEL_VERSION" in
  20250806) WHITEFOX_TF_WHEEL="$_WHEEL_DIR/tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806-cp312-cp312-linux_x86_64.whl" ;;
  20230507) WHITEFOX_TF_WHEEL="$_WHEEL_DIR/tensorflow_cpu-2.14.0+selfbuilt.20230507-cp310-cp310-linux_x86_64.whl" ;;
  *) echo "ERROR: unknown WHITEFOX_WHEEL_VERSION=$WHITEFOX_WHEEL_VERSION" >&2; exit 1 ;;
esac

case "$WHITEFOX_PROMPTS_VERSION" in
  20250806) WHITEFOX_PROMPTS_DIR="xilo_xla/artifacts/generation-prompts-20250806" ;;
  20230507) WHITEFOX_PROMPTS_DIR="xilo_xla/artifacts/generation-prompts-20230507" ;;
  *) echo "ERROR: unknown WHITEFOX_PROMPTS_VERSION=$WHITEFOX_PROMPTS_VERSION" >&2; exit 1 ;;
esac


case "$WHITEFOX_MODEL" in
  bigcode/starcoder)                  WHITEFOX_MODEL_DISPLAY="StarCoder (7B)"              ;;
  Qwen/Qwen2.5-Coder-14B)            WHITEFOX_MODEL_DISPLAY="Qwen2.5-Coder-14B"           ;;
  Qwen/Qwen2.5-Coder-14B-Instruct)  WHITEFOX_MODEL_DISPLAY="Qwen2.5-Coder-14B-Instruct"  ;;
  *)                                  WHITEFOX_MODEL_DISPLAY="$WHITEFOX_MODEL"             ;;
esac
export WHITEFOX_MODEL_DISPLAY

: "${USER:=$(id -un)}"
: "${HOME:=$(getent passwd "$USER" 2>/dev/null | cut -d: -f6)}"
: "${HOME:=/tmp}"
export HOME USER

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

if [ -n "${WHITEFOX_OPT_OVERRIDE:-}" ]; then
    OPT_CSV="$WHITEFOX_OPT_OVERRIDE"
    BATCH_LABEL="batch${SLURM_ARRAY_TASK_ID}_smoke"
    echo "[$(date)] WHITEFOX_OPT_OVERRIDE active: only running '$OPT_CSV'"
    echo "[$(date)] Logging to $BATCH_LABEL (separate from regular batch dir)"
fi


LOCAL_COV_DIR="/data/whitefox_cov_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$LOCAL_COV_DIR"
export WHITEFOX_PROFRAW_DIR="$LOCAL_COV_DIR"
export WHITEFOX_PROFRAW_POOL_SIZE="${WHITEFOX_PROFRAW_POOL_SIZE:-8}"

# Model weights on local SSD — avoids 40-min NFS load per job.
# First run downloads to /data/hf_cache; subsequent runs reuse it.
export HF_HOME=/data/hf_cache
mkdir -p "$HF_HOME"

# Keep vLLM telemetry and compile cache off /homes (disk quota).
export VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/data/vllm_cache
mkdir -p "$VLLM_CACHE_ROOT"


export WHITEFOX_COVERAGE_MERGE_EVERY_ITERS="${WHITEFOX_COVERAGE_MERGE_EVERY_ITERS:-100}"
export WHITEFOX_LLVM_PROFDATA_JOBS="${WHITEFOX_LLVM_PROFDATA_JOBS:-0}"
export WHITEFOX_MERGE_BATCH_SIZE="${WHITEFOX_MERGE_BATCH_SIZE:-8}"


export WHITEFOX_PARALLEL_TEST_WORKERS="${WHITEFOX_PARALLEL_TEST_WORKERS:-4}"
export WHITEFOX_TEST_MEM_LIMIT_GB="${WHITEFOX_TEST_MEM_LIMIT_GB:-6}"
export WHITEFOX_EARLY_STOP_ITERS="${WHITEFOX_EARLY_STOP_ITERS:-0}"

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
export PROJECT_ROOT
export WHITEFOX_LOGGING_DIR="$PROJECT_ROOT/logging/$BATCH_LABEL"

if [ -d "$WHITEFOX_LOGGING_DIR" ]; then
  _archive="${WHITEFOX_LOGGING_DIR}.$(date +%s)"
  if mv "$WHITEFOX_LOGGING_DIR" "$_archive" 2>/dev/null; then
    echo "[$(date)] Archived prior logging dir to $_archive"
  fi
fi
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

BASE_CONFIG_PATH="${1:-xilo_xla/config/generator.toml}"

if [ ! -d "$PROJECT_ROOT/$WHITEFOX_PROMPTS_DIR" ]; then
  echo "ERROR: WHITEFOX_PROMPTS_DIR not found: $PROJECT_ROOT/$WHITEFOX_PROMPTS_DIR" >&2
  exit 1
fi


TMP_CONFIG="$(mktemp -t whitefox-config-XXXXXX.toml)"
trap 'rm -f "$TMP_CONFIG"' EXIT
python3 - "$PROJECT_ROOT/$BASE_CONFIG_PATH" "$TMP_CONFIG" \
    "$WHITEFOX_MODEL" "$WHITEFOX_PROMPTS_DIR" <<'PATCH_PY'
import sys, re

src, dst, model, prompts_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(src) as f:
    lines = f.readlines()

# Patch section-aware: track which [section] we are in so we only replace
# name = "..." inside [model] and optimizations_dir = "..." inside [generation].
# A plain count=1 regex on the full file hits [sut] name = "xla" first, which
# overwrites the SUT name with the model string and causes a KeyError at startup.
current_section = None
model_done = False
prompts_done = False
out = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('[') and not stripped.startswith('[['):
        current_section = re.match(r'^\[(\w+)\]', stripped)
        current_section = current_section.group(1) if current_section else None

    if current_section == 'model' and not model_done and re.match(r'^name\s*=\s*"', stripped):
        line = re.sub(r'^(name\s*=\s*)"[^"]*"', rf'\1"{model}"', line)
        model_done = True
    elif current_section == 'generation' and not prompts_done and re.match(r'^optimizations_dir\s*=\s*"', stripped):
        line = re.sub(r'^(optimizations_dir\s*=\s*)"[^"]*"', rf'\1"{prompts_dir}"', line)
        prompts_done = True
    out.append(line)

with open(dst, "w") as f:
    f.writelines(out)

print(f"[config patch] model={model!r}  optimizations_dir={prompts_dir!r}")
PATCH_PY

CONFIG_PATH="$TMP_CONFIG"

echo "[$(date)] Model:         $WHITEFOX_MODEL_DISPLAY  [$WHITEFOX_MODEL]"
echo "[$(date)] Wheel:         $WHITEFOX_TF_WHEEL"
echo "[$(date)] Prompts dir:   $WHITEFOX_PROMPTS_DIR"
echo "[$(date)] Base config:   $BASE_CONFIG_PATH"
echo "[$(date)] Patched config: $TMP_CONFIG"

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

# ---------------------------------------------------------------------------
# Model-cache pre-seed: copy from NFS to node-local SSD on first use.
# safetensors mmap()s the shard files; if they live on NFS the mapping can
# go stale after the 16-18 min cold load (server evicts cached pages), and
# the first GPU forward pass triggers SIGBUS.  Loading from /data (SSD)
# eliminates both the mmap-over-NFS hazard and the 18-min load penalty.
# ---------------------------------------------------------------------------
_HF_SSD_CACHE=/data/hf_cache
mkdir -p "$_HF_SSD_CACHE"
# HF Hub cache directories are named  models--{org}--{repo}  (/ → --)
_HF_MODEL_SLUG="models--$(echo "$WHITEFOX_MODEL" | sed 's|/|--|g')"
_HF_NFS_SRC="$PROJECT_ROOT/hf_cache/$_HF_MODEL_SLUG"
_HF_SSD_DST="$_HF_SSD_CACHE/$_HF_MODEL_SLUG"
if [ -d "$_HF_NFS_SRC" ] && [ ! -d "$_HF_SSD_DST" ]; then
  echo "[$(date)] [$BATCH_LABEL] Pre-seeding model cache from NFS to SSD (~30 GB, ~60 s)…"
  # flock guards against two array tasks racing on the same node
  (
    flock -x 200
    if [ ! -d "$_HF_SSD_DST" ]; then
      cp -a "$_HF_NFS_SRC" "${_HF_SSD_DST}.tmp" && \
        mv "${_HF_SSD_DST}.tmp" "$_HF_SSD_DST"
    fi
  ) 200>/tmp/wf_hf_preseed_"${_HF_MODEL_SLUG}".lock
  echo "[$(date)] [$BATCH_LABEL] Model cache pre-seed done"
else
  echo "[$(date)] [$BATCH_LABEL] Model cache already on SSD: $_HF_SSD_DST"
fi
export HF_HOME="$_HF_SSD_CACHE"
export HUGGINGFACE_HUB_CACHE="$_HF_SSD_CACHE"
export WHITEFOX_HF_CACHE="$_HF_SSD_CACHE"

XLA_DUMP_DIR="$LOCAL_COV_DIR/xla_dump"
rm -rf "$XLA_DUMP_DIR" 2>/dev/null || true
rm -rf /tmp/wf_profraw_* 2>/dev/null || true
mkdir -p "$XLA_DUMP_DIR"
# Clear the node-local vLLM torch.compile cache.  Stale mmap'd compiled kernels
# left by a previous job on this node can cause SIGBUS when replayed on a new
# CUDA context.  The cache is rebuilt in ~17 s and costs much less than a
# crashed run.
rm -rf /data/vllm_cache/torch_compile_cache 2>/dev/null || true

export XLA_FLAGS="--xla_dump_to=$XLA_DUMP_DIR"
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TOKENIZERS_PARALLELISM=false


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# Restrict CUDA to the single GPU allocated by Slurm.  Without this, the CUDA
# runtime enumerates and opens a context on every visible GPU, mapping each
# 80 GB A100 BAR window into the process address space (~80 GB RSS per GPU).
# With --gres=gpu:1 Slurm already sets CUDA_VISIBLE_DEVICES; this line is a
# belt-and-suspenders guard in case the binding is absent.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


export PYTHONMALLOC=malloc

if [ ! -f "$WHITEFOX_TF_WHEEL" ]; then
  echo "ERROR: TF wheel not found: $WHITEFOX_TF_WHEEL" >&2
  exit 1
fi

case "$WHITEFOX_WHEEL_VERSION" in
  20250806)
    poetry --version || exit 1
    LOCKFILE="$PROJECT_ROOT/.install.lock"
    (
      if flock -w 120 200; then
        echo "[$(date)] [$BATCH_LABEL] Acquired install lock"
      else
        echo "[$(date)] [$BATCH_LABEL] WARN: flock failed/timeout on $(hostname); proceeding without lock"
        : > "$WHITEFOX_LOGGING_DIR/.flock_skipped" 2>/dev/null || true
      fi
      poetry lock --check 2>/dev/null || poetry lock
      poetry install --no-interaction
      echo "[$(date)] Force-reinstalling TensorFlow wheel: $WHITEFOX_TF_WHEEL"
      poetry run pip install --force-reinstall --no-deps "$WHITEFOX_TF_WHEEL"
      # NFS write-back can leave newly written package files invisible for a few
      # seconds after pip returns.  transformers imports TF at module-load time
      # (image_transforms.py), so we verify here and retry if needed.
      for _tf_attempt in 1 2 3; do
        if poetry run python -c "import tensorflow; print('[$(date)] TF import OK:', tensorflow.__version__)" 2>&1; then
          break
        fi
        echo "[$(date)] WARN: TF import failed (attempt $_tf_attempt/3); sleeping 15 s for NFS sync..."
        sleep 15
        if [ "$_tf_attempt" -eq 3 ]; then
          echo "[$(date)] ERROR: TF import failed after 3 attempts, aborting" >&2
          exit 1
        fi
      done
      echo "[$(date)] [$BATCH_LABEL] Install done"
    ) 200>"$LOCKFILE"

    # ---------------------------------------------------------------------------
    # Venv-to-SSD copy: eliminates NFS mmap SIGBUS for Python .so extension modules.
    #
    # The model-weight fix (Bug 3 / commit 259c4b7) moved the safetensors mmaps
    # from NFS to SSD.  The same hazard exists for the .venv: TF, vLLM, and torch
    # each mmap() dozens of .so files via dlopen().  These mappings live for the
    # entire process lifetime.  After ~1 h the NFS server evicts its page cache
    # for those files; the next code-path that touches a reclaimed page delivers
    # SIGBUS to the main Python process (batch0 at it71, batch1 at it44).
    #
    # Fix: identical pattern to the model pre-seed — flock-guarded cp to /data.
    # Subsequent jobs on the same node skip the copy (marker file present).
    # ---------------------------------------------------------------------------
    _VENV_NFS=$(cd "$PROJECT_ROOT" && poetry env info --path 2>/dev/null || true)
    _VENV_SSD="/data/whitefox_venv"
    _VENV_MARKER="$_VENV_SSD/.wf_installed"
    _VENV_EXPECTED="${WHITEFOX_WHEEL_VERSION}"
    if [ -n "$_VENV_NFS" ] && [ -d "$_VENV_NFS" ]; then
      (
        flock -x 203
        if [ ! -f "$_VENV_MARKER" ] || \
           [ "$(cat "$_VENV_MARKER" 2>/dev/null)" != "$_VENV_EXPECTED" ]; then
          echo "[$(date)] [$BATCH_LABEL] Pre-seeding venv from NFS to SSD (~30-90 s)…"
          rm -rf "${_VENV_SSD}.tmp" 2>/dev/null || true
          cp -a "$_VENV_NFS" "${_VENV_SSD}.tmp"
          echo "$_VENV_EXPECTED" > "${_VENV_SSD}.tmp/.wf_installed"
          rm -rf "$_VENV_SSD" 2>/dev/null || true
          mv "${_VENV_SSD}.tmp" "$_VENV_SSD"
          echo "[$(date)] [$BATCH_LABEL] Venv SSD pre-seed done"
        else
          echo "[$(date)] [$BATCH_LABEL] Venv already on SSD: $_VENV_SSD"
        fi
      ) 203>/tmp/wf_venv_preseed.lock
      export VIRTUAL_ENV="$_VENV_SSD"
      RUN_PYTHON="$_VENV_SSD/bin/python"
    else
      echo "[$(date)] WARN: poetry env path not found; falling back to 'poetry run python'"
      RUN_PYTHON="poetry run python"
    fi
    ;;
  20230507)
    VENV_CP310="$PROJECT_ROOT/venv-cp310"
    if [ ! -d "$VENV_CP310" ]; then
      echo "ERROR: venv-cp310 not found. Run setup_venv_20230507.sh once first." >&2
      exit 1
    fi
    source "$VENV_CP310/bin/activate"
    echo "[$(date)] Force-reinstalling TensorFlow wheel: $WHITEFOX_TF_WHEEL"
    pip install --force-reinstall --no-deps "$WHITEFOX_TF_WHEEL"
    echo "[$(date)] [$BATCH_LABEL] Install done"
    # Same venv-to-SSD pattern as the 20250806 case.
    _VENV_SSD_310="/data/whitefox_venv_cp310"
    _VENV_MARKER_310="$_VENV_SSD_310/.wf_installed"
    (
      flock -x 204
      if [ ! -f "$_VENV_MARKER_310" ] || \
         [ "$(cat "$_VENV_MARKER_310" 2>/dev/null)" != "${WHITEFOX_WHEEL_VERSION}" ]; then
        echo "[$(date)] [$BATCH_LABEL] Pre-seeding venv-cp310 from NFS to SSD…"
        rm -rf "${_VENV_SSD_310}.tmp" 2>/dev/null || true
        cp -a "$VENV_CP310" "${_VENV_SSD_310}.tmp"
        echo "${WHITEFOX_WHEEL_VERSION}" > "${_VENV_SSD_310}.tmp/.wf_installed"
        rm -rf "$_VENV_SSD_310" 2>/dev/null || true
        mv "${_VENV_SSD_310}.tmp" "$_VENV_SSD_310"
        echo "[$(date)] [$BATCH_LABEL] venv-cp310 SSD pre-seed done"
      else
        echo "[$(date)] [$BATCH_LABEL] venv-cp310 already on SSD: $_VENV_SSD_310"
      fi
    ) 204>/tmp/wf_venv_cp310_preseed.lock
    deactivate 2>/dev/null || true
    source "$_VENV_SSD_310/bin/activate"
    RUN_PYTHON="python"
    ;;
esac

if [ ! -f "$BASE_CONFIG_PATH" ] && [ ! -f "$PROJECT_ROOT/$BASE_CONFIG_PATH" ]; then
  echo "Base config not found: $BASE_CONFIG_PATH" >&2
  exit 1
fi


CG_REL=$(awk -F'::' 'NR==1 {print $2}' /proc/self/cgroup 2>/dev/null)
CG_DIR="/sys/fs/cgroup${CG_REL}"
TRACE_FILE="$WHITEFOX_LOGGING_DIR/cgroup_mem.tsv"
XLA_DUMP_GLOB="$LOCAL_COV_DIR/xla_dump"
(

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


GPU_TRACE_FILE="$WHITEFOX_LOGGING_DIR/gpu_mem.tsv"
(
  set +e; set +o pipefail
  printf 'ts\tgpu\tused_mb\tfree_mb\ttotal_mb\n' > "$GPU_TRACE_FILE"
  while true; do
    _ts=$(date +%s)
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
               --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=', ' read -r _idx _used _free _total; do
        printf '%s\t%s\t%s\t%s\t%s\n' "$_ts" "$_idx" "$_used" "$_free" "$_total"
      done >> "$GPU_TRACE_FILE"
    sleep 5
  done
) &
GPU_TRACER_PID=$!

echo "[$(date)] Starting $BATCH_LABEL: --only-opt $OPT_CSV"
echo "[$(date)] cgroup trace: $TRACE_FILE (PID=$TRACER_PID)"
echo "[$(date)] GPU trace:    $GPU_TRACE_FILE (PID=$GPU_TRACER_PID)"


ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
echo "[$(date)] nproc limit: $(ulimit -u)"

GEN_EXIT=0
$RUN_PYTHON -m generation.main --sut xla --config "$CONFIG_PATH" --only-opt "$OPT_CSV" \
  || GEN_EXIT=$?

kill "$TRACER_PID" 2>/dev/null || true
kill "$GPU_TRACER_PID" 2>/dev/null || true
echo "[$(date)] Finished $BATCH_LABEL (exit $GEN_EXIT)"


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
    echo "--- GPU memory at OOM time ---"
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
               --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
    echo "--- last 10 GPU trace rows ---"
    tail -10 "$GPU_TRACE_FILE" 2>/dev/null || echo "(no GPU trace file)"
    echo "--- /tmp usage ---"
    df -h /tmp 2>/dev/null
    du -sh /tmp/xla_dump_* 2>/dev/null
    echo "--- last 20 cgroup_mem.tsv samples ---"
    tail -20 "$TRACE_FILE" 2>/dev/null || echo "(no trace file)"
  } > "$OOM_LOG" 2>&1
  echo "[$(date)] OOM postmortem written to $OOM_LOG"
fi

echo "[$(date)] Copying coverage artifacts to NFS …"
if [ -f "$WHITEFOX_LOGGING_DIR/coverage/merged.profdata" ]; then
  echo "[$(date)] merged.profdata: $(du -h "$WHITEFOX_LOGGING_DIR/coverage/merged.profdata" | cut -f1)"
fi
rm -rf "$LOCAL_COV_DIR" 2>/dev/null || true
echo "[$(date)] Local coverage dir cleaned up"


echo "[$(date)] Aggregating batch summaries …"

_AGG_SCRIPT=$(mktemp -t wf_agg_XXXXXX.py)
trap 'rm -f "$TMP_CONFIG" "$_AGG_SCRIPT"' EXIT

cat > "$_AGG_SCRIPT" << 'PYEOF'
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---- Configuration injected via environment variables ---------------------
project_root   = os.environ.get("PROJECT_ROOT", "")
job_id         = os.environ.get("SLURM_ARRAY_JOB_ID", "")
tf_version     = os.environ.get("WHITEFOX_WHEEL_VERSION", "")
model_name     = os.environ.get("WHITEFOX_MODEL", "")
model_display  = os.environ.get("WHITEFOX_MODEL_DISPLAY", model_name)
llvm_dir       = os.environ.get("WHITEFOX_LLVM_DIR", "")

logging_root      = Path(project_root) / "logging"
combined_summary  = logging_root / "run_summary_combined.log"
combined_coverage = logging_root / "coverage_combined.log"

ORACLE_TYPES = [
    "NaiveFail", "XLAFail", "ACFail",
    "Naive_XLAFail", "Naive_ACFail", "XLA_ACFail",
    "AllFail", "AllPass",
    "AllDiff", "AllDiff_Rand", "AllDiff_LessLikely", "AllDiff_TypeMismatch",
]
BUG_ORACLE_TYPES = frozenset([
    "NaiveFail", "XLAFail", "ACFail",
    "Naive_XLAFail", "Naive_ACFail", "XLA_ACFail",
    "AllDiff", "AllDiff_Rand", "AllDiff_LessLikely", "AllDiff_TypeMismatch",
])

# ---- Batch discovery -------------------------------------------------------
def batch_belongs_to_current_run(batch_dir):
    jf = batch_dir / ".job_id"
    if not jf.exists():
        return False
    return jf.read_text().strip() == job_id

batch_dirs = [
    d for d in sorted(logging_root.glob("batch*"))
    if d.is_dir() and batch_belongs_to_current_run(d)
]
stale = [
    d.name for d in sorted(logging_root.glob("batch*"))
    if d.is_dir() and not batch_belongs_to_current_run(d)
]
if stale:
    print(f"Skipping stale batches (wrong job_id): {stale}")
print(f"Aggregating {len(batch_dirs)} batches: {[d.name for d in batch_dirs]}")

if not batch_dirs:
    print("No matching batch dirs found; skipping aggregation.")
    sys.exit(0)

# ---- Load run_stats.json sidecars ------------------------------------------
all_stats = []
missing_json = []
for batch_dir in batch_dirs:
    stats_file = batch_dir / "run_stats.json"
    if stats_file.exists():
        try:
            all_stats.append((batch_dir.name, json.loads(stats_file.read_text())))
        except Exception as e:
            print(f"Warning: could not parse {stats_file}: {e}")
            missing_json.append(batch_dir.name)
    else:
        print(f"Warning: run_stats.json missing in {batch_dir.name} "
              "(batch may use an older code version)")
        missing_json.append(batch_dir.name)

if missing_json:
    print(f"Batches without run_stats.json (absent from Tables 3-5): {missing_json}")

# ---- Aggregate opt_stats (Tables 1, 3) -------------------------------------
agg_opt = defaultdict(lambda: defaultdict(int))
for _bname, stats in all_stats:
    for opt, od in stats.get("opt_stats", {}).items():
        for k, v in od.items():
            if isinstance(v, int):
                agg_opt[opt][k] += v

# ---- Aggregate oracle_counts (Tables 4, 5) ----------------------------------
agg_oracle = defaultdict(lambda: defaultdict(int))
for _bname, stats in all_stats:
    for opt, oc in stats.get("oracle_counts", {}).items():
        for otype, cnt in oc.items():
            agg_oracle[opt][otype] += cnt

# ---- Aggregate Thompson sampling (Statistical Tests) ------------------------
agg_thompson = defaultdict(lambda: {"n": 0, "sum_alpha": 0.0, "sum_beta": 0.0})
for _bname, stats in all_stats:
    for opt, td in stats.get("thompson", {}).items():
        agg_thompson[opt]["n"]         += td.get("n", 0)
        agg_thompson[opt]["sum_alpha"] += td.get("sum_alpha", 0.0)
        agg_thompson[opt]["sum_beta"]  += td.get("sum_beta",  0.0)

all_opt_names = sorted(set(list(agg_opt.keys()) + list(agg_oracle.keys())))

# ---- Helpers ---------------------------------------------------------------
def pct(n, total):
    if total <= 0:
        return "  0.0%"
    return f"{100.0 * n / total:5.1f}%"

def section(f, title):
    f.write("\n" + "=" * 95 + "\n")
    f.write(f"{title}\n")
    f.write("=" * 95 + "\n\n")

QUALITY_KEYS = [
    ("syntax_valid",         "syntactically valid Python"),
    ("imports_successfully", "imports successfully"),
    ("eager_executable",     "eager executable"),
    ("xla_compilable",       "XLA compilable"),
    ("invalid_tf_api",       "invalid TensorFlow API usage"),
    ("unsupported_by_xla",   "unsupported by XLA"),
    ("timeout",              "timeout"),
]

# ========================== COVERAGE UNION (computed before summary) =========
def find_llvm_tool(name):
    if llvm_dir:
        p = os.path.join(llvm_dir, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    found = shutil.which(name)
    return found if found else name

def find_tf_so_files():
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = "/dev/null"
    r = subprocess.run(
        [sys.executable, "-c", "import tensorflow as tf; print(tf.__file__)"],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        print(f"Warning: cannot locate TensorFlow: {r.stderr.strip()[:200]}")
        return []
    tf_dir = Path(r.stdout.strip()).parent
    so_files = sorted(str(p) for p in tf_dir.rglob("*.so"))
    print(f"Found {len(so_files)} TF .so files under {tf_dir}")
    return so_files

profdata_files = []
for bd in batch_dirs:
    pf = bd / "coverage" / "merged.profdata"
    if pf.exists() and pf.stat().st_size > 0:
        profdata_files.append(pf)
    else:
        print(f"  {bd.name}: no merged.profdata")

cov_union_text = ""
union_profdata = logging_root / "coverage_union.profdata"

if profdata_files:
    profdata_tool = find_llvm_tool("llvm-profdata")
    cov_tool      = find_llvm_tool("llvm-cov")
    tmp_pd        = union_profdata.with_suffix(".profdata.tmp")

    print(f"Merging {len(profdata_files)} profdata files …")
    merge_cmd = [
        profdata_tool, "merge", "-sparse", "--failure-mode=all",
        "-o", str(tmp_pd),
    ] + [str(p) for p in profdata_files]

    mr = subprocess.run(merge_cmd, capture_output=True, text=True)
    if mr.returncode != 0:
        print(f"llvm-profdata merge failed: {mr.stderr.strip()[:500]}")
    else:
        tmp_pd.replace(union_profdata)
        print(f"Union profdata: {union_profdata}  "
              f"({union_profdata.stat().st_size // 1024} KB)")

        so_files = find_tf_so_files()
        if so_files:
            XLA_MARKERS = (
                "/xla/", "xla/",
                "/tensorflow/compiler/xla/", "tensorflow/compiler/xla/",
                "/third_party/xla/", "third_party/xla/",
            )
            cov_cmd = [cov_tool, "report", so_files[0],
                       f"-instr-profile={union_profdata}"]
            for so in so_files[1:]:
                cov_cmd += ["-object", so]

            cr = subprocess.run(cov_cmd, capture_output=True, text=True)
            if cr.returncode != 0:
                print(f"llvm-cov report failed: {cr.stderr.strip()[:500]}")
            else:
                report_lines = cr.stdout.splitlines()
                lines_col = 7
                for ln in report_lines:
                    if "Filename" in ln and "Lines" in ln:
                        for i, tok in enumerate(ln.split()):
                            if tok.lower().rstrip(":").startswith("lines"):
                                lines_col = i
                                break
                        break

                xla_total = xla_missed = all_total = all_missed = 0
                xla_files = total_files = 0
                for line in report_lines:
                    parts = line.split()
                    if not parts:
                        continue
                    fn   = parts[0]
                    ints = [int(t.replace(",","")) for t in parts[1:]
                            if t.replace(",","").isdigit()]
                    if len(ints) < 2:
                        continue
                    total, missed = ints[-2], ints[-1]
                    if fn.startswith("TOTAL"):
                        all_total  = total
                        all_missed = missed
                        continue
                    total_files += 1
                    if any(m in fn for m in XLA_MARKERS):
                        xla_files  += 1
                        xla_total  += total
                        xla_missed += missed

                xla_hit = xla_total - xla_missed
                all_hit = all_total - all_missed
                xla_pct = (xla_hit / xla_total * 100) if xla_total else 0.0
                cov_union_text = (
                    f"Union of {len(profdata_files)} batch profdata files "
                    f"({', '.join(d.name for d in batch_dirs)}):\n\n"
                    f"  XLA lines hit:     {xla_hit:>10,}\n"
                    f"  XLA lines total:   {xla_total:>10,}\n"
                    f"  XLA coverage:      {xla_pct:>9.2f}%\n"
                    f"  XLA source files:  {xla_files:>10,}\n"
                    f"\n"
                    f"  All TF lines hit:  {all_hit:>10,}\n"
                    f"  All TF lines total:{all_total:>10,}\n"
                    f"  All TF files:      {total_files:>10,}\n"
                )
                print(f"Union XLA coverage: {xla_hit:,} / {xla_total:,} lines ({xla_pct:.2f}%)")
        else:
            print("No TF .so files; skipping llvm-cov union report.")
else:
    print("No profdata files found; skipping coverage union.")

# ---- Write coverage_combined.log (union first, per-batch for reference) ----
sep60 = "=" * 60
cov_sections = []
if cov_union_text:
    cov_sections.append(f"{sep60}\nCOVERAGE UNION (all batches merged)\n{sep60}\n\n"
                        + cov_union_text + "\n")
cov_sections.append(f"{sep60}\nPER-BATCH COVERAGE (for reference)\n{sep60}\n")
for bd in batch_dirs:
    cov_file = bd / "coverage_report.log"
    if not cov_file.exists():
        cov_sections.append(f"--- {bd.name}: no coverage_report.log ---\n")
        continue
    text = cov_file.read_text().strip()
    label = f"--- {bd.name}: coverage unavailable ---" if (
        "UNAVAILABLE" in text and len(text) < 200) else f"--- {bd.name} ---"
    cov_sections.append(f"{label}\n{text}\n")

with open(combined_coverage, "w") as f:
    f.write("\n".join(cov_sections) + "\n")
print(f"Combined coverage written:  {combined_coverage}")

# ========================== WRITE COMBINED SUMMARY =========================
with open(combined_summary, "w") as f:

    # ---- Preamble ----------------------------------------------------------
    f.write("=" * 95 + "\n")
    f.write("WHITEFOX COMBINED RUN SUMMARY (all batches)\n")
    f.write("=" * 95 + "\n\n")

    tf_label = tf_version or "unknown"
    if tf_version == "20250806":
        tf_label = "20250806  (tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806, cp312)"
    elif tf_version == "20230507":
        tf_label = "20230507  (tensorflow_cpu-2.14.0+selfbuilt.20230507, cp310)"

    f.write(f"TF Version:  {tf_label}\n")
    model_label = model_display if model_display else (model_name or "unknown")
    if model_name and model_name != model_display:
        model_label = f"{model_display}  [{model_name}]"
    f.write(f"Model:       {model_label}\n")
    f.write(f"Batches:     {len(batch_dirs)}  ({', '.join(d.name for d in batch_dirs)})\n")
    f.write(f"Aggregated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- Table 1: Test Generation ------------------------------------------
    section(f, "TABLE 1 — TEST GENERATION")

    mode_keys   = sorted({k for d in agg_opt.values() for k in d if k.startswith("success_")})
    mode_labels = [k.replace("success_", "") for k in mode_keys]

    t1_header = (f"{'Optimization':40s} | {'Created':>7s} | {'Valid':>7s} | "
                 f"{'Invalid':>7s} | {'Triggered':>9s}")
    for lbl in mode_labels:
        t1_header += f" | {lbl.capitalize():>11s}"
    sep_len = max(95, len(t1_header) + 5)
    f.write(t1_header + "\n")
    f.write("-" * sep_len + "\n\n")

    grand = defaultdict(int)
    for opt in all_opt_names:
        d    = agg_opt[opt]
        gen  = d.get("generated", 0)
        valid = d.get("valid", 0)
        inv  = d.get("invalid", 0)
        trig = d.get("triggered", 0)
        grand["generated"] += gen
        grand["valid"]     += valid
        grand["invalid"]   += inv
        grand["triggered"] += trig
        f.write(f"{opt:40s} | {gen:7d} | {valid:7d} | {inv:7d} | {trig:9d}")
        for mk in mode_keys:
            v = d.get(mk, 0)
            grand[mk] += v
            f.write(f" | {v:11d}")
        f.write("\n")

    f.write("-" * sep_len + "\n")
    f.write(f"{'TOTAL':40s} | {grand['generated']:7d} | {grand['valid']:7d} | "
            f"{grand['invalid']:7d} | {grand['triggered']:9d}")
    for mk in mode_keys:
        f.write(f" | {grand[mk]:11d}")
    f.write("\n" + "=" * sep_len + "\n")

    # ---- Table 2: Code Coverage (union) ------------------------------------
    section(f, "TABLE 2 — CODE COVERAGE (union of all batches)")
    if cov_union_text:
        f.write(cov_union_text + "\n")
        f.write("  (See coverage_combined.log for per-batch breakdown.)\n")
    else:
        f.write("  Coverage union not available (no profdata files or llvm-cov failed).\n")
        f.write("  See coverage_combined.log for per-batch results.\n")
    f.write("=" * 95 + "\n")

    # ---- Table 3: Generation Quality Distribution --------------------------
    section(f, "TABLE 3 — GENERATION QUALITY DISTRIBUTION (pre-oracle)")

    gen_total   = sum(agg_opt[opt].get("generated", 0)     for opt in all_opt_names)
    exec_total  = sum(agg_opt[opt].get("executed", 0)      for opt in all_opt_names)
    wfail_total = sum(agg_opt[opt].get("worker_failed", 0) for opt in all_opt_names)

    f.write(f"{'Generated test category':40s} | {'Count':>7s} | {'%':>7s}\n")
    f.write("-" * 60 + "\n\n")
    for key, label in QUALITY_KEYS:
        cnt = sum(agg_opt[opt].get(key, 0) for opt in all_opt_names)
        f.write(f"{label:40s} | {cnt:7d} | {pct(cnt, gen_total):>7s}\n")

    f.write(f"\nGenerated (denominator): {gen_total}\n")
    f.write(f"Executed:                {exec_total}\n")
    f.write(f"Worker failures:         {wfail_total}\n")
    f.write("=" * 60 + "\n")

    f.write("\nPer optimization (% of generated tests):\n\n")
    q_header = (f"{'Optimization':40s} | {'Gen':>5s} | "
                f"{'Syntax':>6s} | {'Import':>6s} | {'Eager':>6s} | "
                f"{'XLA':>6s} | {'JIT':>6s} | "
                f"{'BadAPI':>6s} | {'Unsup':>6s} | {'T/O':>5s}")
    f.write(q_header + "\n")
    f.write("-" * 115 + "\n\n")

    for opt in all_opt_names:
        d   = agg_opt[opt]
        gen = d.get("generated", 0)
        jit = next((d[k] for k in ("success_xla","success_jit","success_compiled") if k in d), 0)
        f.write(f"{opt:40s} | {gen:5d} | "
                f"{pct(d.get('syntax_valid',0),gen):>6s} | "
                f"{pct(d.get('imports_successfully',0),gen):>6s} | "
                f"{pct(d.get('eager_executable',0),gen):>6s} | "
                f"{pct(d.get('xla_compilable',0),gen):>6s} | "
                f"{pct(jit,gen):>6s} | "
                f"{d.get('invalid_tf_api',0):6d} | "
                f"{d.get('unsupported_by_xla',0):6d} | "
                f"{d.get('timeout',0):5d}\n")
    f.write("=" * 115 + "\n")

    # ---- Table 4: Oracle Outcomes ------------------------------------------
    section(f, "TABLE 4 — ORACLE OUTCOMES")

    col_w     = 14
    oc_header = f"{'Optimization':40s}"
    for ot in ORACLE_TYPES:
        oc_header += f" | {ot:>{col_w}s}"
    oc_header += f" | {'Total':>7s}"
    oc_sep_len = len(oc_header) + 2
    f.write(oc_header + "\n")
    f.write("-" * oc_sep_len + "\n\n")

    oc_grand = defaultdict(int)
    for opt in all_opt_names:
        counts    = agg_oracle[opt]
        row_total = sum(counts.get(ot, 0) for ot in ORACLE_TYPES)
        oc_grand["Total"] += row_total
        f.write(f"{opt:40s}")
        for ot in ORACLE_TYPES:
            v = counts.get(ot, 0)
            oc_grand[ot] += v
            f.write(f" | {v:>{col_w}d}")
        f.write(f" | {row_total:>7d}\n")

    f.write("-" * oc_sep_len + "\n")
    f.write(f"{'TOTAL':40s}")
    for ot in ORACLE_TYPES:
        f.write(f" | {oc_grand[ot]:>{col_w}d}")
    f.write(f" | {oc_grand['Total']:>7d}\n")
    f.write("=" * oc_sep_len + "\n")

    # ---- Table 5: Bug Pipeline ---------------------------------------------
    section(f, "TABLE 5 — BUG PIPELINE  (automated fields only)")

    f.write("NOTE: Only 'RawFails' is tracked automatically.\n"
            "      Filtered / Deduped / Triaged / Inspected / Likely Bugs /\n"
            "      Reported / Accepted / Duplicates / False Positives\n"
            "      all require manual post-processing review.\n\n")
    f.write(f"{'Optimization':40s} | {'RawFails':>9s}\n")
    f.write("-" * 55 + "\n\n")

    bp_total = 0
    for opt in all_opt_names:
        counts = agg_oracle[opt]
        raw    = sum(counts.get(bt, 0) for bt in BUG_ORACLE_TYPES)
        bp_total += raw
        f.write(f"{opt:40s} | {raw:9d}\n")

    f.write("-" * 55 + "\n")
    f.write(f"{'TOTAL':40s} | {bp_total:9d}\n")
    f.write("=" * 55 + "\n")

    # ---- Statistical Tests: Thompson Sampling ------------------------------
    section(f, "STATISTICAL TESTS — THOMPSON SAMPLING")

    f.write(f"{'Optimization':40s} | {'Tests':>5s} | "
            f"{'Avg Alpha':>9s} | {'Avg Beta':>9s} | {'Avg Theta':>9s}\n")
    f.write("-" * 85 + "\n\n")

    for opt in all_opt_names:
        td = agg_thompson.get(opt, {})
        n  = td.get("n", 0)
        if n == 0:
            f.write(f"{opt:40s} |     0 |         - |         - |         -\n")
            continue
        avg_a     = td["sum_alpha"] / n
        avg_b     = td["sum_beta"]  / n
        avg_theta = avg_a / (avg_a + avg_b) if (avg_a + avg_b) > 0 else 0.0
        f.write(f"{opt:40s} | {n:5d} | {avg_a:9.2f} | {avg_b:9.2f} | {avg_theta:9.4f}\n")

    f.write("=" * 85 + "\n")

print(f"Combined run summary written: {combined_summary}")

PYEOF

export PROJECT_ROOT SLURM_ARRAY_JOB_ID WHITEFOX_WHEEL_VERSION WHITEFOX_MODEL WHITEFOX_MODEL_DISPLAY WHITEFOX_LLVM_DIR
$RUN_PYTHON "$_AGG_SCRIPT" 2>&1 || echo "[$(date)] Aggregation failed (non-fatal)"
rm -f "$_AGG_SCRIPT"

exit $GEN_EXIT
