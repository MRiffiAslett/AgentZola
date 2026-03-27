#!/bin/bash
#SBATCH --job-name=wf_merge_cov
#SBATCH --partition=a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/output_tf/whitefox_merge_%j.out

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
PER_OPT_DIR="$PROJECT_ROOT/logging/per_opt"
MERGED_DIR="$PROJECT_ROOT/logging/merged"
mkdir -p "$MERGED_DIR"

LLVM17_BIN="/vol/bitbucket/mtr25/tfbuild/llvm17/bin"
if [ -x "$LLVM17_BIN/llvm-profdata" ]; then
  PROFDATA_TOOL="$LLVM17_BIN/llvm-profdata"
else
  PROFDATA_TOOL="$(command -v llvm-profdata 2>/dev/null || true)"
fi

if [ -z "$PROFDATA_TOOL" ]; then
  echo "ERROR: llvm-profdata not found" >&2
  exit 1
fi

echo "[$(date)] Merging per-optimization coverage profiles"
echo "[$(date)] Using: $PROFDATA_TOOL"

PROFDATA_FILES=()
for opt_dir in "$PER_OPT_DIR"/*/coverage; do
  pf="$opt_dir/merged.profdata"
  if [ -f "$pf" ] && [ -s "$pf" ]; then
    PROFDATA_FILES+=("$pf")
    echo "  Found: $pf ($(du -h "$pf" | cut -f1))"
  fi
done

echo "[$(date)] Found ${#PROFDATA_FILES[@]} profdata files to merge"

if [ ${#PROFDATA_FILES[@]} -eq 0 ]; then
  echo "No profdata files found. Nothing to merge."
  exit 0
fi

JOBS="${WHITEFOX_LLVM_PROFDATA_JOBS:-6}"
MERGED_OUTPUT="$MERGED_DIR/merged.profdata"

"$PROFDATA_TOOL" merge -sparse \
  --failure-mode=all \
  -j "$JOBS" \
  -o "$MERGED_OUTPUT" \
  "${PROFDATA_FILES[@]}"

echo "[$(date)] Merged profdata: $MERGED_OUTPUT ($(du -h "$MERGED_OUTPUT" | cut -f1))"

# ---- Collect run summaries --------------------------------------------------
echo ""
echo "=== Per-optimization run summaries ==="
for opt_dir in "$PER_OPT_DIR"/*/; do
  opt_name="$(basename "$opt_dir")"
  summary="$opt_dir/run_summary_detailed.log"
  if [ -f "$summary" ]; then
    echo "--- $opt_name ---"
    head -20 "$summary"
    echo ""
  fi
done

# ---- Collect bug reports ----------------------------------------------------
BUG_DIR="$MERGED_DIR/bug_reports"
mkdir -p "$BUG_DIR"
bug_count=0
for opt_dir in "$PER_OPT_DIR"/*/; do
  if [ -d "$opt_dir/bug_reports" ]; then
    opt_name="$(basename "$opt_dir")"
    for bug in "$opt_dir"/bug_reports/*.json; do
      [ -f "$bug" ] || continue
      cp "$bug" "$BUG_DIR/${opt_name}_$(basename "$bug")"
      bug_count=$((bug_count + 1))
    done
  fi
done
echo "[$(date)] Collected $bug_count bug reports into $BUG_DIR"

echo "[$(date)] Done."
