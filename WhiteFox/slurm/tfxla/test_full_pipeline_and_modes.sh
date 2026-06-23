#!/bin/bash
# Smoke-test both run modes (20250806 / 20230507) on a CPU node.
# Validates: wheel file exists, prompts dir populated, venv activates,
# TensorFlow imports cleanly, and StarCoder model name passes config patch.
#
# Submit:
#   sbatch WhiteFox/slurm/tfxla/test_modes.sh
#
#SBATCH --job-name=wf_test_modes
#SBATCH --partition=gpucluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/vol/bitbucket/mtr25/AgentZola/WhiteFox/slurm/tfxla/out/test_modes_%j.out
#SBATCH --array=0-1

set -euo pipefail

: "${USER:=$(id -un)}"
: "${HOME:=$(getent passwd "$USER" 2>/dev/null | cut -d: -f6)}"
: "${HOME:=/tmp}"
export HOME USER

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
_WHEEL_DIR="/vol/bitbucket/mtr25/tfbuild/wheels"

# Map array task → mode
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) MODE="20250806" ;;
  1) MODE="20230507" ;;
  *) echo "ERROR: unexpected task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

WHITEFOX_MODEL="bigcode/starcoder"
WHITEFOX_WHEEL_VERSION="$MODE"
WHITEFOX_PROMPTS_VERSION="$MODE"

case "$WHITEFOX_WHEEL_VERSION" in
  20250806) WHITEFOX_TF_WHEEL="$_WHEEL_DIR/tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806-cp312-cp312-linux_x86_64.whl" ;;
  20230507) WHITEFOX_TF_WHEEL="$_WHEEL_DIR/tensorflow_cpu-2.14.0+selfbuilt.20230507-cp310-cp310-linux_x86_64.whl" ;;
esac

case "$WHITEFOX_PROMPTS_VERSION" in
  20250806) WHITEFOX_PROMPTS_DIR="xilo_xla/artifacts/generation-prompts-20250806" ;;
  20230507) WHITEFOX_PROMPTS_DIR="xilo_xla/artifacts/generation-prompts-20230507" ;;
esac

PASS=0
FAIL=0

ok()   { echo "[PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }

echo "============================================================"
echo "MODE: $MODE  (task ${SLURM_ARRAY_TASK_ID:-0})"
echo "Node: $(hostname)  Date: $(date)"
echo "============================================================"

# 1 — wheel file present
if [ -f "$WHITEFOX_TF_WHEEL" ]; then
  ok  "wheel exists: $WHITEFOX_TF_WHEEL"
else
  fail "wheel MISSING: $WHITEFOX_TF_WHEEL"
fi

# 2 — prompts directory present and non-empty
FULL_PROMPTS="$PROJECT_ROOT/$WHITEFOX_PROMPTS_DIR"
if [ -d "$FULL_PROMPTS" ]; then
  N_PROMPTS=$(find "$FULL_PROMPTS" -maxdepth 1 -name '*.txt' | wc -l)
  if [ "$N_PROMPTS" -gt 0 ]; then
    ok  "prompts dir has $N_PROMPTS .txt files: $WHITEFOX_PROMPTS_DIR"
  else
    fail "prompts dir exists but contains no .txt files: $WHITEFOX_PROMPTS_DIR"
  fi
else
  fail "prompts dir MISSING: $FULL_PROMPTS"
fi

# 3 — venv/interpreter available
cd "$PROJECT_ROOT"
case "$MODE" in
  20250806)
    if poetry env info --path &>/dev/null; then
      PYTHON_CMD="$(poetry env info --path)/bin/python"
      ok  "poetry venv found: $PYTHON_CMD"
    else
      fail "poetry venv not found (run: poetry install)"
      PYTHON_CMD=""
    fi
    ;;
  20230507)
    VENV_CP310="$PROJECT_ROOT/venv-cp310"
    if [ -d "$VENV_CP310" ]; then
      PYTHON_CMD="$VENV_CP310/bin/python"
      ok  "venv-cp310 found: $PYTHON_CMD"
    else
      fail "venv-cp310 MISSING — run setup_venv_20230507.sh first"
      PYTHON_CMD=""
    fi
    ;;
esac

# 4 — Python version matches wheel ABI
if [ -n "$PYTHON_CMD" ] && [ -x "$PYTHON_CMD" ]; then
  PY_VER=$("$PYTHON_CMD" -c "import sys; print(sys.version.split()[0])")
  case "$MODE" in
    20250806) EXPECTED_MAJ_MIN="3.12" ;;
    20230507) EXPECTED_MAJ_MIN="3.10" ;;
  esac
  if [[ "$PY_VER" == "$EXPECTED_MAJ_MIN"* ]]; then
    ok  "Python version $PY_VER matches expected $EXPECTED_MAJ_MIN.x"
  else
    fail "Python version $PY_VER — expected $EXPECTED_MAJ_MIN.x"
  fi
fi

# 5 — TensorFlow imports without error
if [ -n "$PYTHON_CMD" ] && [ -x "$PYTHON_CMD" ]; then
  if TF_VER=$("$PYTHON_CMD" -c "import tensorflow as tf; print(tf.__version__)" 2>&1); then
    ok  "import tensorflow OK — version: $TF_VER"
  else
    fail "import tensorflow FAILED:\n$TF_VER"
  fi
fi

# 6 — config patch produces correct model + prompts_dir
if [ -n "$PYTHON_CMD" ] && [ -x "$PYTHON_CMD" ]; then
  TMP_CONFIG="$(mktemp -t wf_test_config_XXXXXX.toml)"
  trap 'rm -f "$TMP_CONFIG"' EXIT

  PATCH_OUT=$(python3 - "$PROJECT_ROOT/xilo_xla/config/generator.toml" "$TMP_CONFIG" \
      "$WHITEFOX_MODEL" "$WHITEFOX_PROMPTS_DIR" <<'PATCH_PY'
import sys, re
src, dst, model, prompts_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src) as f:
    text = f.read()
text = re.sub(r'^(name\s*=\s*)"[^"]*"', rf'\1"{model}"', text, count=1, flags=re.MULTILINE)
text = re.sub(r'^(optimizations_dir\s*=\s*)"[^"]*"', rf'\1"{prompts_dir}"', text, count=1, flags=re.MULTILINE)
with open(dst, "w") as f:
    f.write(text)
print(f"model={model!r}  optimizations_dir={prompts_dir!r}")
PATCH_PY
  )

  if grep -q "name = \"$WHITEFOX_MODEL\"" "$TMP_CONFIG" && \
     grep -q "optimizations_dir = \"$WHITEFOX_PROMPTS_DIR\"" "$TMP_CONFIG"; then
    ok  "config patch OK — $PATCH_OUT"
  else
    fail "config patch did not produce expected values. Patched config:"
    cat "$TMP_CONFIG"
  fi
fi

# 7 — XLA source directory for the matching version present
SRC_DIR="$PROJECT_ROOT/xilo_xla/artifacts/source-code-data-$MODE"
if [ -d "$SRC_DIR" ]; then
  N_SRC=$(find "$SRC_DIR" -maxdepth 1 -name '*.cc' | wc -l)
  ok  "source-code-data-$MODE present ($N_SRC .cc files)"
else
  fail "source-code-data-$MODE MISSING: $SRC_DIR"
fi

echo
echo "------------------------------------------------------------"
echo "Results for MODE=$MODE: $PASS passed, $FAIL failed"
echo "------------------------------------------------------------"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
