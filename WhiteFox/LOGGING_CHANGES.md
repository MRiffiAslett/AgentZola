# WhiteFox Logging and Code Generation Fixes

## Summary of Changes

This update addresses several critical issues in WhiteFox code generation and adds comprehensive logging to help debug problems.

## Issues Fixed

### 1. Missing TensorFlow Imports
**Problem:** Generated code was missing `import tensorflow as tf`, causing `NameError: name 'tf' is not defined`.

**Solution:** Added `code_cleaner.py` with `ensure_imports()` function that automatically adds required imports when code uses `tf` or `np` but doesn't import them.

### 2. Markdown Code Blocks in Generated Code
**Problem:** LLM was including markdown formatting (```python ... ```) in generated code, causing syntax errors.

**Solution:** Added `extract_code_from_markdown()` function that extracts Python code from markdown code blocks and removes formatting artifacts.

### 3. Input Variable Name Mismatch
**Problem:** Harness expected `input_data` but generated code used variables like `x1`, `x`, etc.

**Solution:** Updated harness to automatically detect input variables by:
- Looking for model calls like `m(x1)` or `m.call(x)`
- Falling back to common variable names (`x1`, `x`, `input_data`, etc.)

### 4. No Few-Shot Examples Being Used
**Problem:** `triggering_tests` remained empty because:
- Tests were failing to compile/run (missing imports, syntax errors)
- Even when tests ran, pass detection might not be working correctly

**Solution:** 
- Fixed code generation issues so tests can actually run
- Added comprehensive logging to track why passes aren't being triggered

## New Logging System

### Location
All logs are now consolidated in `WhiteFox/logging/` directory.

### Log Types

1. **Prompts** (`logging/prompts/`)
   - Base and feedback prompts sent to LLM
   - Example tests included in prompts
   - Metadata about prompt generation

2. **Generated Code** (`logging/generated_code/`)
   - Raw LLM output
   - Cleaned code after processing
   - Details of cleaning changes (markdown removal, import additions)

3. **Execution Results** (`logging/execution_results/`)
   - Compile/runtime success/failure for all three modes
   - Pass triggering status
   - Error messages
   - Pass detection analysis (separate files with `-pass-analysis.json` suffix)

4. **State Changes** (`logging/state_changes/`)
   - Bandit state before/after each iteration
   - Number of triggered vs not triggered tests
   - New triggering tests added
   - Alpha/beta updates for example tests

5. **Errors** (`logging/errors/`)
   - All errors with full context
   - Tracebacks
   - Error type classification

6. **Execution Logs** (`logging/execution_logs/`)
   - Original execution logs (for compatibility)
   - Moved from `generated-outputs/whitefox_logs/`

7. **Bug Reports** (`logging/bug_reports/`)
   - Bug reports from oracles
   - Moved from `generated-outputs/whitefox_bugs/`

## Usage

### Running with New Logging

The logging system is automatically enabled when you run WhiteFox:

```bash
python -m generation.main --config path/to/config.toml
```

Logs will be created in `WhiteFox/logging/` automatically.

### Consolidating Existing Logs

To move existing logs to the new location:

```bash
cd WhiteFox
python consolidate_logs.py
```

This will:
- Move `generated-outputs/whitefox_logs/` → `logging/execution_logs/`
- Move `generated-outputs/whitefox_bugs/` → `logging/bug_reports/`

### Debugging Workflow

1. **Check why no few-shot examples:**
   ```bash
   # Check state changes
   cat logging/state_changes/AllReduceSimplifier-it0.json | jq '.after.num_triggering_tests'
   
   # Check if passes are being detected
   grep -l '"pass_triggered": true' logging/execution_results/*.json
   
   # Check pass detection analysis
   cat logging/execution_results/AllReduceSimplifier-it0-sample0-pass-analysis.json
   ```

2. **Check code generation issues:**
   ```bash
   # See raw vs cleaned code
   cat logging/generated_code/AllReduceSimplifier-it0-sample0.json | jq '.raw_text, .cleaned_code'
   
   # Check what cleaning was applied
   cat logging/generated_code/AllReduceSimplifier-it0-sample0.json | jq '.cleaning_changes'
   ```

3. **Check compilation errors:**
   ```bash
   # Find all failed compilations
   grep -l '"compile_success_naive": false' logging/execution_results/*.json
   
   # See error details
   cat logging/execution_results/AllReduceSimplifier-it0-sample0.json | jq '.compile_error_naive'
   ```

4. **Check pass detection:**
   ```bash
   # See which passes were detected
   cat logging/execution_results/AllReduceSimplifier-it0-sample0-pass-analysis.json | jq '.triggered_passes, .expected_pass_triggered'
   ```

## Files Changed

- `generation/code_cleaner.py` - NEW: Code cleaning utilities
- `generation/logger.py` - NEW: Comprehensive logging module
- `generation/generator.py` - UPDATED: Uses code cleaning and new logging
- `generation/harness.py` - UPDATED: Flexible input variable detection
- `logging/README.md` - NEW: Documentation for logging structure
- `consolidate_logs.py` - NEW: Script to move existing logs

## Next Steps for Debugging

1. Run WhiteFox with the new logging system
2. Check `logging/execution_results/*-pass-analysis.json` to see if passes are being detected in logs
3. Check `logging/generated_code/` to verify code cleaning is working
4. Check `logging/state_changes/` to see if triggering tests are being added
5. Review `logging/errors/` for any unexpected errors

The comprehensive logging should help identify:
- Why passes aren't being triggered (check pass detection analysis)
- Why code isn't compiling (check generated code and errors)
- Why few-shot examples aren't being selected (check state changes)
- Why only some optimizations are running (check errors)

