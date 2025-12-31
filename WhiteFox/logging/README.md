# WhiteFox Logging Directory

This directory contains comprehensive logs for debugging WhiteFox generation issues.

## Directory Structure

```
logging/
├── prompts/              # Prompts sent to LLM (base and feedback)
├── generated_code/       # Raw and cleaned generated code
├── execution_results/    # Execution results and pass detection analysis
├── state_changes/        # Bandit state updates after each iteration
├── errors/              # Error logs with full context
├── execution_logs/      # Original execution logs (compatibility)
├── bug_reports/         # Bug reports from oracles
├── generated-outputs/   # All generated test files and outputs (moved from root)
│   └── whitefox_tests/  # Generated test files per optimization
└── whitefox_state.json  # Bandit state file (moved from root)
```

## Log File Formats

### Prompts (`prompts/`)
- Format: `{optimization}-it{iteration}-{base|feedback}.json`
- Contains: Prompt text, example tests used, metadata

### Generated Code (`generated_code/`)
- Format: `{optimization}-it{iteration}-sample{sample_idx}.json`
- Contains: Raw LLM output, cleaned code, cleaning changes applied

### Execution Results (`execution_results/`)
- Format: `{optimization}-it{iteration}-sample{sample_idx}.json`
- Contains: Compile/runtime success, errors, pass triggering status
- Pass Analysis: `{optimization}-it{iteration}-sample{sample_idx}-pass-analysis.json`
- Contains: Detailed pass detection analysis, log text snippets, pattern matches

### State Changes (`state_changes/`)
- Format: `{optimization}-it{iteration}.json`
- Contains: Before/after state, iteration stats, alpha/beta updates

### Errors (`errors/`)
- Format: `{optimization}-it{iteration}-{error_type}.json`
- Contains: Error message, traceback, context

## Consolidated Files

All WhiteFox output files are now in the logging directory:
- **whitefox_state.json** - Bandit state (moved from WhiteFox root)
- **generated-outputs/** - All generated tests and outputs (moved from WhiteFox root)
  - Contains `whitefox_tests/` with all generated test files

## Key Debugging Information

### Why no few-shot examples?
1. Check `whitefox_state.json` - look at `triggering_tests` for each optimization
2. Check `state_changes/` - look for `num_triggering_tests` staying at 0
3. Check `execution_results/` - look for `pass_triggered: false`
4. Check `execution_results/*-pass-analysis.json` - see if passes are being detected in logs

### Why code generation failures?
1. Check `generated_code/` - compare `raw_text` vs `cleaned_code`
2. Check `errors/` - look for `code_cleaning_error` or `compile_error`
3. Check if markdown blocks are being extracted correctly

### Why only a few optimizations run?
1. Check main log file for errors
2. Check `errors/optimization_processing_error*.json`
3. Check `whitefox_state.json` - see which optimizations have been processed

## Sanity Check Report

A comprehensive sanity check report is automatically generated at the start and end of each run:
- `sanity_check_report.json` - Machine-readable JSON format
- `sanity_check_report.txt` - Human-readable text format (easy to paste in chat)

The sanity check verifies:
- Code cleaning logic (markdown extraction, import addition)
- API validation (invalid API detection)
- Prompt length limiting (token counting)
- State management (triggering tests structure)
- File structure (all required directories)
- Recent activity (system running)
- Code quality (sample analysis)

To run manually:
```bash
cd WhiteFox
python run_sanity_check.py
```

## Log Analysis Tips

1. **Find all failed compilations:**
   ```bash
   grep -l '"compile_success_naive": false' execution_results/*.json
   ```

2. **Find all triggered passes:**
   ```bash
   grep -l '"pass_triggered": true' execution_results/*.json
   ```

3. **Check pass detection issues:**
   ```bash
   grep -l '"expected_pass_triggered": false' execution_results/*-pass-analysis.json
   ```

4. **Find code cleaning issues:**
   ```bash
   grep -l '"had_markdown": true' generated_code/*.json
   ```

