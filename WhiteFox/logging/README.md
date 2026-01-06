# WhiteFox Logging Directory

This directory contains comprehensive logs for debugging WhiteFox generation issues.

## Directory Structure

```
logging/
├── source/                          # JSON source files (structured data)
│   ├── prompts.json                # All prompts sent to LLM
│   ├── cleaned_text_before_and_after.json  # Code before/after cleaning
│   ├── all_cleaned_code.json       # Final cleaned code
│   ├── execution_results.json      # Test execution results
│   ├── pass_detection_analysis.json # Pass detection details
│   ├── state_changes.json          # Bandit state updates
│   ├── errors.json                 # All errors
│   ├── bug_reports.json            # Bug reports from oracles
│   └── execution_diagnostics.json  # Execution diagnostics
├── prompts_readable.log             # Prompts with actual newlines (easy to read)
├── cleaned_code_readable.log        # Cleaned code with actual newlines
├── code_before_after_readable.log   # Before/after comparison with actual newlines
├── execution_logs/                  # Raw execution logs per test
├── generated-outputs/               # All generated test files and outputs
│   └── whitefox_tests/             # Generated test files per optimization
├── whitefox_state.json             # Bandit state file (moved from root)
├── sanity_check_report.json        # Sanity check report (JSON)
└── sanity_check_report.txt         # Sanity check report (readable text)
```

## Log File Organization

### JSON Source Files (`source/`)
These contain structured data in JSON format with escaped newlines (`\n`). Use these for:
- Programmatic analysis
- Parsing with scripts
- Machine-readable data

### Readable Text Files (root logging directory)
These contain the same content as JSON files but with actual newlines for easy reading. Use these for:
- Quick inspection of prompts and generated code
- Copying code snippets
- Manual review and debugging

**Key readable files:**
- **prompts_readable.log** - All prompts with properly formatted code examples
- **cleaned_code_readable.log** - All cleaned generated code, properly formatted
- **code_before_after_readable.log** - Side-by-side comparison of raw LLM output vs cleaned code

## Log File Formats

### Prompts (JSON: `source/prompts.json`, Text: `prompts_readable.log`)
- Organized by optimization
- Contains: Prompt text, example tests used, metadata
- Readable version shows prompts with actual newlines for easy reading

### Generated Code (JSON: `source/all_cleaned_code.json`, Text: `cleaned_code_readable.log`)
- Organized by optimization, iteration, and sample
- Contains: Final cleaned code ready for execution
- Readable version shows code with proper formatting

### Code Before/After (JSON: `source/cleaned_text_before_and_after.json`, Text: `code_before_after_readable.log`)
- Shows raw LLM output alongside cleaned code
- Useful for debugging code cleaning logic
- Readable version allows easy comparison

### Execution Results (JSON: `source/execution_results.json`)
- Contains: Compile/runtime success, errors, pass triggering status
- Organized by optimization

### Pass Detection Analysis (JSON: `source/pass_detection_analysis.json`)
- Contains: Detailed pass detection analysis, triggered passes
- Minimal summary format

### State Changes (JSON: `source/state_changes.json`)
- Contains: Bandit state updates, alpha/beta values, test counts
- Organized by optimization and iteration

### Errors (JSON: `source/errors.json`)
- Contains: Error message, traceback, context
- All errors consolidated

### Bug Reports (JSON: `source/bug_reports.json`)
- Contains: Oracle-detected bugs with details
- All bug reports consolidated

## Consolidated Files

All WhiteFox output files are now in the logging directory:
- **whitefox_state.json** - Bandit state (moved from WhiteFox root)
- **generated-outputs/** - All generated tests and outputs (moved from WhiteFox root)
  - Contains `whitefox_tests/` with all generated test files

## Key Debugging Information

### Why no few-shot examples?
1. Check `whitefox_state.json` - look at `triggering_tests` for each optimization
2. Check `source/state_changes.json` - look for `tests_after` staying at 0
3. Check `source/execution_results.json` - look for `pass_triggered: false`
4. Check `source/pass_detection_analysis.json` - see if passes are being detected

### Why code generation failures?
1. Check `code_before_after_readable.log` - compare raw vs cleaned code (easy to read!)
2. Check `source/cleaned_text_before_and_after.json` - structured comparison
3. Check `source/errors.json` - look for code cleaning or compile errors
4. Check if markdown blocks are being extracted correctly

### Why only a few optimizations run?
1. Check main log file for errors
2. Check `source/errors.json` - look for optimization_processing_error entries
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

### Using JSON Source Files (for programmatic analysis)

1. **Find all failed compilations:**
   ```bash
   grep '"compile_success_naive": false' source/execution_results.json
   ```

2. **Find all triggered passes:**
   ```bash
   grep '"pass_triggered": true' source/execution_results.json
   ```

3. **Check pass detection issues:**
   ```bash
   grep '"triggered": false' source/pass_detection_analysis.json
   ```

4. **Count errors by type:**
   ```bash
   jq '[.[] | .error_type] | group_by(.) | map({type: .[0], count: length})' source/errors.json
   ```

### Using Readable Log Files (for manual review)

1. **Review all prompts with proper formatting:**
   ```bash
   less prompts_readable.log
   ```

2. **Compare raw LLM output vs cleaned code:**
   ```bash
   less code_before_after_readable.log
   ```

3. **Review all generated code:**
   ```bash
   less cleaned_code_readable.log
   ```

