#!/bin/bash
# Final aggregation: merges all run summaries including pre-patch backups
# into a single correct combined report. Run on cluster after all jobs complete.

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
LOGGING="$PROJECT_ROOT/logging"

cd "$PROJECT_ROOT"

echo "[$(date)] Final aggregation — merging all sources"

# Check patch2 completed
tail -3 "$PROJECT_ROOT/slurm/output_tf/whitefox_patch2_235839.out" 2>/dev/null || true

python3 -c "
from pathlib import Path
from collections import defaultdict

logging_root = Path('$LOGGING')
combined_summary = logging_root / 'run_summary_combined.log'
combined_coverage = logging_root / 'coverage_combined.log'

# Sources to merge (order matters: pre-patch files contain original data
# that was overwritten by patches; patches contain the gap-filling data).
summary_files = [
    logging_root / 'batch0' / 'run_summary_detailed.log',
    # batch1: original run had Defuser/DotDecomposer/DotMerger, but was
    # overwritten by patch1.  The Slurm output confirms:
    #   Defuser=200/0, DotDecomposer=1000/441, DotMerger=200/0
    # Patch1 wrote the remaining 14 opts into batch1's summary.
    logging_root / 'batch1' / 'run_summary_detailed.log',
    # batch2: original run had 5 opts, saved before patch2 overwrote.
    logging_root / 'batch2' / 'run_summary_pre_patch.log',
    # patch2 wrote remaining opts into batch2's summary.
    logging_root / 'batch2' / 'run_summary_detailed.log',
]

# Manually inject the 3 batch1 originals that were overwritten by patch1.
# Values confirmed from diagnostics output on Apr 26.
manual_overrides = {
    'Defuser':       {'created': 200, 'triggered': 0},
    'DotDecomposer': {'created': 1000, 'triggered': 441},
    'DotMerger':     {'created': 200, 'triggered': 0},
}

totals = defaultdict(lambda: defaultdict(int))
header_line = ''

for sf in summary_files:
    if not sf.exists():
        print(f'WARNING: {sf} not found, skipping')
        continue
    print(f'Reading: {sf}')
    for line in sf.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith(('=', '-', 'Optimization', 'WHITEFOX')):
            if line.startswith('Optimization'):
                header_line = line
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            continue
        opt_name = parts[0]
        if opt_name == 'TOTAL':
            continue
        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            continue
        # Only add if non-zero (avoid double-counting zeros)
        if values[0] > 0:
            for i, v in enumerate(values):
                totals[opt_name][i] += v

# Apply manual overrides for batch1 originals
for opt_name, vals in manual_overrides.items():
    if totals[opt_name].get(0, 0) == 0:
        # Not present from any file — inject
        totals[opt_name][0] = vals['created']
        totals[opt_name][1] = vals['triggered']
        print(f'Injected override: {opt_name} Created={vals[\"created\"]} Triggered={vals[\"triggered\"]}')
    else:
        # Already has data from patch1 — add the original counts
        totals[opt_name][0] += vals['created']
        totals[opt_name][1] += vals['triggered']
        print(f'Added override: {opt_name} +Created={vals[\"created\"]} +Triggered={vals[\"triggered\"]}')

# --- Write combined summary ---
if totals:
    with open(combined_summary, 'w') as f:
        sep = '=' * 95
        dash = '-' * 95
        f.write(f'{sep}\nWHITEFOX COMBINED RUN SUMMARY (all batches + patches)\n{sep}\n\n')
        if header_line:
            f.write(header_line + '\n')
        f.write(dash + '\n\n')
        grand = defaultdict(int)
        ncols = max(len(v) for v in totals.values())
        n_done = 0
        n_triggered = 0
        for opt_name in sorted(totals):
            vals = totals[opt_name]
            f.write(f'{opt_name:40s}')
            for i in range(ncols):
                v = vals.get(i, 0)
                grand[i] += v
                f.write(f' | {v:7d}')
            f.write('\n')
            if vals.get(0, 0) > 0:
                n_done += 1
            if vals.get(1, 0) > 0:
                n_triggered += 1
        f.write(dash + '\n')
        f.write(f'{\"TOTAL\":40s}')
        for i in range(len(grand)):
            f.write(f' | {grand[i]:7d}')
        f.write('\n' + sep + '\n')
        f.write(f'\nOptimizations completed: {n_done} / 49\n')
        f.write(f'Optimizations triggered: {n_triggered} / 49\n')
    print(f'\nCombined run summary: {combined_summary}')
else:
    print('ERROR: No data found!')

# --- Merge coverage ---
cov_lines = []
for batch_dir in sorted(logging_root.glob('batch*')):
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

# --- Quick summary ---
print()
print('=' * 60)
print('FINAL SUMMARY')
print('=' * 60)
print(f'  Optimizations completed:  {n_done} / 49')
print(f'  Optimizations triggered:  {n_triggered} / 49')
print(f'  Total tests created:      {grand.get(0, 0)}')
print(f'  Total tests triggered:    {grand.get(1, 0)}')
print()
for opt_name in sorted(totals):
    c = totals[opt_name].get(0, 0)
    t = totals[opt_name].get(1, 0)
    status = 'TRIGGERED' if t > 0 else 'early-stop' if c == 200 else 'no-trigger'
    print(f'  {opt_name:44s} {c:5d} created  {t:5d} triggered  [{status}]')
"

echo ""
echo "[$(date)] Done. Results:"
echo "  Summary: $LOGGING/run_summary_combined.log"
echo "  Coverage: $LOGGING/coverage_combined.log"
