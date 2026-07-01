# tfxla SLURM job

## Submitting

Must be run from inside `WhiteFox/`:

```bash
cd /vol/bitbucket/mtr25/AgentZola/WhiteFox
sbatch slurm/tfxla/tfxla_a100_array.sh
```

Output files: `slurm/tfxla/out/whitefox_<JOBID>_<TASKID>.out`

## Configuring a run

Edit the three lines at the top of `tfxla_a100_array.sh`:

```bash
WHITEFOX_MODEL="bigcode/starcoder"   # or Qwen/Qwen2.5-Coder-14B
WHITEFOX_WHEEL_VERSION="20250806"    # or 20230507
WHITEFOX_PROMPTS_VERSION="20250806"
```

Model-specific vLLM params (dtype, context length, stop tokens) are picked up automatically from the registry in `generation/generator.py` — nothing else needs editing.

## Array layout

`--array=0-2` runs 3 tasks in parallel, each assigned ~17 optimisations.  
If you change the number of tasks, update `N_TASKS` and the opt-list slicing logic in the script accordingly.

## Outputs

| Path | Contents |
|------|---------|
| `logging/<batch>/run_summary_detailed.log` | Per-batch run stats |
| `logging/<batch>/run_stats.json` | Machine-readable sidecar |
| `logging/run_summary_combined.log` | Aggregated summary across batches |
| `logging/coverage_combined.log` | Union coverage across batches |
| `logging/<batch>/cgroup_mem.tsv` | cgroup memory trace |
| `logging/<batch>/oom_postmortem.log` | Written on OOM exit |
