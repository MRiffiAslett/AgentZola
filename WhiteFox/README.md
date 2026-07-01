# WhiteFox

## Dependencies

Poetry manages the virtualenv. The SLURM script runs `poetry install` automatically on each job.  
To install manually on a compute node:

```bash
cd /vol/bitbucket/mtr25/AgentZola/WhiteFox
poetry lock --check || poetry lock
poetry install
```

The TF wheel is not on PyPI — it must exist locally before `poetry install` runs.

## TF wheels

| Key | Path |
|-----|------|
| `20250806` | `/vol/bitbucket/mtr25/tfbuild/wheels/tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806-cp312-cp312-linux_x86_64.whl` |
| `20230507` | `/vol/bitbucket/mtr25/tfbuild/wheels/tensorflow_cpu-2.14.0+selfbuilt.20230507-cp310-cp310-linux_x86_64.whl` |

## LLVM (coverage)

`llvm-profdata` and `llvm-cov` must be at `/vol/bitbucket/mtr25/tfbuild/llvm17/bin/`.  
The SLURM script sets `WHITEFOX_LLVM_DIR` to that path automatically if it exists.

## HuggingFace cache

Not set globally. The TOML config has `hf_home = "hf_cache"` which resolves relative to the project root.  
To avoid re-downloading weights on every node, pin it to a shared volume:

```bash
export HF_HOME=/vol/bitbucket/mtr25/hf_cache
```

or update `hf_home` in `xilo_xla/config/generator.toml`.

## Key environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WHITEFOX_MODEL` | — | HuggingFace model ID |
| `WHITEFOX_WHEEL_VERSION` | — | Selects TF wheel (`20250806` / `20230507`) |
| `WHITEFOX_PROMPTS_VERSION` | — | Selects prompts artifact |
| `WHITEFOX_LOGGING_DIR` | `logging/<batch>` | Override log output path |
| `WHITEFOX_EARLY_STOP_ITERS` | `20` | Stop opt early if no triggers after N iters |
| `WHITEFOX_TEST_MEM_LIMIT_GB` | `8` | Per-test subprocess memory cap |
| `WHITEFOX_PARALLEL_TEST_WORKERS` | from TOML | Override worker count |
