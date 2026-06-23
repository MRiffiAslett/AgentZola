# Reproducibility Notes — TF/XLA Fuzzer

This document captures external artifacts required to reproduce the WhiteFox
TF/XLA experiment outside Imperial's `/vol/bitbucket` filesystem (e.g. when
running the Docker workflow in `tfxla_a100_docker.sh`).

## Custom TensorFlow wheel (REQUIRED)

The fuzzer depends on a self-built TensorFlow wheel that adds LLVM source-based
coverage instrumentation (`-fprofile-instr-generate`) to the TF/XLA runtime.
This is what produces the `.profraw` files merged by `llvm-profdata` during a
fuzzing run. **Stock upstream `tensorflow` from PyPI will not work** — the
fuzzer will run, but coverage will always be unavailable.

- File name: `tensorflow_cpu-2.20.0.dev0+selfbuilt.20250806-cp312-cp312-linux_x86_64.whl`
- Python: 3.12 (cp312)
- Platform: Linux x86_64
- Approx. size: ~500–700 MB (TBC)
- Contents: TensorFlow 2.20.0.dev0 built with Clang/LLVM 17, source-based
  coverage instrumentation enabled, CPU build.

### How reproducers obtain the wheel

The wheel is **not** automatically downloaded by the Docker script.
Reproducers must request it from the project authors and place it at a
known absolute path on the host before running the script.

> **TODO (author):** decide on a distribution channel (e.g. GitHub Release
> on AgentZola, Zenodo with a DOI, signed-link on a faculty server) and
> document the exact URL / contact procedure here. Until this is filled
> in, reproducers cannot run the experiment from a clean checkout.

Once obtained, point the run script at the wheel:

```bash
export WHITEFOX_TF_WHEEL_PATH="/absolute/path/to/the.whl"
```

A SHA-256 checksum will be published alongside the wheel; reproducers can
optionally verify it by also exporting:

```bash
export WHITEFOX_TF_WHEEL_SHA256="<sha256>"
```

## LLVM 17 binaries (`llvm-profdata`, `llvm-cov`)

Fetched at image build time from the public LLVM project release page:

```
https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz
```

No action required from reproducers; the Dockerfile streams + extracts only
`llvm-profdata` and `llvm-cov` (see `fetch_llvm17.sh`).

## Hugging Face StarCoder weights

The fuzzer drives `bigcode/starcoder` via vLLM. This is a gated model, so
reproducers must:

1. Accept the model licence at <https://huggingface.co/bigcode/starcoder>.
2. Provide an HF access token at run time, either:
   - `export HF_TOKEN=hf_xxx` before invoking the run script, **or**
   - place the token in `~/.hf_token` (the script will pick it up).

## GPU requirements

vLLM serving StarCoder requires **a single** CUDA-capable GPU with ≥ ~24 GB
VRAM (StarCoder is 15B parameters in fp16; smaller GPUs OOM at load time).
Tested on:

- NVIDIA A100 (40 GB / 80 GB)
- NVIDIA H100 (80 GB)

Smaller GPUs are not currently supported without changing
`xilo_xla/config/generator.toml::[model]` to a quantised or smaller code-LLM.

If no GPU is detected at run time the Docker script prints a warning and
continues; vLLM will then fail at model load. This is intentional so that
image builds and dry runs work on CPU-only machines.

> **Note on the SLURM script's `--gres=gpu:3` and `--array=0-2`.** The
> production SLURM job (`tfxla_a100_array.sh`) requests 3 GPUs and runs 3
> array tasks. **This is purely to get more CPU RAM** (3 × A100 nodes ≈
> 570 GB system RAM total) — TF/XLA test subprocesses, profraw scratch
> and coverage merging are CPU/RAM-bound, not GPU-bound. The fuzzer
> itself only uses one GPU per task for vLLM. Reproducers therefore
> need **one** suitable GPU, not three. The 50-optimization workload is
> only sharded across 3 SLURM tasks because a single 190 GB node could
> not hold all of it within the 72 h wall-time without OOM.
>
> The Docker workflow runs all 50 optimizations sequentially in a
> single container; on a machine with sufficient RAM (~64 GB+ recommended)
> this works fine, just slower.

## SLURM-only artefacts (NOT needed for Docker reproducers)

The following only matter for the Imperial cluster execution path
(`tfxla_a100_array.sh`) and are intentionally absent from the Docker workflow:

- `/vol/cuda/12.0.0/setup.sh` — CUDA module file. The Docker image bundles
  CUDA via the `nvidia/cuda` base layer instead.
- `/data/whitefox_cov_*` — node-local profraw merge pool. The Docker image
  uses a tmpfs / volume mounted at `/whitefox-data` instead.
- `/vol/bitbucket/mtr25/tfbuild/llvm17/bin` — pre-staged LLVM 17. Docker
  fetches it at build time as described above.

## Running the experiment via Docker

The script `tfxla_a100_docker.sh` (in this folder) is a one-shot
reproducer. It builds a self-contained image and runs the fuzzer:

```bash
export WHITEFOX_TF_WHEEL_PATH=/abs/path/to/tensorflow_cpu-...whl  # required
export WHITEFOX_TF_WHEEL_SHA256=<sha256>          # optional but recommended
export HF_TOKEN=hf_xxx                            # or place in ~/.hf_token
./tfxla_a100_docker.sh                            # all 50 optimizations
./tfxla_a100_docker.sh --only-opt HloDce,HloCse   # subset
```

Knobs:

| Variable                    | Default                          | Purpose                                       |
|-----------------------------|----------------------------------|-----------------------------------------------|
| `WHITEFOX_TF_WHEEL_PATH`    | (required)                       | Absolute path to the custom TF wheel on the host |
| `WHITEFOX_TF_WHEEL_SHA256`  | unset                            | Optional sha256 to verify the local wheel     |
| `WHITEFOX_DOCKER_IMAGE`     | `whitefox-tfxla:latest`          | Tag of the built image                        |
| `WHITEFOX_RESULTS_DIR`      | `$PWD/whitefox-results`          | Host directory for `logging/`, `output/`, `hf_cache/` |
| `WHITEFOX_DOCKER_NO_BUILD`  | `0`                              | If `1`, reuse existing image, skip rebuild    |
| `WHITEFOX_DOCKER_SHELL`     | `0`                              | If `1`, drop into bash inside the container   |
| `HF_TOKEN`                  | from `~/.hf_token` if unset      | Hugging Face access token (gated StarCoder)   |

## Running the experiment via Apptainer (Imperial GPU cluster)

The cluster compute nodes do not run a Docker daemon (they use Apptainer /
Singularity instead). `tfxla_a100_apptainer.sh` + `build_apptainer.sh`
provide the cluster-native container path and mirror `tfxla_a100_docker.sh`
exactly — same base image, same Apptainer `.def` file built from the same
Dockerfile logic.

### Prerequisites

Check that Apptainer is available and that `--fakeroot` is configured for
your account (required to build the image without root):

```bash
apptainer --version
apptainer build --fakeroot /dev/null /dev/null 2>&1 | head -3
```

If the second command prints `fakeroot: not configured`, ask DoC CSG to
enable fakeroot for your username, **or** use sandbox mode as a fallback
(see below).

### Build the image (once, from the login node)

```bash
# Wheel is already at the default path on the Imperial cluster:
bash WhiteFox/slurm/tfxla/build_apptainer.sh

# Produces: /vol/bitbucket/mtr25/whitefox-tfxla.sif (~3–4 GB)
# Takes ~10–20 min (downloads LLVM 17 + installs deps)
```

If `--fakeroot` is not available, use sandbox mode instead:

```bash
WHITEFOX_APPTAINER_SANDBOX=1 bash WhiteFox/slurm/tfxla/build_apptainer.sh
# Produces: /vol/bitbucket/mtr25/whitefox-tfxla_sandbox/  (a directory)
```

### Submit a run

```bash
# Quick single-opt test:
sbatch WhiteFox/slurm/tfxla/tfxla_a100_apptainer.sh --only-opt HloDce

# Full run (all 49 optimizations, sequential — slower than the 3-node array):
sbatch WhiteFox/slurm/tfxla/tfxla_a100_apptainer.sh
```

Output lands in `WhiteFox/slurm/tfxla/out/whitefox-apptainer_<jobid>.out`.

| Variable                    | Default                                                | Purpose                                              |
|-----------------------------|--------------------------------------------------------|------------------------------------------------------|
| `WHITEFOX_SIF_PATH`         | `/vol/bitbucket/mtr25/whitefox-tfxla.sif`              | Path to the built `.sif` or sandbox directory        |
| `WHITEFOX_RESULTS_DIR`      | `/vol/bitbucket/mtr25/AgentZola/WhiteFox`              | Host directory for `logging/`, `output/`, `hf_cache/`|
| `WHITEFOX_EARLY_STOP_ITERS` | `20`                                                   | 0 to disable early-stop                              |
| `HF_TOKEN`                  | from `~/.hf_token` if unset                            | Hugging Face access token (gated StarCoder)          |

## Result artefacts

A successful run writes to (host paths chosen by the reproducer at run time):

- `<host>/logging/<batch>/run_summary_detailed.log` — per-batch results.
- `<host>/logging/run_summary_combined.log` — aggregated across batches.
- `<host>/logging/coverage_combined.log` — aggregated coverage.
- `<host>/output/` — generated test cases that triggered bugs.
