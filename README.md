# WhiteFox Reproduction Study

## Paths on cluster

```
/vol/bitbucket/<user>/
├── AgentZola/              # this repo
├── tfbuild/
│   ├── wheels/             # pre-built TF CPU wheels
│   └── llvm17/bin/         # llvm-profdata, llvm-cov
└── .pyenv/                 # Python managed via pyenv
```

The SLURM scripts hardcode `/vol/bitbucket/<user>/...` — update `<user>` for your environment.

## Python

pyenv at `/vol/bitbucket/<user>/.pyenv`, active version **3.12.3**.  
`.bashrc` must contain:

```bash
export PYENV_ROOT="/vol/bitbucket/<user>/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Poetry is installed in `~/.local/bin/` and is on PATH on compute nodes after pyenv init.  
On login nodes it may not be in PATH — run jobs via SLURM, not manually from the login node.

## Cloning

```bash
git clone <artifact-repo-url> /vol/bitbucket/<user>/AgentZola
```
