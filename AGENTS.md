## Python Environment

- **Python Version:** 3.11+
- **Package Manager:** Use `uv` for dependency management.
- **Virtual Environment:** `prl` (created via `uv venv`).

## Core Commands

### Environment Setup

```sh
# Create virtual environment and install dependencies
./install.sh
```

## Experimenta

Experiments are launched with Slurm commands like:

```sh
timestamp=$(date +'%Y%m%d-%H%M%S'); sbatch --mail-type=ALL --mail-user=lewis+hfc@huggingface.co --job-name=imo-qwen3-4b-thinking_rc_v13.00 --nodes=14 run_hf.slurm --config rc_proof_qwen3-4b-thinking_v13.00 --job-name "rc_proof_qwen3-4b-thinking_v13.00-${timestamp}"
```

The Slurm logs can then be accessed via the `slog {slurm_job_id}` alias, e.g.

```sh
slog 21958662
```

The vLLM and training logs are stored in the `runs/` folder.

### Restarting a run from a checkpoint

To restart runs from a checkpoint, first clean up the existing run with a command like:

```sh
./scratch/cleanup_run_dir.sh runs/rc_proof_qwen3-4b-thinking_v13.01-{timesatamp}
```

where `timestamp` corresponds to the latest corresponding run in the `runs` folder. Then run the launch command pointing to the run directory in the `--job-name`:

```sh
sbatch --mail-type=ALL --mail-user=lewis+hfc@huggingface.co --job-name=imo-qwen3-4b-thinking_rc_v13.00 --nodes=14 run_hf.slurm --config rc_proof_qwen3-4b-thinking_v13.00 --job-name rc_proof_qwen3-4b-thinking_v13.00-{timestamp}
```

Notes:

- When asked to relaunch a job, do not create a fresh folder in runs. Always re-use the latest one corresponding to the config (e.g. v13.0) that has been requested.