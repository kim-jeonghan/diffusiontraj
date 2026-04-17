# Comp Diffuser

Maze2D-focused diffusion planning code, packaged as a `uv` project.

## Setup

Use Python 3.11 and install dependencies with:

```bash
uv sync --extra dev
```

## Repository Layout

- `src/comp_diffuser/`: library code
- `scripts/`: train and plan entrypoints
- `configs/experiment/maze2d/`: Maze2D experiment configs
- `tests/`: smoke and regression tests
- `data/m2d/`: local Maze2D HDF5 datasets
- `data/eval_problems/`: planning problem sets
- `artifacts/runs/`: checkpoints, config snapshots, renders, and plan outputs

## Train

Baseline training:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu
```

Smoke run:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_smoke_config.py \
  --device cpu \
  --n_train_steps 0
```

Expected smoke output ends with `Testing forward... ✓`.

## Plan

Planning expects an existing baseline run:

```bash
uv run python scripts/plan_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu \
  --plan_n_ep 1
```

`--plan_n_ep 1` is a minimal sanity run. Planning resolves `diffusion_epoch=latest` from the matching training directory under `artifacts/runs/`.

## Test and Lint

```bash
uv run pytest -q
uv run pytest -q tests/test_maze2d_smoke.py tests/test_planning_config_schema.py
uvx ruff check .
uvx ruff check . --fix
```

## Config Notes

Each file in `configs/experiment/maze2d/` exports a `base` dictionary with `dataset`, `dset_h5path`, `diffusion`, and `plan` blocks. Keep `dataset`, `dset_h5path`, horizon values, and `n_diffusion_steps` aligned across training and planning.

Current configs:

- `maze2d_umaze_baseline_smoke_config.py`
- `maze2d_umaze_baseline_config.py`
- `maze2d_medium_baseline_config.py`
- `maze2d_large_baseline_config.py`

## Outputs

Training and planning write to `artifacts/runs/<dataset>/<exp_name>/...`.

Common generated files:

- `args.json`: parsed runtime arguments
- `*_config.pkl`: serialized constructor configs
- `model_config.txt`: readable model summary
- checkpoints, sampled renders, and rollout summaries

## Current Limitation

Baseline training and planning are the validated paths. Trajectory stitching code is still being reorganized and should not be mixed with baseline configs until a stitching-specific config and trainer/model pairing are restored.
