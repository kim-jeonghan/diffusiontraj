# Comp Diffuser

Maze2D-focused diffusion planning and trajectory stitching code, organized as a `uv` project.

## Setup

Create or reuse a Python 3.11 environment, then install the project with `uv`:

```bash
uv sync --extra dev
```

This installs the package, test dependencies, and lint tooling into the managed `uv` environment.

## Package Layout

Core repository structure:

- `src/comp_diffuser/`: library code
- `src/comp_diffuser/datasets/`: dataset loading, preprocessing, normalization
- `src/comp_diffuser/models/`: diffusion, stitching, and helper modules
- `src/comp_diffuser/planners/`: planning policies and trajectory blending helpers
- `src/comp_diffuser/trainers/`: training loops
- `src/comp_diffuser/rendering/`: Maze2D rendering utilities
- `src/comp_diffuser/utils/`: shared config, serialization, logging, and array helpers
- `scripts/`: runnable train and planning entrypoints
- `configs/experiment/maze2d/`: Maze2D experiment configs
- `tests/`: smoke and regression tests
- `data/`: local datasets and planning problem files
- `artifacts/runs/`: generated checkpoints, configs, and evaluation outputs

## Smoke Run

```bash
DIFFUSER_DEVICE=cpu \
WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_smoke_config.py \
  --device cpu \
  --n_train_steps 0
```

Expected end of output:

```text
Testing forward... ✓
```

## Train

Maze2D umaze baseline example:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu
```

Trajectory stitching training example:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_trajectory_stitching.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu
```

Training writes run outputs under `artifacts/runs/...`, including saved config snapshots and model artifacts through `args.savepath`.

## Plan

Planning uses a training config plus the latest saved checkpoint in the corresponding run directory.

Maze baseline planning example:

```bash
uv run python scripts/plan_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu \
  --plan_n_ep 1
```

Trajectory stitching planning example:

```bash
uv run python scripts/plan_trajectory_stitching.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu \
  --plan_n_ep 1
```

Planning outputs are also written beneath `artifacts/runs/...` using the experiment save path plus a timestamped planning subdirectory.

## Test

```bash
uv run pytest -q tests/test_maze2d_smoke.py
```

## Data and Artifacts

Included smoke datasets:

- `data/smoke/maze2d-umaze-sparse-v1-smoke.hdf5`
- `data/smoke/maze2d-medium-sparse-v1-smoke.hdf5`
- `data/smoke/maze2d-large-sparse-v1-smoke.hdf5`

Other local paths used by the repository:

- `data/smoke/`: small local datasets for quick validation
- `data/eval_problems/`: saved planning start/goal problem files
- `scripts/eval_problems/`: helpers for generating Maze2D evaluation problem sets
- `artifacts/runs/`: checkpoints, saved config pickles, renders, and planning outputs
