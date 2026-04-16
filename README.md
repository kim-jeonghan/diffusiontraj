# Comp Diffuser

Maze2D-focused diffusion planning and trajectory stitching code, organized as a `uv` project.

## Layout

- `src/comp_diffuser/`: library code
- `scripts/`: runnable train and planning entrypoints
- `configs/experiment/maze2d/`: Maze2D experiment configs
- `data/smoke/`: small local smoke datasets
- `data/eval_problems/`: local evaluation problem sets
- `artifacts/runs/`: generated run artifacts

## Setup

Create or reuse a Python 3.11 environment, then install the project with `uv`:

```bash
uv sync --extra dev
```

If you want to use an existing environment, `uv` can still drive commands through it.

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

## Test

```bash
uv run pytest -q tests/test_maze2d_smoke.py
```

## Data

Included smoke datasets:

- `data/smoke/maze2d-umaze-sparse-v1-smoke.hdf5`
- `data/smoke/maze2d-medium-sparse-v1-smoke.hdf5`
- `data/smoke/maze2d-large-sparse-v1-smoke.hdf5`
