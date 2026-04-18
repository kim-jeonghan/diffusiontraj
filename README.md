# Comp Diffuser

Maze2D-focused diffusion training and planning code, packaged as a `uv` project.

## Setup

Use Python 3.11 and install dependencies with:

```bash
uv sync --extra dev
```

## Repository Layout

- `src/comp_diffuser/`: library code
- `scripts/`: train and plan entrypoints
- `configs/maze2d/`: Maze2D configs
- `tests/`: regression tests
- `data/m2d/`: local Maze2D HDF5 datasets
- `data/eval_problems/`: planning problem sets
- `artifacts/`: checkpoints, renders, and planning outputs

## Training

Baseline training:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/maze2d/maze2d_umaze_baseline_config.py \
  --device cuda
```

Trajectory stitching training:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_trajectory_stitching.py \
  --config configs/maze2d/maze2d_umaze_trajectory_stitching_config.py \
  --device cuda
```

## Planning

Planning requires a trained checkpoint, not just `state_0.pt`.

Baseline planning:

```bash
uv run python scripts/plan_maze_baseline.py \
  --config configs/maze2d/maze2d_umaze_baseline_config.py \
  --device cuda \
  --plan_n_ep 1
```

Trajectory stitching planning:

```bash
uv run python scripts/plan_trajectory_stitching.py \
  --config configs/maze2d/maze2d_umaze_trajectory_stitching_config.py \
  --device cuda \
  --plan_n_ep 1
```

If planning exits with “No trained checkpoint found”, run training until a non-zero checkpoint such as `state_4000.pt` exists.

## Test and Lint

```bash
uv run pytest -q
uv run pytest -q tests/test_planning_config_schema.py
uvx ruff check .
uvx ruff check . --fix
pre-commit run --all-files
```

## Config Notes

Each file in `configs/maze2d/` exports a `base` dictionary with `dataset`, `dset_h5path`, `diffusion`, and `plan` blocks.

Available configs:

- `maze2d_umaze_baseline_config.py`
- `maze2d_medium_baseline_config.py`
- `maze2d_large_baseline_config.py`
- `maze2d_umaze_trajectory_stitching_config.py`
- `maze2d_medium_trajectory_stitching_config.py`
- `maze2d_large_trajectory_stitching_config.py`

Do not mix baseline configs with stitching scripts, or stitching configs with baseline scripts.

## Outputs

Training and planning write under `artifacts/<dataset>/<exp_name>/...`.

Typical training files:

- `args.json`
- `dataset_config.pkl`
- `render_config.pkl`
- `model_config.pkl`
- `diffusion_model.pkl`
- `trainer_config.pkl`
- `state_*.pt`
- sampled `.png` renders

Planning outputs include `00_rollout.json`, predicted trajectory renders, and rollout renders.
