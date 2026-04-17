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
- `scripts/`: runnable train and planning entrypoints
- `configs/experiment/maze2d/`: Maze2D experiment configs
- `tests/`: smoke and regression tests
- `data/`: local datasets and planning problem files
- `artifacts/runs/`: generated checkpoints, config snapshots, and planning outputs

## Train

Baseline diffusion training:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu
```

Trajectory stitching training:

```bash
WANDB_MODE=disabled \
uv run python scripts/train_trajectory_stitching.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu
```

Useful config variants:

- `configs/experiment/maze2d/maze2d_umaze_baseline_smoke_config.py`
- `configs/experiment/maze2d/maze2d_umaze_baseline_config.py`
- `configs/experiment/maze2d/maze2d_medium_baseline_config.py`
- `configs/experiment/maze2d/maze2d_large_baseline_config.py`

`--config` is the main experiment selector. CLI flags such as `--device cpu` override the values loaded from the config file.

## Plan

Planning expects an existing training run and loads the latest checkpoint from the matching diffusion run directory.

Maze baseline planning:

```bash
uv run python scripts/plan_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu \
  --plan_n_ep 1
```

Trajectory stitching planning:

```bash
uv run python scripts/plan_trajectory_stitching.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu \
  --plan_n_ep 1
```

Notes:

- `--plan_n_ep 1` is a minimal sanity run. `-100` means "all episodes" in the current planning scripts.
- The planning scripts resolve `diffusion_epoch=latest` by scanning the corresponding training run directory.
- The planning save directory is created under the plan config save path and then extended with a timestamped subdirectory such as `YYMMDD-HHMMSS-mmm-nm1-phzn160-...`.

## Smoke Test

Minimal training smoke test:

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

Repo smoke/regression tests:

```bash
uv run pytest -q tests/test_maze2d_smoke.py tests/test_planning_config_schema.py
```

What these cover:

- `tests/test_maze2d_smoke.py`: local smoke dataset loading
- `tests/test_planning_config_schema.py`: readable planning/policy config key compatibility

## Artifacts

Training and planning both write under `artifacts/runs/<dataset>/<exp_name>/...`.

Typical training directory:

```text
artifacts/runs/maze2d-umaze-v1/
  diffusion/maze2d_umaze_baseline_smoke_config_T512/
    args.json
    dataset_config.pkl
    render_config.pkl
    model_config.pkl
    model_config.txt
    diffusion_model.pkl
    trainer_config.pkl
    ...
```

Important output conventions:

- `args.json`: parsed runtime arguments after config loading and CLI overrides
- `*_config.pkl`: serialized constructor configs used to recreate datasets, models, and trainers
- `model_config.txt`: text dump for quick inspection
- trainer outputs: checkpoints, sampled renders, and other training-time outputs written into the same run folder
- planning outputs: timestamped subdirectories containing rollout summaries such as `00_rollout.json`, rendered media, and run summary JSON files

Local data paths used by the repo:

- `data/smoke/`: bundled small HDF5 datasets for quick validation
- `data/eval_problems/`: saved planning start/goal problem files
- `scripts/eval_problems/`: helpers for generating Maze2D evaluation problem sets

## Config Authoring Rules

Use the existing Maze2D configs in `configs/experiment/maze2d/` as the template. Each config file exports a top-level `base` dictionary with at least:

- `dataset`: Maze2D environment name
- `dset_h5path`: local HDF5 dataset path
- `diffusion`: training-time settings
- `plan`: planning-time settings

Rules to follow:

- Keep both `diffusion` and `plan` blocks in the same file so train and plan share the same experiment identity.
- Set `logbase` to `artifacts/runs` unless there is a strong reason to move outputs.
- Keep `prefix` stable by task type: training uses `diffusion/`, planning uses `plans/release`.
- Build `exp_name` with `watch(...)` so directory names stay deterministic and human-readable.
- Keep `dataset`, `dset_h5path`, horizon values, and `n_diffusion_steps` aligned across `diffusion` and `plan`.
- Use relative local paths for datasets and artifacts so the repo remains portable.
- Put dataset-specific knobs inside `dataset_config`, model-specific knobs inside `network_config`, and diffusion-specific knobs inside `diff_config`.
- Prefer adding a new config file over heavily branching one config with ad hoc conditionals.

Current Maze2D config patterns:

- `maze2d_umaze_baseline_smoke_config.py`: smallest local smoke run
- `maze2d_umaze_baseline_config.py`: umaze baseline
- `maze2d_medium_baseline_config.py`: medium horizon and medium smoke dataset
- `maze2d_large_baseline_config.py`: large horizon and large smoke dataset
