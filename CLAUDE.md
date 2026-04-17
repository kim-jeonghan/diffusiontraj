# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv sync --extra dev
```

Requires Python 3.11 (locked in `.python-version`). Uses `uv` as the package manager.

## Commands

**Training:**
```bash
WANDB_MODE=disabled uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu

WANDB_MODE=disabled uv run python scripts/train_trajectory_stitching.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu
```

**Planning** (requires an existing training run):
```bash
uv run python scripts/plan_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu --plan_n_ep 1

uv run python scripts/plan_trajectory_stitching.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_config.py \
  --device cpu --plan_n_ep 1
```
`--plan_n_ep -100` means "all episodes". Planning scripts auto-resolve `diffusion_epoch=latest` from the matching training run directory.

**Smoke test** (verifies forward pass only):
```bash
DIFFUSER_DEVICE=cpu WANDB_MODE=disabled \
uv run python scripts/train_maze_baseline.py \
  --config configs/experiment/maze2d/maze2d_umaze_baseline_smoke_config.py \
  --device cpu --n_train_steps 0
```
Expected final output: `Testing forward... ✓`

**Pytest:**
```bash
uv run pytest -q tests/test_maze2d_smoke.py tests/test_planning_config_schema.py
uv run pytest tests/   # all tests
```

**Lint/format:**
```bash
uv run ruff check --fix src/ tests/ scripts/
uv run black src/ tests/ scripts/
```
Pre-commit hooks enforce ruff, black, mypy, and YAML/TOML validation automatically.

## Architecture

This is a research package for diffusion-based planning in Maze2D environments. Two main approaches:
1. **Baseline diffusion planning** — standard diffusion model generates full trajectories
2. **Trajectory stitching** — compositional approach blending overlapping trajectory segments

### Data Flow

**Training:**
```
Config file (Python) → ArgsParser (config + CLI overrides)
  → Dataset (Maze2D HDF5 or D4RL) → DataLoader
  → Diffusion model (MazeTemporalUNet) forward/loss
  → Optimizer → Checkpoints saved under artifacts/runs/<dataset>/<exp_name>/
```

**Planning:**
```
Load checkpoint → Policy (MazePolicy or TrajectoryStitchingPolicy)
  → Environment loop: state → policy.plan() → sampled trajectory → env.step()
  → Rollout saved as artifacts/runs/.../plans/release/<YYMMDD-HHMMSS>/00_rollout.json
```

### Module Responsibilities

- **`models/diffusion/`** — `MazeGaussianDiffusion` wraps `MazeTemporalUNet`; implements forward (noise) and reverse (denoise) diffusion with cosine beta schedule and inverse dynamics for action prediction.
- **`models/trajectory_stitching/`** — Extends baseline with overlap region blending; `StitchingDiffusion` and its trainer handle segment composition.
- **`trainers/maze_trainer.py`** — `MazeTrainer`: training loop, EMA updates, checkpoint saving, sample rendering.
- **`planners/`** — `MazePolicy` and `Maze2DEnvPlanner` handle inference-time rollouts; `trajectory_blender.py` computes blended segments for stitching.
- **`datasets/`** — D4RL/HDF5 loading, normalization, sequence dataset construction. Compositional variants under `datasets/comp/`.
- **`utils/serialization.py`** — `Config`-based pickle serialization for reconstructing datasets, models, and trainers from checkpoints.
- **`rendering/`** — `MazeRenderer` visualizes trajectories during training and planning.
- **`guides/`** — Inverse dynamics policies and compositional planning guides.

### Config System

Each experiment is a single Python file in `configs/experiment/maze2d/` that exports a `base` dict with `diffusion` and `plan` blocks. CLI flags override config values. Key rules:
- `exp_name` is built with `watch(...)` for deterministic, human-readable directory names.
- `dataset`, `dset_h5path`, horizon, and `n_diffusion_steps` must be aligned across both blocks.
- Knobs belong in `dataset_config`, `network_config`, or `diff_config` sub-dicts.
- Add a new config file rather than branching one file with conditionals.
- `logbase = artifacts/runs`, `prefix = diffusion/` for training, `plans/release` for planning.

Config variants: `*_smoke_config.py` (tiny/fast), `*_umaze_*`, `*_medium_*`, `*_large_*`.

### Artifacts Layout

```
artifacts/runs/<dataset>/<exp_name>/
  args.json                # runtime arguments after config + CLI merge
  *_config.pkl             # serialized constructor configs (dataset, model, trainer, render)
  model_config.txt         # human-readable model spec
  diffusion_model.pkl      # model weights
  ckpt_N.pt                # training checkpoints
  plans/release/<timestamp>/
    00_rollout.json        # per-episode metrics
```

Local data lives in `data/m2d/` (bundled Maze2D HDF5 for tests) and `data/eval_problems/` (planning start/goal sets).
