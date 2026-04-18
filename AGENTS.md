# Repository Guidelines

## Project Structure & Module Organization

This repository is a `uv`-managed Python project focused on Maze2D diffusion training and planning.

- `src/comp_diffuser/`: core library code
- `src/comp_diffuser/models/`: diffusion, common blocks, and networks
- `src/comp_diffuser/trainers/`: training loops
- `src/comp_diffuser/planners/`: planning and policy logic
- `configs/maze2d/`: experiment configs
- `scripts/`: runnable entrypoints such as `train_maze_baseline.py`
- `tests/`: regression tests
- `data/m2d/`: local Maze2D HDF5 datasets
- `data/eval_problems/`: saved planning problem sets
- `artifacts/`: generated checkpoints, config snapshots, and renders

## Build, Test, and Development Commands

- `uv sync --extra dev`: install runtime and dev dependencies
- `uv run pytest -q`: run the full test suite
- `uvx ruff check .`: run lint checks
- `uvx ruff check . --fix`: apply safe lint fixes
- `uv run python scripts/train_maze_baseline.py --config configs/maze2d/maze2d_umaze_baseline_config.py --device cpu`: run baseline training

## Coding Style & Naming Conventions

Target Python 3.11. Use 4-space indentation and keep lines within Ruff’s configured limit of 100 characters. Prefer explicit imports over star imports. Use `snake_case` for files, functions, variables, and module names; use `PascalCase` for classes. Keep new modules aligned with the current layout, for example `models/common/`, `models/diffusion/`, and `models/networks/`.

## Testing Guidelines

Tests use `pytest` and live under `tests/` with names like `test_*.py`. Add focused regression tests for any change that affects config loading, dataset parsing, model wiring, or planning behavior.

## Commit & Pull Request Guidelines

Follow the repository’s recent commit style: the subject line should explain why the change exists, not just what changed. Prefer small, reviewable commits. Include validation notes in the commit body when relevant, for example `Tested: uv run pytest -q`.

For pull requests, include:

- a short problem statement
- the key files changed
- commands used for verification
- sample outputs or screenshots when training, planning, or rendering behavior changes

## Configuration Tips

Keep `dataset`, `dset_h5path`, horizon values, and `n_diffusion_steps` aligned across the `diffusion` and `plan` blocks in each config. Use relative paths such as `data/m2d/...` and `artifacts/...` to keep runs portable.

## Logic Refactors

When changing planning, training, or config logic:

- check the original repository behavior before changing contracts or control flow
- keep baseline and trajectory stitching as separate lanes; do not reuse one lane’s config file in the other
- prefer explicit early failure over silent fallback when model, trainer, planner, or artifact paths do not match
- verify both code paths and artifacts: config import, checkpoint path, planner inputs, and at least one real command relevant to the changed lane
