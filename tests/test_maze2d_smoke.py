import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_maze2d_ben_dataset_loads():
    d4rl = importlib.import_module("comp_diffuser.datasets.d4rl")

    env = d4rl.load_env_gym_robo("maze2d-umaze-v1")
    env.dset_h5path = str(ROOT / "data/m2d/maze2d-umaze-sparse-v1-smoke.hdf5")

    dataset = d4rl.get_dataset(env)

    assert dataset["observations"].ndim == 2
    assert dataset["actions"].ndim == 2
    assert dataset["observations"].shape[0] == dataset["actions"].shape[0]
    assert "terminals" in dataset
