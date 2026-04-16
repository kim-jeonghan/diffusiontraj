import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_maze2d_ben_dataset_loads(monkeypatch):
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "hi_diffuser_ben")

    d4rl = importlib.import_module("comp_diffuser.datasets.d4rl")

    env = d4rl.load_env_gym_robo("maze2d-umaze-v1")
    env.dset_h5path = str(ROOT / "data/smoke/maze2d-umaze-sparse-v1-smoke.hdf5")

    dataset = d4rl.get_dataset(env)

    assert dataset["observations"].ndim == 2
    assert dataset["actions"].ndim == 2
    assert dataset["observations"].shape[0] == dataset["actions"].shape[0]
    assert "terminals" in dataset
