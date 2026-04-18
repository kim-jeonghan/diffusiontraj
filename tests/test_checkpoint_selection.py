import torch

from comp_diffuser.utils.serialization import (
    get_checkpoint_step,
    has_trained_checkpoint,
)


def test_state_zero_checkpoint_counts_as_trained_when_step_is_positive(tmp_path):
    torch.save({"step": 4000}, tmp_path / "state_0.pt")

    assert get_checkpoint_step((str(tmp_path),), 0) == 4000
    assert has_trained_checkpoint((str(tmp_path),))


def test_state_zero_checkpoint_is_untrained_when_step_is_zero(tmp_path):
    torch.save({"step": 0}, tmp_path / "state_0.pt")

    assert get_checkpoint_step((str(tmp_path),), 0) == 0
    assert not has_trained_checkpoint((str(tmp_path),))
