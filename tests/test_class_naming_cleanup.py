import torch

from comp_diffuser.planners.trajectory_blender import TrajBlender
from comp_diffuser.planners.inverse_dynamics_policy import PolicyInvDyn
from comp_diffuser.models.helpers import (
    Conv1dBlockDd,
    WeightedLossL2InvDynV3,
    WeightedLossL2V2,
)
from comp_diffuser.models.hi_helpers import (
    HiResidualTemporalBlock,
    MlpInvDyn,
    MlpInvDynV2,
    SinusoidalPosEmb2D,
)


def test_public_class_names_follow_capwords():
    assert PolicyInvDyn.__name__ == "PolicyInvDyn"
    assert TrajBlender.__name__ == "TrajBlender"
    assert Conv1dBlockDd.__name__ == "Conv1dBlockDd"
    assert WeightedLossL2V2.__name__ == "WeightedLossL2V2"
    assert WeightedLossL2InvDynV3.__name__ == "WeightedLossL2InvDynV3"
    assert HiResidualTemporalBlock.__name__ == "HiResidualTemporalBlock"
    assert SinusoidalPosEmb2D.__name__ == "SinusoidalPosEmb2D"
    assert MlpInvDyn.__name__ == "MlpInvDyn"
    assert MlpInvDynV2.__name__ == "MlpInvDynV2"


def test_helper_modules_keep_their_shapes():
    conv_block = Conv1dBlockDd(8, 8, 3)
    conv_input = torch.randn(3, 8, 8)
    conv_output = conv_block(conv_input)
    assert conv_output.shape == (3, 8, 8)

    weighted_loss = WeightedLossL2V2(torch.ones(8, 8), action_dim=2)
    pred = torch.randn(3, 8, 8)
    targ = torch.randn(3, 8, 8)
    loss_value, info = weighted_loss(pred, targ)
    assert loss_value.ndim == 0
    assert "a0_loss" in info

    invdyn_loss = WeightedLossL2InvDynV3(torch.ones(8, 8))
    invdyn_value, info = invdyn_loss(pred, targ)
    assert invdyn_value.ndim == 0
    assert info == {}


def test_hi_helper_modules_keep_their_shapes():
    embedding = SinusoidalPosEmb2D(6)
    emb_input = torch.randn(2, 4)
    emb_output = embedding(emb_input)
    assert emb_output.shape == (2, 4, 6)

    residual_block = HiResidualTemporalBlock(8, 8, 6, 8)
    residual_input = torch.randn(3, 8, 8)
    time_input = torch.randn(3, 6)
    residual_output = residual_block(residual_input, time_input)
    assert residual_output.shape == (3, 8, 8)

    mlp = MlpInvDyn(input_dim=4, hidden_dim=[8], output_dim=2)
    mlp_output = mlp(torch.randn(5, 4))
    assert mlp_output.shape == (5, 2)

    mlp_v2 = MlpInvDynV2(
        input_dim=4,
        output_dim=2,
        inv_m_config={
            "inv_hid_dims": [8],
            "act_f": "relu",
            "use_dpout": False,
            "prob_dpout": 0.1,
        },
    )
    mlp_v2_output = mlp_v2(torch.randn(5, 4))
    assert mlp_v2_output.shape == (5, 2)
