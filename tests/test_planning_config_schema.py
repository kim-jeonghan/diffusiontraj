from comp_diffuser.models.stitching.trajectory_stitching_policy import (
    TrajectoryStitchingPolicy,
)
from comp_diffuser.planners.maze_policy import MazePolicy


class _DummyParameter:
    device = "cpu"


class _DummyDiffusionModel:
    def __init__(self, horizon=40):
        self.horizon = horizon
        self.observation_dim = 2
        self.action_dim = 2
        self.len_ovlp_cd = 8

    def eval(self):
        return self

    def parameters(self):
        yield _DummyParameter()


class _DummyNormalizer:
    action_dim = 2


def _make_stitching_policy(policy_config, blender_config):
    return TrajectoryStitchingPolicy(
        diffusion_model=_DummyDiffusionModel(),
        normalizer=_DummyNormalizer(),
        policy_config=policy_config,
        trajectory_blender_config=blender_config,
    )


def test_trajectory_stitching_policy_accepts_readable_config_keys():
    policy = _make_stitching_policy(
        policy_config={
            "num_segments": 6,
            "top_k": 3,
            "trajectory_selection": "rand",
            "inference_schedule": "interleaved",
        },
        blender_config={
            "blend_type": "linear",
            "blend_exponential_beta": 7,
        },
    )

    assert policy.num_segments == 6
    assert policy.top_n == 3
    assert policy.pick_type == "rand"
    assert policy.inference_schedule == "interleaved"
    assert policy.trajectory_blender.blend_type == "linear"
    assert policy.trajectory_blender.exp_beta == 7


def test_maze_policy_accepts_readable_plan_horizon_key():
    diffusion_model = _DummyDiffusionModel(horizon=40)
    policy = MazePolicy(
        diffusion_model=diffusion_model,
        normalizer=_DummyNormalizer(),
        policy_config={"plan_horizon": 96},
    )

    assert policy.plan_horizon == 96
    assert diffusion_model.horizon == 96
