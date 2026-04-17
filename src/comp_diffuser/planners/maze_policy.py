import time

import einops
import torch

from ..models.common.helpers import apply_conditioning
from ..models.diffusion.maze_diffusion import MazeGaussianDiffusion
from ..models.stitching.trajectory_stitching_policy import (
    TrajectoryStitchingPrediction,
)
from ..utils.arrays import to_np
from ..utils.planning_config import normalize_maze_policy_config


class MazePolicy:

    def __init__(
        self,
        diffusion_model: MazeGaussianDiffusion,
        normalizer,
        policy_config,
    ):
        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        policy_config = normalize_maze_policy_config(
            policy_config, diffusion_model.horizon
        )
        self.plan_horizon = policy_config["plan_horizon"]
        self.diffusion_model.horizon = self.plan_horizon
        self.prediction_time_history = []
        self.return_diffusion = False

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def generate_conditioned_trajectories_parallel(
        self, planner_inputs, debug=False, batch_size=1
    ):
        """
        `planner_inputs["start_goal_pairs"]` is an unnormalized array shaped `(2, n_problems, dim)`.
        """
        horizon = self.diffusion_model.horizon
        # observation_dim = self.diffusion_model.observation_dim

        start_goal_pairs = planner_inputs["start_goal_pairs"]
        start_goal_pairs = torch.tensor(
            self.normalizer.normalize(start_goal_pairs, "observations")
        )

        assert start_goal_pairs.ndim == 3 and start_goal_pairs.shape[0] == 2
        num_problems = start_goal_pairs.shape[1]

        batch_size = 1

        boundary_conditions = {
            0: einops.repeat(
                start_goal_pairs[0, :, :], "n d -> (n r) d", r=batch_size
            ).clone(),
            horizon
            - 1: einops.repeat(
                start_goal_pairs[1, :, :], "n d -> (n r) d", r=batch_size
            ).clone(),
        }

        diffusion_inputs = dict(boundary_conditions=boundary_conditions)

        sample = self.diffusion_model.conditional_sample(
            diffusion_inputs,
            horizon=self.plan_horizon,
            verbose=False,
            return_diffusion=self.return_diffusion,
        )
        if self.return_diffusion:
            self.diffusion_process_all = sample[1]
            sample = sample[0]

        sample = apply_conditioning(sample, boundary_conditions, 0)

        sample = to_np(sample)
        normalized_observations = sample[:, :, 0:]
        predicted_observation_trajectories = self.normalizer.unnormalize(
            normalized_observations,
            "observations",
        )

        predictions = []
        selected_trajectories = []

        assert len(predicted_observation_trajectories) == num_problems

        for problem_index in range(num_problems):
            selected_trajectory = predicted_observation_trajectories[problem_index]
            prediction = TrajectoryStitchingPrediction(selected_trajectory, None, None)
            predictions.append(prediction)
            selected_trajectories.append(prediction.pick_traj)

        return predictions, selected_trajectories

    def generate_conditioned_trajectory(
        self, planner_inputs, debug=False, batch_size=1
    ):
        start_time = time.time()
        predictions, _ = self.generate_conditioned_trajectories_parallel(
            planner_inputs,
            debug,
            batch_size,
        )
        assert len(predictions) == 1
        self.prediction_time_history.append([1, time.time() - start_time])
        return predictions[0]
