import time

import einops
import numpy as np
import torch

import comp_diffuser.utils as utils
from comp_diffuser.guides.comp.traj_blender import Traj_Blender
from comp_diffuser.guides.comp.trajectory_batches import InverseDynamicsTrajectories
from comp_diffuser.models.helpers import apply_conditioning
from comp_diffuser.models.trajectory_stitching import (
    TrajectoryStitchingGaussianDiffusionWithInverseDynamics,
)
from comp_diffuser.utils.composition.plan_utils import split_trajs_list_by_prob


class TrajectoryStitchingPolicy:

    def __init__(self, diffusion_model, 
                 normalizer, 
                 policy_config,
                 trajectory_blender_config,
                 ):
        self.diffusion_model: TrajectoryStitchingGaussianDiffusionWithInverseDynamics = diffusion_model
        self.diffusion_model.eval()
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        
        self.num_segments = policy_config['ev_n_comp']
        self.top_n = policy_config['ev_top_n']
        self.pick_type = policy_config['ev_pick_type']
        assert self.pick_type in ['first', 'rand']
        
        self.inference_schedule = policy_config.get('ev_cp_infer_t_type', 'interleaved')

        self.trajectory_blender = Traj_Blender(
            diffusion_model,
            normalizer,
            **trajectory_blender_config,
        )
        self.prediction_time_history = []

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device
    



    def generate_conditioned_trajectories_parallel(self, planner_inputs, debug=False, batch_size=1):
        """
        `planner_inputs["start_goal_pairs"]` is an unnormalized array shaped `(2, n_problems, dim)`.
        """
        horizon = self.diffusion_model.horizon
        observation_dim = self.diffusion_model.observation_dim

        start_goal_pairs = planner_inputs['start_goal_pairs']
        start_goal_pairs = torch.tensor(self.normalizer.normalize(start_goal_pairs, 'observations'))
        
        assert start_goal_pairs.ndim == 3 and start_goal_pairs.shape[0] == 2
        num_problems = start_goal_pairs.shape[1]
        
        sample_shape = [batch_size * num_problems, horizon, observation_dim]

        boundary_conditions = {
            0: einops.repeat(start_goal_pairs[0, :, :], 'n d -> (n r) d', r=batch_size).clone(),
            horizon - 1: einops.repeat(start_goal_pairs[1, :, :], 'n d -> (n r) d', r=batch_size).clone(),
        }

        if self.inference_schedule == 'interleaved':
            batched_segment_trajectories = self.diffusion_model.comp_pred_p_loop_n(
                sample_shape,
                boundary_conditions,
                n_comp=self.num_segments,
                return_diffusion=False,
            )
        elif self.inference_schedule == 'global_sync':
            batched_segment_trajectories = self.diffusion_model.comp_pred_p_loop_n_GSC(
                sample_shape,
                boundary_conditions,
                n_comp=self.num_segments,
                return_diffusion=False,
            )
        else:
            raise NotImplementedError

        segment_trajectories_per_problem = split_trajs_list_by_prob(
            batched_segment_trajectories,
            num_problems,
        )
        
        predictions = []
        selected_trajectories = []

        for problem_index in range(num_problems):
            segment_trajectories = segment_trajectories_per_problem[problem_index]
            unnormalized_segment_trajectories = utils.get_np_trajs_list(
                segment_trajectories,
                do_unnorm=True,
                normalizer=self.normalizer,
            )
            sorted_indices, _ = utils.compute_ovlp_dist(
                unnormalized_segment_trajectories,
                self.diffusion_model.len_ovlp_cd,
            )
            top_ranked_segment_trajectories = utils.pick_top_n_trajs(
                unnormalized_segment_trajectories,
                sorted_indices,
                self.top_n,
            )
            merged_top_trajectories = self.trajectory_blender.blend_traj_lists(
                top_ranked_segment_trajectories,
                do_unnorm=False,
            )

            if self.pick_type == 'first':
                selected_trajectory = merged_top_trajectories[0]
            elif self.pick_type == 'rand':
                selected_index = np.random.randint(low=0, high=self.top_n)
                selected_trajectory = merged_top_trajectories[selected_index]
            else:
                raise NotImplementedError

            prediction = TrajectoryStitchingPrediction(
                selected_trajectory,
                merged_top_trajectories,
                unnormalized_segment_trajectories,
            )

            predictions.append(prediction)
            selected_trajectories.append(prediction.pick_traj)

        return predictions, selected_trajectories







    

    def generate_conditioned_trajectory(self, planner_inputs, debug=False, batch_size=1):
        """
        Generate one stitched trajectory for a single planning problem.
        """
        if self.num_segments == 1:
            return self.generate_single_segment_trajectory(planner_inputs, b_s=batch_size)
        
        horizon = self.diffusion_model.horizon
        observation_dim = self.diffusion_model.observation_dim
        sample_shape = [batch_size, horizon, observation_dim]

        start_goal_pairs = planner_inputs['start_goal_pairs']
        start_goal_pairs = torch.tensor(self.normalizer.normalize(start_goal_pairs, 'observations'))
        
        assert start_goal_pairs.ndim == 3 and start_goal_pairs.shape[0] == 2
        
        boundary_conditions = {
            0: einops.repeat(start_goal_pairs[0, :, :], 'n_p d -> (n_p rr) d', rr=batch_size).clone(),
            horizon - 1: einops.repeat(start_goal_pairs[1, :, :], 'n_p d -> (n_p rr) d', rr=batch_size).clone(),
        }

        cur_time = time.time()

        if self.inference_schedule == 'interleaved':
            segment_trajectories = self.diffusion_model.comp_pred_p_loop_n(
                sample_shape, boundary_conditions, n_comp=self.num_segments, return_diffusion=False)
        
        elif self.inference_schedule == 'shared_timestep':
            segment_trajectories = self.diffusion_model.comp_pred_p_loop_n_same_t(
                sample_shape, boundary_conditions, n_comp=self.num_segments, return_diffusion=False)
            
        elif self.inference_schedule == 'global_sync':
            segment_trajectories = self.diffusion_model.comp_pred_p_loop_n_GSC(
                sample_shape, boundary_conditions, n_comp=self.num_segments, return_diffusion=False)
        
        elif self.inference_schedule == 'shared_timestep_parallel':
            segment_trajectories = self.diffusion_model.comp_pred_p_loop_n_same_t_parallel(
                sample_shape, boundary_conditions, n_comp=self.num_segments, return_diffusion=False)
        
        elif self.inference_schedule == 'autoregressive_backward':
            segment_trajectories = self.diffusion_model.comp_pred_p_loop_n_ar_backward(
                sample_shape, boundary_conditions, n_comp=self.num_segments, return_diffusion=False)
        else:
            raise NotImplementedError
        
        self.prediction_time_history.append([self.num_segments, time.time() - cur_time])

        unnormalized_segment_trajectories = utils.get_np_trajs_list(
            segment_trajectories,
            do_unnorm=True,
            normalizer=self.normalizer,
        )
        sorted_indices, _ = utils.compute_ovlp_dist(
            unnormalized_segment_trajectories,
            self.diffusion_model.len_ovlp_cd,
        )
        top_ranked_segment_trajectories = utils.pick_top_n_trajs(
            unnormalized_segment_trajectories,
            sorted_indices,
            self.top_n,
        )
        merged_top_trajectories = self.trajectory_blender.blend_traj_lists(
            top_ranked_segment_trajectories,
            do_unnorm=False,
        )

        if self.pick_type == 'first':
            pick_traj = merged_top_trajectories[0]
        elif self.pick_type == 'rand':
            p_idx = np.random.randint(low=0, high=self.top_n)
            pick_traj = merged_top_trajectories[p_idx]
        else:
            raise NotImplementedError

        out = TrajectoryStitchingPrediction(
            pick_traj,
            merged_top_trajectories,
            unnormalized_segment_trajectories,
        )

        return out
    

    def generate_single_segment_trajectory(self, planner_inputs, debug=False, batch_size=1):
        """
        Generate one trajectory without stitching.
        """
        horizon = self.diffusion_model.horizon
        observation_dim = self.diffusion_model.observation_dim
        sample_shape = [batch_size, horizon, observation_dim]

        start_goal_pairs = planner_inputs['start_goal_pairs']
        start_goal_pairs = torch.tensor(self.normalizer.normalize(start_goal_pairs, 'observations'))

        num_problems = start_goal_pairs.shape[1]
        assert start_goal_pairs.ndim == 3 and start_goal_pairs.shape[0] == 2 and num_problems == 1

        boundary_conditions = {
            0: einops.repeat(start_goal_pairs[0, :, :], 'n d -> (n r) d', r=batch_size).clone(),
            horizon - 1: einops.repeat(start_goal_pairs[1, :, :], 'n d -> (n r) d', r=batch_size).clone(),
        }

        diffusion_inputs = {
            'conditioning_mode': 'boundary_only',
            'boundary_conditions': boundary_conditions,
            't_type': '0',
            'traj_full': np.random.random(size=(boundary_conditions[0].shape[0],)),
        }

        predicted_trajectories = self.diffusion_model.conditional_sample(g_cond=diffusion_inputs)

        predicted_trajectories = apply_conditioning(predicted_trajectories, boundary_conditions, 0)

        predicted_trajectories = utils.to_np(predicted_trajectories)
        unnormalized_predicted_trajectories = self.normalizer.unnormalize(
            predicted_trajectories,
            'observations',
        )

        out = TrajectoryStitchingPrediction(
            unnormalized_predicted_trajectories[0],
            unnormalized_predicted_trajectories,
            unnormalized_predicted_trajectories,
        )

        return out





    




    def _format_g_cond(self, g_cond, batch_size):
        
        traj_f = g_cond['traj_full'] 
        ## normalize the traj
        traj_f =  self.normalizer.normalize(traj_f, 'observations')
        traj_f = torch.tensor(traj_f, dtype=torch.float32, device='cuda:0')

        traj_f = einops.repeat(traj_f, 'b h d -> (repeat b) h d', repeat=batch_size)
        
        g_cond['traj_full'] = traj_f
        
        return g_cond


    def gen_cond(self, g_cond, debug=False, batch_size=1):
        '''conditional sampling
        conditioned on start and end chunks, just for sanity test
        '''

        g_cond = self._format_g_cond(g_cond, batch_size)

        
        sample = self.diffusion_model.conditional_sample(g_cond)

        
        actions = np.zeros(shape=(*sample.shape[0:2], self.action_dim))
        sample = utils.to_np(sample)
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)
        
        ## extract first action
        action = actions[0, 0]

        # pdb.set_trace()

        # if debug:
        normed_observations = sample[:, :, 0:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = InverseDynamicsTrajectories(actions, observations)
        return action, trajectories
        




    

class TrajectoryStitchingPrediction:
    def __init__(self, pick_traj, 
                 trajs_list_topn_bl, trajs_list_np_un) -> None:
        '''
        pick_traj: np2d unnormed (tot_hzn,dim), the traj to follow
        trajs_list_topn_bl: np3d unnormed (B,tot_hzn,dim), all topn


        '''
        self.pick_traj = pick_traj
        self.trajs_list_topn_bl = trajs_list_topn_bl
        self.trajs_list_np_un = trajs_list_np_un
