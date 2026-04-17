import einops
import numpy as np
import torch
from tqdm import tqdm

from ...utils.arrays import batch_repeat_tensor_in_dict
from ...utils.eval_utils import print_color
from ..common.base_diffusion import BaseDiffusion
from ..common.helpers import apply_conditioning
from ..common.model_outputs import ModelPrediction
from ..networks.trajectory_stitching_temporal_unet import (
    StitchingTemporalUNet,
)


class StitchingDiffusion(BaseDiffusion):
    def __init__(
        self,
        model: StitchingTemporalUNet,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        diff_config={},
    ):
        super().__init__(
            model=model,
            horizon=horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=n_timesteps,
            loss_type=loss_type,
            clip_denoised=clip_denoised,
            predict_epsilon=predict_epsilon,
            action_weight=action_weight,
            loss_discount=loss_discount,
            loss_weights=loss_weights,
            diff_config=diff_config,
        )
        self.len_ovlp_cd = self.diff_config["len_ovlp_cd"]
        self.is_direct_train = self.diff_config.get("is_direct_train", False)
        if self.is_direct_train:
            self.infer_deno_type = self.diff_config["infer_deno_type"]
            self.w_loss_type = self.diff_config["w_loss_type"]
            self.is_train_inv = False
            self.is_train_diff = True
            self.tr_cond_type = "stgl"
            self.condition_guidance_w = self.diff_config.get(
                "condition_guidance_w", 2.0
            )
            self.var_temp = 1.0
            self.tr_inpat_prob = self.diff_config["tr_inpat_prob"]
            self.tr_ovlp_prob = self.diff_config["tr_ovlp_prob"]
            print_color(f"{self.diff_config['tr_1side_drop_prob']=}")
            self.tr_no_ovlp_none = self.diff_config.get("tr_no_ovlp_none", False)
            print_color(f"{self.tr_no_ovlp_none=}")
            assert self.tr_inpat_prob + self.tr_ovlp_prob == 1.0
            ddim_set_alpha_to_one = self.diff_config.get("ddim_set_alpha_to_one", True)
            self.final_alpha_cumprod = (
                torch.tensor([1.0])
                if ddim_set_alpha_to_one
                else torch.clone(self.schedule.alphas_cumprod[0:1])
            )
            self.num_train_timesteps = self.n_timesteps
            self.ddim_num_inference_steps = self.diff_config.get("ddim_steps", 50)
            self.ddim_eta = 1.0
            self.use_ddim = True
            self.use_eta_noise = False

    # ------------------------------------------ sampling ------------------------------------------#
    def p_mean_variance(self, x_t, t_2d, cond, return_modelout=False):
        if cond["use_conditioning"]:
            x_2, t_2d_2, cond_2 = batch_repeat_tensor_in_dict(x_t, t_2d, cond, n_rp=2)
            assert (t_2d_2[0] == t_2d_2[0, 0]).all(), "sanity check"
            t_1d_2 = t_2d_2[:, 0]
            out = self.model(x_2, t_1d_2, cond_2, force_dropout=True, half_fd=True)
            out_cd = out[: len(x_t), :, :]
            out_uncd = out[len(x_t) :, :, :]
            out_model = out_uncd + self.condition_guidance_w * (out_cd - out_uncd)
        else:
            out_model = self.small_model_pred(x_t, t_2d, cond)

        x_0 = self.schedule.predict_start_from_noise(
            x_t, t_2d=t_2d, noise=out_model, predict_epsilon=self.predict_epsilon
        )
        x_0.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = (
            self.schedule.q_posterior(x_0=x_0, x_t=x_t, t=t_2d)
        )

        if return_modelout:
            pred_epsilon = self.schedule.predict_noise_from_start(
                x_t=x_t, t_2d=t_2d, x_0=x_0
            )
            return (
                model_mean,
                posterior_variance,
                posterior_log_variance,
                x_0,
                pred_epsilon,
            )
        return model_mean, posterior_variance, posterior_log_variance

    def get_segment_conditioning(self, x, g_cond, timesteps):
        if g_cond["conditioning_mode"] == "overlap_only":
            st_traj, end_traj = self.extract_ovlp_from_full(g_cond["traj_full"])
            x, segment_conditioning = self.create_eval_conditioning(
                x_et=x,
                st_traj=st_traj,
                end_traj=end_traj,
                t_1d_st=timesteps[:, 0],
                t_1d_end=timesteps[:, 0],
                t_type=g_cond["t_type"],
                is_noisy=False,
                boundary_conditions={},
            )
            segment_conditioning["use_conditioning"] = True
        elif g_cond["conditioning_mode"] == "boundary_only":
            x, segment_conditioning = self.create_eval_conditioning(
                x_et=x,
                st_traj=None,
                end_traj=None,
                t_1d_st=timesteps[:, 0],
                t_1d_end=timesteps[:, 0],
                t_type=g_cond["t_type"],
                is_noisy=False,
                boundary_conditions=g_cond["boundary_conditions"],
            )
            segment_conditioning["use_conditioning"] = True
        elif g_cond["conditioning_mode"] == "start_boundary_end_overlap":
            _, end_traj = self.extract_ovlp_from_full(g_cond["traj_full"])
            x, segment_conditioning = self.create_eval_conditioning(
                x_et=x,
                st_traj=None,
                end_traj=end_traj,
                t_1d_st=timesteps[:, 0],
                t_1d_end=timesteps[:, 0],
                t_type=g_cond["t_type"],
                is_noisy=False,
                boundary_conditions={0: g_cond["boundary_conditions"][0]},
            )
            segment_conditioning["use_conditioning"] = True
        elif g_cond["conditioning_mode"] == "start_overlap_goal_boundary":
            st_traj, _ = self.extract_ovlp_from_full(g_cond["traj_full"])
            x, segment_conditioning = self.create_eval_conditioning(
                x_et=x,
                st_traj=st_traj,
                end_traj=None,
                t_1d_st=timesteps[:, 0],
                t_1d_end=timesteps[:, 0],
                t_type=g_cond["t_type"],
                is_noisy=False,
                boundary_conditions={
                    self.horizon - 1: g_cond["boundary_conditions"][self.horizon - 1]
                },
            )
            segment_conditioning["use_conditioning"] = True
        elif not g_cond["conditioning_mode"]:
            segment_conditioning = dict(
                start_overlap_is_drop=None,
                end_overlap_is_drop=None,
                uses_start_inpaint=torch.zeros_like(x[:, 0, 0]).to(torch.bool),
                uses_end_inpaint=torch.zeros_like(x[:, 0, 0]).to(torch.bool),
            )
            segment_conditioning["use_conditioning"] = False
        else:
            raise NotImplementedError
        return x, segment_conditioning

    @torch.no_grad()
    def p_sample_loop(self, shape, g_cond, return_diffusion=False):
        device = self.schedule.betas.device
        batch_size = shape[0]
        x = self.var_temp * torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, self.n_timesteps))):
            timesteps = torch.full(
                (batch_size, self.horizon), i, device=device, dtype=torch.long
            )
            x, segment_conditioning = self.get_segment_conditioning(
                x, g_cond, timesteps
            )
            x = self.p_sample(x, segment_conditioning, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(self, g_cond, *args, horizon=None, **kwargs):
        batch_size = len(g_cond["traj_full"])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        if self.infer_deno_type == "same":
            if self.use_ddim:
                return self.ddim_p_sample_loop(shape, g_cond, *args, **kwargs)
            return self.p_sample_loop(shape, g_cond, *args, **kwargs)
        raise NotImplementedError

    @torch.no_grad()
    def sample_unCond(self, batch_size, *args, horizon=None, **kwargs):
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)
        g_cond = dict(conditioning_mode=False)

        if self.infer_deno_type == "same":
            return self.p_sample_loop(shape, g_cond, *args, **kwargs)
        raise NotImplementedError

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, g_cond, return_diffusion=False):
        print_color(f"ddim steps: {self.ddim_num_inference_steps}", c="y")

        device = self.schedule.betas.device
        batch_size = shape[0]
        x = self.var_temp * torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        for i in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), i, device=device, dtype=torch.long
            )
            x, segment_conditioning = self.get_segment_conditioning(
                x, g_cond, timesteps
            )
            x = self.ddim_p_sample(
                x,
                segment_conditioning,
                timesteps,
                self.ddim_eta,
                use_clipped_model_output=True,
            )

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    # ------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_0, x_noisy, noise, t_2d, cond, batch_loss_w=None):
        assert self.w_loss_type == "all"
        batch_loss_w = torch.ones_like(x_0[:, :, :1])
        batch_loss_w[cond["uses_start_inpaint"], 0] = 0.0
        batch_loss_w[cond["uses_end_inpaint"], self.horizon - 1] = 0.0
        assert x_0.shape[1] == self.horizon

        pred = self.small_model_pred(x_noisy, t_2d, cond)
        assert x_noisy.shape == pred.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(pred, noise, ext_loss_w=batch_loss_w)
        else:
            loss, info = self.loss_fn(pred, x_0, ext_loss_w=batch_loss_w)

        return loss, info

    def extract_ovlp_from_full(self, x: torch.Tensor):
        """x: either np or tensor"""
        st_traj = x[:, : self.len_ovlp_cd, :]
        end_traj = x[:, -self.len_ovlp_cd :, :]
        if torch.is_tensor(st_traj):
            assert torch.is_tensor(end_traj)
            st_traj = st_traj.detach().clone()
            end_traj = end_traj.detach().clone()
        else:
            assert isinstance(st_traj, np.ndarray)
            assert isinstance(end_traj, np.ndarray)
        return st_traj, end_traj

    def create_train_conditioning(
        self,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
        start_timesteps_1d: torch.Tensor,
        end_timesteps_1d: torch.Tensor,
        boundary_conditions: dict,
        is_rand,
    ):
        """t_1d: (B,)"""
        batch_size = x_clean.shape[0]
        all_drop_prob = self.diff_config["tr_1side_drop_prob"]
        device = x_clean.device

        st_is_all_drop = torch.rand(size=(batch_size,), device=device) < all_drop_prob
        end_is_all_drop = torch.rand(size=(batch_size,), device=device) < all_drop_prob

        st_cd_use_ovlp = (
            torch.rand(size=(batch_size,), device=device) < self.tr_ovlp_prob
        )
        end_cd_use_ovlp = (
            torch.rand(size=(batch_size,), device=device) < self.tr_ovlp_prob
        )

        st_cd_use_inpat = ~st_cd_use_ovlp
        end_cd_use_inpat = ~end_cd_use_ovlp

        st_cd_use_ovlp[st_is_all_drop] = False
        st_cd_use_inpat[st_is_all_drop] = False
        end_cd_use_ovlp[end_is_all_drop] = False
        end_cd_use_inpat[end_is_all_drop] = False

        if self.tr_no_ovlp_none:
            tmp_st_nn = torch.logical_and(st_cd_use_ovlp, end_is_all_drop)
            tmp_stnn_vv = (
                torch.rand_like(tmp_st_nn.to(torch.float32))
                < self.diff_config["non_repla_inpat_prob"]
            )
            tmp_stnn_inpat = torch.logical_and(tmp_st_nn, tmp_stnn_vv)
            st_cd_use_ovlp[tmp_st_nn] = False
            st_cd_use_inpat[tmp_stnn_inpat] = True

            tmp_end_nn = torch.logical_and(st_is_all_drop, end_cd_use_ovlp)
            tmp_endnn_vv = (
                torch.rand_like(tmp_end_nn.to(torch.float32))
                < self.diff_config["non_repla_inpat_prob"]
            )
            tmp_endnn_inpat = torch.logical_and(tmp_end_nn, tmp_endnn_vv)
            end_cd_use_ovlp[tmp_end_nn] = False
            end_cd_use_inpat[tmp_endnn_inpat] = True

        cond_st = {0: boundary_conditions[0][st_cd_use_inpat]}
        x_noisy[st_cd_use_inpat] = apply_conditioning(
            x_noisy[st_cd_use_inpat], conditions=cond_st, action_dim=0
        )
        cond_end = {
            self.horizon - 1: boundary_conditions[self.horizon - 1][end_cd_use_inpat]
        }
        x_noisy[end_cd_use_inpat] = apply_conditioning(
            x_noisy[end_cd_use_inpat], cond_end, 0
        )

        if is_rand:
            start_timesteps_1d = start_timesteps_1d - torch.randint_like(
                start_timesteps_1d, low=0, high=2
            )
            end_timesteps_1d = end_timesteps_1d - torch.randint_like(
                end_timesteps_1d, low=0, high=2
            )
        else:
            assert False

        start_timesteps_1d = torch.clamp(
            start_timesteps_1d, min=0, max=self.n_timesteps - 1
        )
        end_timesteps_1d = torch.clamp(
            end_timesteps_1d, min=0, max=self.n_timesteps - 1
        )

        t_2d_st = torch.repeat_interleave(
            start_timesteps_1d[:, None], repeats=self.len_ovlp_cd, dim=1
        )
        t_2d_end = torch.repeat_interleave(
            end_timesteps_1d[:, None], repeats=self.len_ovlp_cd, dim=1
        )

        st_traj = x_clean[:, : self.len_ovlp_cd, :].detach().clone()
        st_traj = self.schedule.q_sample(x_0=st_traj, t_2d=t_2d_st)

        end_traj = x_clean[:, -self.len_ovlp_cd :, :].detach().clone()
        end_traj = self.schedule.q_sample(x_0=end_traj, t_2d=t_2d_end)

        segment_conditioning = {
            "start_overlap_is_drop": ~st_cd_use_ovlp,
            "end_overlap_is_drop": ~end_cd_use_ovlp,
            "start_overlap_traj": st_traj,
            "end_overlap_traj": end_traj,
            "start_overlap_t": start_timesteps_1d,
            "end_overlap_t": end_timesteps_1d,
            "uses_start_inpaint": st_cd_use_inpat,
            "uses_end_inpaint": end_cd_use_inpat,
        }
        return x_noisy, segment_conditioning

    def loss(self, x_clean, boundary_conditions):
        assert self.is_direct_train
        batch_size = len(x_clean)
        t_1d = torch.randint(
            0, self.n_timesteps, (batch_size, 1), device=x_clean.device
        ).long()
        t_2d = torch.repeat_interleave(t_1d, repeats=self.horizon, dim=1)

        noise = torch.randn_like(x_clean)
        x_noisy = self.schedule.q_sample(x_0=x_clean, t_2d=t_2d, noise=noise)
        x_noisy, segment_conditioning = self.create_train_conditioning(
            x_clean,
            x_noisy,
            t_1d[:, 0],
            t_1d[:, 0].clone(),
            boundary_conditions,
            is_rand=True,
        )

        return self.p_losses(x_clean, x_noisy, noise, t_2d, segment_conditioning)

    def model_predictions(self, x, t_2d, cond: dict):
        if cond["use_conditioning"]:
            x_2, t_2d_2, cond_2 = batch_repeat_tensor_in_dict(x, t_2d, cond, n_rp=2)
            assert (t_2d_2[0] == t_2d_2[0, 0]).all(), "sanity check"
            t_1d_2 = t_2d_2[:, 0]
            out = self.model(x_2, t_1d_2, cond_2, force_dropout=True, half_fd=True)
            out_cd = out[: len(x), :, :]
            out_uncd = out[len(x) :, :, :]
            out_model = out_uncd + self.condition_guidance_w * (out_cd - out_uncd)
        else:
            out_model = self.small_model_pred(x, t_2d, cond)

        out_pred = out_model

        if self.predict_epsilon:
            pred_noise = torch.clamp(out_pred, -self.clip_noise, self.clip_noise)
            x_0 = self.schedule.predict_start_from_noise(
                x_t=x, t_2d=t_2d, noise=pred_noise, predict_epsilon=True
            )
        else:
            x_0 = out_pred
            pred_noise = self.schedule.predict_noise_from_start(x, t_2d, x_0)

        x_0.clamp_(-1.0, 1.0)

        return ModelPrediction(pred_noise, x_0, out_pred)

    def get_total_hzn(self, num_comp):
        return num_comp * self.horizon - (num_comp - 1) * self.len_ovlp_cd

    def create_eval_conditioning(
        self,
        x_et: torch.Tensor,
        st_traj,
        end_traj,
        t_1d_st: torch.Tensor,
        t_1d_end: torch.Tensor,
        t_type: str,
        is_noisy,
        boundary_conditions: dict,
    ):
        """
        if st_traj is not None, then do st traj inpainting;
        if end_traj is not None, then do end traj inpainting;
        if 0 in boundary_conditions, then do start inpainting;
        if hzn-1 in boundary_conditions, then do end inpainting;
        """
        assert t_1d_st.ndim == 1 and t_1d_end.ndim == 1

        batch_size = x_et.shape[0]
        device = x_et.device

        if st_traj is None:
            st_is_drop = None
        else:
            st_is_drop = torch.zeros(
                size=(batch_size,), dtype=torch.bool, device=device
            )
            assert 0 not in boundary_conditions

        if 0 in boundary_conditions:
            assert st_traj is None
            x_et = apply_conditioning(x_et, {0: boundary_conditions[0]}, 0)
            is_st_inpat = torch.ones(
                size=(batch_size,), dtype=torch.bool, device=device
            )
        else:
            is_st_inpat = torch.zeros(
                size=(batch_size,), dtype=torch.bool, device=device
            )

        hzn_minus1 = self.horizon - 1
        if end_traj is None:
            end_is_drop = None
        else:
            end_is_drop = torch.zeros(
                size=(batch_size,), dtype=torch.bool, device=device
            )
            assert hzn_minus1 not in boundary_conditions

        if hzn_minus1 in boundary_conditions:
            assert end_traj is None
            x_et = apply_conditioning(
                x_et, {hzn_minus1: boundary_conditions[hzn_minus1]}, 0
            )
            is_end_inpat = torch.ones(
                size=(batch_size,), dtype=torch.bool, device=device
            )
        else:
            is_end_inpat = torch.zeros(
                size=(batch_size,), dtype=torch.bool, device=device
            )

        assert ((st_traj is not None) or 0 in boundary_conditions) and (
            (end_traj is not None) or hzn_minus1 in boundary_conditions
        )

        if t_type == "rand":
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)
        elif t_type == "-1":
            t_1d_st = t_1d_st - torch.ones_like(t_1d_st)
            t_1d_end = t_1d_end - torch.ones_like(t_1d_end)
        elif t_type == "0":
            pass
        else:
            raise NotImplementedError

        t_1d_st = torch.clamp(t_1d_st, min=0, max=self.n_timesteps - 1)
        t_1d_end = torch.clamp(t_1d_end, min=0, max=self.n_timesteps - 1)

        t_2d_st = torch.repeat_interleave(
            t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1
        )
        t_2d_end = torch.repeat_interleave(
            t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1
        )

        if st_traj is None:
            pass
        elif not is_noisy:
            st_traj = self.schedule.q_sample(x_0=st_traj, t_2d=t_2d_st)
        else:
            st_traj = st_traj.clone()

        if end_traj is None:
            pass
        elif not is_noisy:
            end_traj = self.schedule.q_sample(x_0=end_traj, t_2d=t_2d_end)
        else:
            end_traj = end_traj.clone()

        segment_conditioning = {
            "start_overlap_is_drop": st_is_drop,
            "end_overlap_is_drop": end_is_drop,
            "start_overlap_traj": st_traj,
            "end_overlap_traj": end_traj,
            "start_overlap_t": t_1d_st,
            "end_overlap_t": t_1d_end,
            "uses_start_inpaint": is_st_inpat,
            "uses_end_inpaint": is_end_inpat,
        }
        return x_et, segment_conditioning

    def average_overlap_chunk_global_sync(self, segment_samples):
        """Average the overlap region between adjacent segment samples."""
        assert self.horizon / self.len_ovlp_cd >= 2, "otherwise contain overlap."
        segment_samples = [segment.clone() for segment in segment_samples]
        if len(segment_samples) <= 2:
            pass
        else:
            for segment_index in range(1, len(segment_samples)):
                previous_overlap = segment_samples[segment_index - 1][
                    :, -self.len_ovlp_cd :
                ]
                current_overlap = segment_samples[segment_index][:, : self.len_ovlp_cd]
                averaged_overlap = (previous_overlap + current_overlap) / 2
                segment_samples[segment_index - 1][
                    :, -self.len_ovlp_cd :
                ] = averaged_overlap
                segment_samples[segment_index][:, : self.len_ovlp_cd] = averaged_overlap
        return segment_samples

    # ------------------------------------------ comp sampling loops ------------------------------------------#

    @torch.no_grad()
    def comp_pred_p_loop_n(
        self,
        shape,
        boundary_conditions,
        num_segments,
        do_mcmc=False,
        return_diffusion=False,
    ):
        """Generate multiple stitched trajectory segments with interleaved denoising."""
        assert num_segments >= 2 and not do_mcmc
        device = self.schedule.betas.device
        batch_size = shape[0]
        horizon = shape[1]

        segment_samples = [
            torch.randn(shape, device=device) for _ in range(num_segments)
        ]
        diffusion_history = [segment_samples]
        assert len(boundary_conditions[0]) == shape[0]

        time_idx = (
            self.ddim_set_timesteps(self.ddim_num_inference_steps)
            if self.use_ddim
            else reversed(range(0, self.n_timesteps))
        )

        for time_index in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), time_index, device=device, dtype=torch.long
            )

            for segment_index in range(num_segments):
                current_segment = segment_samples[segment_index]

                if segment_index == 0:
                    next_segment = segment_samples[segment_index + 1]
                    next_segment_start_overlap, _ = self.extract_ovlp_from_full(
                        next_segment
                    )
                    current_segment, segment_conditioning = (
                        self.create_eval_conditioning(
                            x_et=current_segment,
                            st_traj=None,
                            end_traj=next_segment_start_overlap,
                            t_1d_st=timesteps[:, 0],
                            t_1d_end=timesteps[:, 0],
                            t_type="0",
                            is_noisy=True,
                            boundary_conditions={0: boundary_conditions[0]},
                        )
                    )
                    segment_conditioning["use_conditioning"] = True
                    if do_mcmc:
                        current_segment = self.resample_same_t_mcmc(
                            current_segment, segment_conditioning, timesteps
                        )
                    if self.use_ddim:
                        current_segment = self.ddim_p_sample(
                            current_segment,
                            segment_conditioning,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        current_segment = self.p_sample(
                            current_segment, segment_conditioning, timesteps
                        )
                    segment_samples[segment_index] = current_segment

                elif segment_index > 0 and segment_index < num_segments - 1:
                    previous_segment = segment_samples[segment_index - 1]
                    _, previous_segment_end_overlap = self.extract_ovlp_from_full(
                        previous_segment
                    )
                    next_segment = segment_samples[segment_index + 1]
                    next_segment_start_overlap, _ = self.extract_ovlp_from_full(
                        next_segment
                    )
                    current_segment, segment_conditioning = (
                        self.create_eval_conditioning(
                            x_et=current_segment,
                            st_traj=previous_segment_end_overlap,
                            end_traj=next_segment_start_overlap,
                            t_1d_st=timesteps[:, 0] - 1,
                            t_1d_end=timesteps[:, 0],
                            t_type="0",
                            is_noisy=True,
                            boundary_conditions={},
                        )
                    )
                    segment_conditioning["use_conditioning"] = True
                    if do_mcmc:
                        current_segment = self.resample_same_t_mcmc(
                            current_segment, segment_conditioning, timesteps
                        )
                    if self.use_ddim:
                        current_segment = self.ddim_p_sample(
                            current_segment,
                            segment_conditioning,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        current_segment = self.p_sample(
                            current_segment, segment_conditioning, timesteps
                        )
                    segment_samples[segment_index] = current_segment

                elif segment_index == num_segments - 1:
                    previous_segment = segment_samples[segment_index - 1]
                    _, previous_segment_end_overlap = self.extract_ovlp_from_full(
                        previous_segment
                    )
                    current_segment, segment_conditioning = (
                        self.create_eval_conditioning(
                            x_et=current_segment,
                            st_traj=previous_segment_end_overlap,
                            end_traj=None,
                            t_1d_st=timesteps[:, 0] - 1,
                            t_1d_end=timesteps[:, 0],
                            t_type="0",
                            is_noisy=True,
                            boundary_conditions={
                                horizon - 1: boundary_conditions[horizon - 1]
                            },
                        )
                    )
                    segment_conditioning["use_conditioning"] = True
                    if do_mcmc:
                        current_segment = self.resample_same_t_mcmc(
                            current_segment, segment_conditioning, timesteps
                        )
                    if self.use_ddim:
                        current_segment = self.ddim_p_sample(
                            current_segment,
                            segment_conditioning,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        current_segment = self.p_sample(
                            current_segment, segment_conditioning, timesteps
                        )
                    segment_samples[segment_index] = current_segment

            if return_diffusion:
                diffusion_history.append([segment for segment in segment_samples])

        segment_samples[0] = apply_conditioning(
            segment_samples[0], {0: boundary_conditions[0]}, 0
        )
        segment_samples[-1] = apply_conditioning(
            segment_samples[-1], {horizon - 1: boundary_conditions[horizon - 1]}, 0
        )

        if return_diffusion:
            return segment_samples, diffusion_history
        return segment_samples

    @torch.no_grad()
    def comp_pred_p_loop_n_same_t(
        self,
        shape,
        boundary_conditions,
        num_segments,
        do_mcmc=False,
        return_diffusion=False,
    ):
        """Generate stitched trajectory segments while keeping a shared timestep across segments."""
        assert num_segments >= 2 and not do_mcmc
        device = self.schedule.betas.device
        batch_size = shape[0]
        horizon = shape[1]

        segment_samples = [
            torch.randn(shape, device=device) for _ in range(num_segments)
        ]
        diffusion_history = [segment_samples]
        assert len(boundary_conditions[0]) == shape[0]

        time_idx = (
            self.ddim_set_timesteps(self.ddim_num_inference_steps)
            if self.use_ddim
            else reversed(range(0, self.n_timesteps))
        )

        for time_index in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), time_index, device=device, dtype=torch.long
            )
            next_segment_samples = [None for _ in range(num_segments)]

            for segment_index in range(num_segments):
                current_segment = segment_samples[segment_index]

                if segment_index == 0:
                    next_segment = segment_samples[segment_index + 1]
                    next_segment_start_overlap, _ = self.extract_ovlp_from_full(
                        next_segment
                    )
                    current_segment, segment_conditioning = (
                        self.create_eval_conditioning(
                            x_et=current_segment,
                            st_traj=None,
                            end_traj=next_segment_start_overlap,
                            t_1d_st=timesteps[:, 0],
                            t_1d_end=timesteps[:, 0],
                            t_type="0",
                            is_noisy=True,
                            boundary_conditions={0: boundary_conditions[0]},
                        )
                    )
                    segment_conditioning["use_conditioning"] = True
                    if do_mcmc:
                        current_segment = self.resample_same_t_mcmc(
                            current_segment, segment_conditioning, timesteps
                        )
                    if self.use_ddim:
                        current_segment = self.ddim_p_sample(
                            current_segment,
                            segment_conditioning,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        current_segment = self.p_sample(
                            current_segment, segment_conditioning, timesteps
                        )
                    next_segment_samples[segment_index] = current_segment

                elif segment_index > 0 and segment_index < num_segments - 1:
                    previous_segment = segment_samples[segment_index - 1]
                    _, previous_segment_end_overlap = self.extract_ovlp_from_full(
                        previous_segment
                    )
                    next_segment = segment_samples[segment_index + 1]
                    next_segment_start_overlap, _ = self.extract_ovlp_from_full(
                        next_segment
                    )
                    current_segment, segment_conditioning = (
                        self.create_eval_conditioning(
                            x_et=current_segment,
                            st_traj=previous_segment_end_overlap,
                            end_traj=next_segment_start_overlap,
                            t_1d_st=timesteps[:, 0],
                            t_1d_end=timesteps[:, 0],
                            t_type="0",
                            is_noisy=True,
                            boundary_conditions={},
                        )
                    )
                    segment_conditioning["use_conditioning"] = True
                    if do_mcmc:
                        current_segment = self.resample_same_t_mcmc(
                            current_segment, segment_conditioning, timesteps
                        )
                    if self.use_ddim:
                        current_segment = self.ddim_p_sample(
                            current_segment,
                            segment_conditioning,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        current_segment = self.p_sample(
                            current_segment, segment_conditioning, timesteps
                        )
                    next_segment_samples[segment_index] = current_segment

                elif segment_index == num_segments - 1:
                    previous_segment = segment_samples[segment_index - 1]
                    _, previous_segment_end_overlap = self.extract_ovlp_from_full(
                        previous_segment
                    )
                    current_segment, segment_conditioning = (
                        self.create_eval_conditioning(
                            x_et=current_segment,
                            st_traj=previous_segment_end_overlap,
                            end_traj=None,
                            t_1d_st=timesteps[:, 0],
                            t_1d_end=timesteps[:, 0],
                            t_type="0",
                            is_noisy=True,
                            boundary_conditions={
                                horizon - 1: boundary_conditions[horizon - 1]
                            },
                        )
                    )
                    segment_conditioning["use_conditioning"] = True
                    if do_mcmc:
                        current_segment = self.resample_same_t_mcmc(
                            current_segment, segment_conditioning, timesteps
                        )
                    if self.use_ddim:
                        current_segment = self.ddim_p_sample(
                            current_segment,
                            segment_conditioning,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        current_segment = self.p_sample(
                            current_segment, segment_conditioning, timesteps
                        )
                    next_segment_samples[segment_index] = current_segment

            segment_samples = next_segment_samples
            if return_diffusion:
                diffusion_history.append([segment for segment in segment_samples])

        segment_samples[0] = apply_conditioning(
            segment_samples[0], {0: boundary_conditions[0]}, 0
        )
        segment_samples[-1] = apply_conditioning(
            segment_samples[-1], {horizon - 1: boundary_conditions[horizon - 1]}, 0
        )

        if return_diffusion:
            return segment_samples, diffusion_history
        return segment_samples

    @torch.no_grad()
    def comp_pred_p_loop_n_same_t_parallel(
        self,
        shape,
        boundary_conditions,
        num_segments,
        do_mcmc=False,
        return_diffusion=False,
    ):
        """Generate stitched trajectory segments with shared-timestep parallel denoising."""
        assert num_segments >= 2 and not do_mcmc
        device = self.schedule.betas.device
        batch_size = shape[0]
        horizon = shape[1]

        segment_samples = [
            torch.randn(shape, device=device) for _ in range(num_segments)
        ]
        diffusion_history = [segment_samples]
        assert len(boundary_conditions[0]) == shape[0]

        time_idx = (
            self.ddim_set_timesteps(self.ddim_num_inference_steps)
            if self.use_ddim
            else reversed(range(0, self.n_timesteps))
        )

        for i_t in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), i_t, device=device, dtype=torch.long
            )

            x_p_list_cur_t = [None for _ in range(num_segments)]
            tj_cond_cur_list = [None for _ in range(num_segments)]

            for i_tj in range(num_segments):
                x_p_i = segment_samples[i_tj]

                if i_tj == 0:
                    x_p_i_plus_1 = segment_samples[i_tj + 1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0],
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={0: boundary_conditions[0]},
                    )
                elif i_tj > 0 and i_tj < num_segments - 1:
                    x_p_i_minus_1 = segment_samples[i_tj - 1]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)
                    x_p_i_plus_1 = segment_samples[i_tj + 1]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0],
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={},
                    )
                elif i_tj == num_segments - 1:
                    x_p_i_minus_1 = segment_samples[i_tj - 1]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0],
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={
                            horizon - 1: boundary_conditions[horizon - 1]
                        },
                    )

                x_p_list_cur_t[i_tj] = x_p_i
                tj_cond_cur_list[i_tj] = tj_cond_p_i

            def merge_tj_cond_list(tj_cond_list: list):
                len_tjc = len(tj_cond_list)
                assert len_tjc >= 2
                keys_tjc = tj_cond_list[0].keys()
                mg_dict = {}
                b_s = tj_cond_list[0]["uses_start_inpaint"].shape[0]
                d_v = tj_cond_list[0]["uses_start_inpaint"].device
                templ_ov_tj = tj_cond_list[0]["end_overlap_traj"]
                for k_c in keys_tjc:
                    tmp_v_l = []
                    for i_c in range(len_tjc):
                        if k_c in ["start_overlap_is_drop", "end_overlap_is_drop"]:
                            if tj_cond_list[i_c][k_c] is None:
                                tj_cond_list[i_c][k_c] = torch.ones(
                                    size=(b_s,), dtype=torch.bool, device=d_v
                                )
                        if k_c in ["start_overlap_traj", "end_overlap_traj"]:
                            if tj_cond_list[i_c][k_c] is None:
                                tj_cond_list[i_c][k_c] = torch.zeros_like(templ_ov_tj)
                        tmp_v_l.append(tj_cond_list[i_c][k_c])
                    mg_dict[k_c] = torch.cat(tmp_v_l, dim=0)
                return mg_dict

            tj_cond_mg = merge_tj_cond_list(tj_cond_cur_list)
            tj_cond_mg["use_conditioning"] = True
            x_p_list_cur_t = torch.cat(x_p_list_cur_t, dim=0)
            assert len(tj_cond_mg["start_overlap_is_drop"]) == len(x_p_list_cur_t)
            t_steps_mg = torch.cat([timesteps] * num_segments, dim=0)

            if do_mcmc:
                x_p_list_cur_t = self.resample_same_t_mcmc(
                    x_p_list_cur_t, tj_cond_mg, t_steps_mg
                )

            if self.use_ddim:
                x_p_list_cur_t = self.ddim_p_sample(
                    x_p_list_cur_t,
                    tj_cond_mg,
                    t_steps_mg,
                    self.ddim_eta,
                    use_clipped_model_output=True,
                )
            else:
                x_p_list_cur_t = self.p_sample(x_p_list_cur_t, tj_cond_mg, t_steps_mg)

            x_p_list_cur_t = einops.rearrange(
                x_p_list_cur_t,
                "(num_segments B) h d -> num_segments B h d",
                num_segments=num_segments,
            )
            segment_samples = x_p_list_cur_t

            if return_diffusion:
                diffusion_history.append([_ for _ in segment_samples])

        segment_samples[0] = apply_conditioning(
            segment_samples[0], {0: boundary_conditions[0]}, 0
        )
        segment_samples[-1] = apply_conditioning(
            segment_samples[-1], {horizon - 1: boundary_conditions[horizon - 1]}, 0
        )

        if return_diffusion:
            return segment_samples, diffusion_history
        return segment_samples

    @torch.no_grad()
    def comp_pred_p_loop_n_GSC(
        self,
        shape,
        boundary_conditions,
        n_comp,
        do_mcmc=False,
        return_diffusion=False,
    ):
        """Inpaint with Avg Values — compose n trajectories."""
        assert n_comp >= 2 and not do_mcmc
        device = self.schedule.betas.device
        batch_size = shape[0]
        hzn = shape[1]

        x_p_list = [torch.randn(shape, device=device) for _ in range(n_comp)]
        x_dfu_all = [x_p_list]
        assert len(boundary_conditions[0]) == shape[0]

        time_idx = (
            self.ddim_set_timesteps(self.ddim_num_inference_steps)
            if self.use_ddim
            else reversed(range(0, self.n_timesteps))
        )

        for i_t in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), i_t, device=device, dtype=torch.long
            )
            x_p_list_next_t = [None for _ in range(n_comp)]
            x_p_list = self.average_overlap_chunk_global_sync(x_p_list)

            for i_tj in range(n_comp):
                x_p_i = x_p_list[i_tj]

                if i_tj == 0:
                    x_p_i_plus_1 = x_p_list[i_tj + 1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0],
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={0: boundary_conditions[0]},
                    )
                    tj_cond_p_i["end_overlap_is_drop"] = None
                    tj_cond_p_i["use_conditioning"] = False
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(
                            x_p_i,
                            tj_cond_p_i,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    x_p_list_next_t[i_tj] = x_p_i

                elif i_tj > 0 and i_tj < n_comp - 1:
                    tj_cond_p_i = dict(
                        start_overlap_is_drop=None,
                        end_overlap_is_drop=None,
                        uses_start_inpaint=torch.zeros_like(x_p_i[:, 0, 0]).to(
                            torch.bool
                        ),
                        uses_end_inpaint=torch.zeros_like(x_p_i[:, 0, 0]).to(
                            torch.bool
                        ),
                    )
                    tj_cond_p_i["use_conditioning"] = False
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(
                            x_p_i,
                            tj_cond_p_i,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    x_p_list_next_t[i_tj] = x_p_i

                elif i_tj == n_comp - 1:
                    x_p_i_minus_1 = x_p_list[i_tj - 1]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0],
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={hzn - 1: boundary_conditions[hzn - 1]},
                    )
                    tj_cond_p_i["start_overlap_is_drop"] = None
                    tj_cond_p_i["use_conditioning"] = False
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(
                            x_p_i,
                            tj_cond_p_i,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    x_p_list_next_t[i_tj] = x_p_i

            x_p_list = x_p_list_next_t
            if return_diffusion:
                x_dfu_all.append([_ for _ in x_p_list])

        x_p_list[0] = apply_conditioning(x_p_list[0], {0: boundary_conditions[0]}, 0)
        x_p_list[-1] = apply_conditioning(
            x_p_list[-1], {hzn - 1: boundary_conditions[hzn - 1]}, 0
        )

        if return_diffusion:
            return x_p_list, x_dfu_all
        return x_p_list

    @torch.no_grad()
    def comp_pred_p_loop_n_ar_backward(
        self,
        shape,
        boundary_conditions,
        num_segments,
        do_mcmc=False,
        return_diffusion=False,
    ):
        """Generate stitched trajectory segments with backward autoregressive denoising."""
        assert num_segments >= 2 and not do_mcmc
        device = self.schedule.betas.device
        batch_size = shape[0]
        horizon = shape[1]

        segment_samples = [
            torch.randn(shape, device=device) for _ in range(num_segments)
        ]
        diffusion_history = [segment_samples]
        assert len(boundary_conditions[0]) == shape[0]

        time_idx = (
            self.ddim_set_timesteps(self.ddim_num_inference_steps)
            if self.use_ddim
            else reversed(range(0, self.n_timesteps))
        )

        for i_t in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), i_t, device=device, dtype=torch.long
            )

            for i_tj in range(num_segments - 1, -1, -1):
                x_p_i = segment_samples[i_tj]

                if i_tj == 0:
                    x_p_i_plus_1 = segment_samples[i_tj + 1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_p_i_plus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=None,
                        end_traj=st_traj_2,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0] - 1,
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={0: boundary_conditions[0]},
                    )
                    tj_cond_p_i["use_conditioning"] = True
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(
                            x_p_i,
                            tj_cond_p_i,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    segment_samples[i_tj] = x_p_i

                elif i_tj > 0 and i_tj < num_segments - 1:
                    x_p_i_minus_1 = segment_samples[i_tj - 1]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)
                    x_p_i_plus_1 = segment_samples[i_tj + 1]
                    st_traj_i_plus_1, _ = self.extract_ovlp_from_full(x_p_i_plus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=st_traj_i_plus_1,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0] - 1,
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={},
                    )
                    tj_cond_p_i["use_conditioning"] = True
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(
                            x_p_i,
                            tj_cond_p_i,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    segment_samples[i_tj] = x_p_i

                elif i_tj == num_segments - 1:
                    x_p_i_minus_1 = segment_samples[i_tj - 1]
                    _, end_traj_i_minus_1 = self.extract_ovlp_from_full(x_p_i_minus_1)
                    x_p_i, tj_cond_p_i = self.create_eval_conditioning(
                        x_et=x_p_i,
                        st_traj=end_traj_i_minus_1,
                        end_traj=None,
                        t_1d_st=timesteps[:, 0],
                        t_1d_end=timesteps[:, 0],
                        t_type="0",
                        is_noisy=True,
                        boundary_conditions={
                            horizon - 1: boundary_conditions[horizon - 1]
                        },
                    )
                    tj_cond_p_i["use_conditioning"] = True
                    if do_mcmc:
                        x_p_i = self.resample_same_t_mcmc(x_p_i, tj_cond_p_i, timesteps)
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(
                            x_p_i,
                            tj_cond_p_i,
                            timesteps,
                            self.ddim_eta,
                            use_clipped_model_output=True,
                        )
                    else:
                        x_p_i = self.p_sample(x_p_i, tj_cond_p_i, timesteps)
                    segment_samples[i_tj] = x_p_i

            if return_diffusion:
                diffusion_history.append([_ for _ in segment_samples])

        segment_samples[0] = apply_conditioning(
            segment_samples[0], {0: boundary_conditions[0]}, 0
        )
        segment_samples[-1] = apply_conditioning(
            segment_samples[-1], {horizon - 1: boundary_conditions[horizon - 1]}, 0
        )

        if return_diffusion:
            return segment_samples, diffusion_history
        return segment_samples
