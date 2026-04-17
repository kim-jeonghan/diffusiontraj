import torch
from tqdm import tqdm

from ...utils.arrays import batch_repeat_tensor_in_dict
from ...utils.eval_utils import print_color
from ..common.base_diffusion import BaseDiffusion
from ..common.model_outputs import ModelPrediction
from ..helpers import apply_conditioning
from .maze_temporal_unet import MazeTemporalUNet


class MazeGaussianDiffusion(BaseDiffusion):
    """Decision Diffuser baseline used in the CompDiffuser paper."""

    def __init__(
        self,
        model: MazeTemporalUNet,
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
        self.diffusion_name = "dd_maze"

        self.is_direct_train = self.diff_config.get("is_direct_train", False)
        if self.is_direct_train:
            self.setup_sep16()

    def setup_sep16(self):
        self.infer_deno_type = self.diff_config["infer_deno_type"]
        self.w_loss_type = self.diff_config["w_loss_type"]
        self.tr_cond_type = "stgl"
        self.condition_guidance_w = self.diff_config.get("condition_guidance_w", 2.0)
        self.var_temp = 1.0
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

    # ------------------------------------------ sampling ------------------------------------------#

    def p_mean_variance(self, x_t, t_2d, cond, return_modelout=False):
        out_model = self.small_model_pred(x_t, t_2d, cond)

        x_0 = self.schedule.predict_start_from_noise(x_t, t_2d=t_2d, noise=out_model)
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
            x = apply_conditioning(x, g_cond["boundary_conditions"], 0)
            x = self.p_sample(x, cond={}, timesteps=timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(self, g_cond, *args, horizon=None, **kwargs):
        batch_size = len(g_cond["boundary_conditions"][0])
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
        boundary_conditions = g_cond["boundary_conditions"]
        x = apply_conditioning(x, boundary_conditions, 0)

        if return_diffusion:
            diffusion = [x]

        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        for i in tqdm(time_idx):
            timesteps = torch.full(
                (batch_size, self.horizon), i, device=device, dtype=torch.long
            )
            x = apply_conditioning(x, boundary_conditions, 0)
            x = self.ddim_p_sample(
                x,
                cond={},
                timesteps=timesteps,
                eta=self.ddim_eta,
                use_clipped_model_output=True,
            )

            if return_diffusion:
                x = apply_conditioning(x, boundary_conditions, 0)
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    # ------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_0, x_noisy, noise, t_2d, cond, batch_loss_w=None):
        assert self.w_loss_type == "all"
        batch_loss_w = torch.ones_like(x_0[:, :, :1])
        batch_loss_w[:, 0] = 0.0
        batch_loss_w[:, self.horizon - 1] = 0.0
        assert x_0.shape[1] == self.horizon

        pred = self.small_model_pred(x_noisy, t_2d, cond)
        assert x_noisy.shape == pred.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(pred, noise, ext_loss_w=batch_loss_w)
        else:
            loss, info = self.loss_fn(pred, x_0, ext_loss_w=batch_loss_w)

        return loss, info

    def loss(self, x_clean, cond_start_goal):
        assert self.is_direct_train
        batch_size = len(x_clean)
        t_1d = torch.randint(
            0, self.n_timesteps, (batch_size, 1), device=x_clean.device
        ).long()
        t_2d = torch.repeat_interleave(t_1d, repeats=self.horizon, dim=1)

        noise = torch.randn_like(x_clean)
        x_noisy = self.schedule.q_sample(x_0=x_clean, t_2d=t_2d, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond_start_goal, 0)

        return self.p_losses(x_clean, x_noisy, noise, t_2d, cond={})

    def model_predictions(self, x, t_2d, cond: dict):
        if cond["do_cond"]:
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
                x_t=x, t_2d=t_2d, noise=pred_noise
            )
        else:
            x_0 = out_pred
            pred_noise = self.schedule.predict_noise_from_start(x, t_2d, x_0)

        x_0.clamp_(-1.0, 1.0)

        return ModelPrediction(pred_noise, x_0, out_pred)
