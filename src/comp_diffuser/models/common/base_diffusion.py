from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .helpers import extract_2d
from .losses import Losses
from .schedule import DiffusionSchedule


class BaseDiffusion(nn.Module):
    """Shared base for maze diffusion models.

    Subclasses must implement:
        p_mean_variance(self, x_t, t_2d, cond, return_modelout=False)
        model_predictions(self, x, t_2d, cond) -> ModelPrediction
        p_losses(self, x_0, x_noisy, noise, t_2d, cond) -> (loss, info)
        loss(self, x_clean, boundary_conditions) -> (loss, info)
        p_sample_loop(self, shape, g_cond, return_diffusion=False)
        ddim_p_sample_loop(self, shape, g_cond, return_diffusion=False)
        conditional_sample(self, g_cond, *args, horizon=None, **kwargs)
        sample_unCond(self, batch_size, *args, horizon=None, **kwargs)
    """

    uses_inverse_dynamics: bool = True
    use_eta_noise: bool = False
    var_temp: float = 1.0
    clip_noise: float = 20.0

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        loss_type: str = "l1",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights=None,
        diff_config: dict = {},
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.schedule = DiffusionSchedule(n_timesteps)
        self.n_timesteps = self.schedule.n_timesteps
        self.predict_epsilon = predict_epsilon

        self.obs_manual_loss_weights = diff_config["obs_manual_loss_weights"]
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights)
        assert self.obs_manual_loss_weights == {}

        self.diff_config = diff_config

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def get_loss_weights(self, action_weight, discount, weights_dict):
        assert discount == 1 and weights_dict is None

        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[ind] *= w

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        assert (loss_weights == 1).all()

        if len(self.obs_manual_loss_weights) > 0:
            for k, v in self.obs_manual_loss_weights.items():
                loss_weights[k, :] = v
                print(f"[set manual loss weight] {k} {v}")

        return loss_weights

    def small_model_pred(self, x, t_2d, cond: dict) -> torch.Tensor:
        assert (t_2d[0] == t_2d[0, 0]).all(), "sanity check"
        t_1d = t_2d[:, 0]
        return self.model(x, t_1d, cond)

    def ddim_set_timesteps(self, num_inference_steps: int) -> np.ndarray:
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        return timesteps

    # ------------------------------------------------------------------
    # Sampling primitives (cond param unifies tj_cond / segment_conditioning)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, cond: dict, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Single DDPM reverse step. timesteps: (B, H)."""
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t=x, t_2d=timesteps, cond=cond
        )
        noise = self.var_temp * torch.randn_like(x)
        nonzero_mask = (1 - (timesteps == 0).float()).reshape(
            b, self.horizon, *((1,) * (len(x.shape) - 2))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def ddim_p_sample(
        self,
        x: torch.Tensor,
        cond: dict,
        timesteps: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
    ) -> torch.Tensor:
        """Single DDIM reverse step. timesteps: (B, H), all values equal."""
        prev_timestep = (
            timesteps - self.num_train_timesteps // self.ddim_num_inference_steps
        )
        alpha_prod_t = extract_2d(self.schedule.alphas_cumprod, timesteps, x.shape)

        assert torch.isclose(prev_timestep[0], prev_timestep[0, 0]).all()
        if prev_timestep[0, 0] >= 0:
            alpha_prod_t_prev = extract_2d(
                self.schedule.alphas_cumprod, prev_timestep, x.shape
            )
        else:
            alpha_prod_t_prev = extract_2d(
                self.final_alpha_cumprod.to(timesteps.device),
                torch.zeros_like(timesteps),
                x.shape,
            )
        assert alpha_prod_t.shape == alpha_prod_t_prev.shape

        beta_prod_t = 1 - alpha_prod_t

        model_mean, _, model_log_variance, x_0, model_output = self.p_mean_variance(
            x_t=x, t_2d=timesteps, cond=cond, return_modelout=True
        )

        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        std_dev_t = eta * variance**0.5

        assert use_clipped_model_output
        pred_original_sample = x_0
        model_output = (x - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

        pred_sample_direction = (
            1 - alpha_prod_t_prev - std_dev_t**2
        ) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        if self.use_eta_noise and eta > 0:
            prev_sample = prev_sample + std_dev_t * torch.randn_like(model_output)

        return prev_sample

    @torch.no_grad()
    def resample_same_t_mcmc(
        self,
        x: torch.Tensor,
        cond: dict,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x_t_mc = torch.clone(x)
        for _ in range(self.eval_n_mcmc):
            model_pred = self.model_predictions(x=x_t_mc, t_2d=timesteps, cond=cond)
            recon_x0 = model_pred.pred_x_0
            recon_x0.clamp_(-1.0, 1.0)
            x_t_mc = self.schedule.q_sample(x_0=recon_x0, t_2d=timesteps)
        return x_t_mc
