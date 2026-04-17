from __future__ import annotations

import torch
from torch import nn

from ..helpers import cosine_beta_schedule, extract_2d


class DiffusionSchedule(nn.Module):
    """Diffusion noise schedule with precomputed buffers.

    Encapsulates all schedule tensors (betas, alphas, cumprod, posterior
    coefficients) so they are registered as nn.Module buffers and move
    automatically with .to(device).
    """

    def __init__(self, n_timesteps: int = 1000) -> None:
        super().__init__()

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.n_timesteps = int(n_timesteps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t_2d: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add noise to x_0 at timesteps t_2d: q(x_t | x_0)."""
        assert t_2d.ndim == 2
        if noise is None:
            noise = torch.randn_like(x_0)
        q_coef1 = extract_2d(self.sqrt_alphas_cumprod, t_2d, x_0.shape)
        q_coef2 = extract_2d(self.sqrt_one_minus_alphas_cumprod, t_2d, x_0.shape)
        return q_coef1 * x_0 + q_coef2 * noise

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance: q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract_2d(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract_2d(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_2d(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_2d(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t_2d: torch.Tensor,
        noise: torch.Tensor,
        predict_epsilon: bool,
    ) -> torch.Tensor:
        """Recover x_0 from noisy x_t and predicted noise (or direct x_0 prediction)."""
        if predict_epsilon:
            return (
                extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t
                - extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape) * noise
            )
        return noise

    def predict_noise_from_start(
        self,
        x_t: torch.Tensor,
        t_2d: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Invert x_0 prediction back to noise estimate."""
        return (
            extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t - x_0
        ) / extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape)
