import torch

from comp_diffuser.models.common.schedule import DiffusionSchedule


def test_diffusion_schedule_buffers_move_with_module():
    schedule = DiffusionSchedule(n_timesteps=10)
    assert schedule.betas.device.type == "cpu"
    assert schedule.betas.shape == (10,)
    assert schedule.alphas_cumprod.shape == (10,)
    assert schedule.posterior_variance.shape == (10,)


def test_diffusion_schedule_q_sample_shape():
    schedule = DiffusionSchedule(n_timesteps=10)
    B, H, D = 2, 4, 6
    x_start = torch.randn(B, H, D)
    t_2d = torch.randint(0, 10, (B, H))
    x_noisy = schedule.q_sample(x_start, t_2d)
    assert x_noisy.shape == x_start.shape


def test_diffusion_schedule_predict_start_from_noise_roundtrip():
    schedule = DiffusionSchedule(n_timesteps=10)
    B, H, D = 2, 4, 6
    x_start = torch.randn(B, H, D)
    noise = torch.randn_like(x_start)
    t_2d = torch.randint(0, 10, (B, H))

    x_noisy = schedule.q_sample(x_start, t_2d, noise)
    x_recon = schedule.predict_start_from_noise(
        x_noisy, t_2d, noise, predict_epsilon=True
    )
    assert torch.allclose(x_recon, x_start, atol=1e-4)


def test_diffusion_schedule_q_posterior_shape():
    schedule = DiffusionSchedule(n_timesteps=10)
    B, H, D = 2, 4, 6
    x_start = torch.randn(B, H, D)
    x_t = torch.randn(B, H, D)
    t_2d = torch.randint(0, 10, (B, H))

    mean, var, log_var = schedule.q_posterior(x_0=x_start, x_t=x_t, t=t_2d)
    assert mean.shape == (B, H, D)
    assert var.shape == (B, H, 1)
    assert log_var.shape == (B, H, 1)
