import torch

from .base_temporal_unet import BaseTemporalUNet


class MazeTemporalUNet(BaseTemporalUNet):
    """Decision Diffuser baseline UNet — conditions only on diffusion timestep."""

    def __init__(
        self,
        horizon,
        transition_dim,
        base_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_dim=32,
        network_config={},
    ):
        tot_cond_dim = time_dim
        super().__init__(
            horizon=horizon,
            transition_dim=transition_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_dim=time_dim,
            network_config=network_config,
            tot_cond_dim=tot_cond_dim,
        )

    def _build_cond_emb(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: dict,
        force_dropout: bool,
        half_fd: bool,
    ) -> torch.Tensor:
        return self.time_mlp(time)
