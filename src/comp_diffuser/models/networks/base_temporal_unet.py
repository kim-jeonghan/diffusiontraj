from __future__ import annotations

import einops
import torch
import torch.nn as nn

from ...utils.eval_utils import print_color
from ..common.nn_blocks import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d
from .residual_temporal_block import ResidualTemporalBlock


class BaseTemporalUNet(nn.Module):
    """Shared U-Net backbone for temporal diffusion models.

    Subclasses must implement:
        _build_cond_emb(x, time, cond, force_dropout, half_fd) -> Tensor (B, tot_cond_dim)

    The ``tot_cond_dim`` argument controls the width of the conditioning vector
    fed into every ResidualTemporalBlock; subclasses pass it during ``super().__init__``.
    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        base_dim: int,
        dim_mults: tuple,
        time_dim: int,
        network_config: dict,
        tot_cond_dim: int,
    ) -> None:
        super().__init__()

        dims = [transition_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print_color(f"[ models/BaseTemporalUNet ] Channel dimensions: {in_out}", c="c")

        self.cat_t_w = network_config["cat_t_w"]
        self.resblock_ksize = network_config.get("resblock_ksize", 5)
        self.use_downup_sample = network_config.get("use_downup_sample", True)
        self.network_config = network_config

        assert (
            self.use_downup_sample and self.resblock_ksize == 5
        ), "the default settings"
        assert self.cat_t_w, "only cat_t_w=True is supported"

        mish = True
        act_fn = nn.Mish()
        self.conv_zero_init = False

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim),
        )

        self.last_conv_ksize = network_config.get("last_conv_ksize", 1)
        self.force_residual_conv = network_config.get("force_residual_conv", False)
        self.time_mlp_config = network_config.get("time_mlp_config", False)
        assert self.time_mlp_config == 3
        assert not self.force_residual_conv, "must be False"
        assert self.last_conv_ksize == 1, "1 is from diffuser"

        resblock_config = dict(
            force_residual_conv=self.force_residual_conv,
            time_mlp_config=self.time_mlp_config,
        )

        print_color(f"[TemporalUnet_WCond] {time_dim=}, {tot_cond_dim=}", c="c")
        print(f"[TemporalUnet_WCond]: in_out: {in_out}")

        self.down_times = network_config.get("down_times", 1e5)
        print_color(f"[Unet down_times] {self.down_times}", c="c")

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        def make_res_tb(dim_in, dim_out, tot_cond_dim):
            return ResidualTemporalBlock(
                dim_in,
                dim_out,
                embed_dim=tot_cond_dim,
                horizon=horizon,
                mish=mish,
                conv_zero_init=self.conv_zero_init,
                resblock_config=resblock_config,
                kernel_size=self.resblock_ksize,
            )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times
            self.downs.append(
                nn.ModuleList(
                    [
                        make_res_tb(dim_in, dim_out, tot_cond_dim),
                        make_res_tb(dim_out, dim_out, tot_cond_dim),
                        (
                            Downsample1d(dim_out)
                            if not is_last and self.use_downup_sample
                            else nn.Identity()
                        ),
                    ]
                )
            )
            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = make_res_tb(mid_dim, mid_dim, tot_cond_dim)
        self.mid_block2 = make_res_tb(mid_dim, mid_dim, tot_cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1) or ind < (
                num_resolutions - self.down_times - 1
            )
            self.ups.append(
                nn.ModuleList(
                    [
                        make_res_tb(dim_out * 2, dim_in, tot_cond_dim),
                        make_res_tb(dim_in, dim_in, tot_cond_dim),
                        (
                            Upsample1d(dim_in)
                            if not is_last and self.use_downup_sample
                            else nn.Identity()
                        ),
                    ]
                )
            )
            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(base_dim, base_dim, kernel_size=self.resblock_ksize),
            nn.Conv1d(base_dim, transition_dim, 1),
        )

    def _build_cond_emb(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: dict,
        force_dropout: bool,
        half_fd: bool,
    ) -> torch.Tensor:
        """Return conditioning feature vector (B, tot_cond_dim).

        ``x`` is already rearranged to (B, transition_dim, horizon).
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: dict = {},
        force_dropout: bool = False,
        half_fd: bool = False,
    ) -> torch.Tensor:
        """
        x    : (B, horizon, transition_dim)
        time : (B,)
        """
        x = einops.rearrange(x, "b h t -> b t h")
        cond_emb = self._build_cond_emb(x, time, cond, force_dropout, half_fd)

        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, cond_emb)
            x = resnet2(x, cond_emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_block2(x, cond_emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond_emb)
            x = resnet2(x, cond_emb)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
