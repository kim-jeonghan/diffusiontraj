import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def zero_module(module, do_zero):
    if do_zero:
        for p in module.parameters():
            p.data.fill_(0)
    return module


class Conv1dBlock_dd(nn.Module):
    """Conv1d --> GroupNorm --> (Mish / SiLU)"""

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        mish=True,
        n_groups=8,
        conv_zero_init=False,
    ):
        super().__init__()
        act_fn = nn.Mish() if mish else nn.SiLU()
        self.block = nn.Sequential(
            zero_module(
                nn.Conv1d(
                    inp_channels, out_channels, kernel_size, padding=kernel_size // 2
                ),
                conv_zero_init,
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb2D(nn.Module):
    """Sinusoidal positional embedding for 2-D inputs (B, H)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.ndim == 2
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
