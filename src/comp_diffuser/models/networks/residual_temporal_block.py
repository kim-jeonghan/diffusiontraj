import torch.nn as nn
from einops.layers.torch import Rearrange

from ..common.nn_blocks import Conv1dBlock, Conv1dBlock_dd


class ResidualTemporalBlock(nn.Module):

    def __init__(
        self,
        inp_channels,
        out_channels,
        embed_dim,
        horizon,
        kernel_size=5,
        mish=True,
        conv_zero_init=False,
        resblock_config={},
        **kwargs,
    ):
        super().__init__()
        assert not conv_zero_init
        force_residual_conv = resblock_config.get("force_residual_conv", False)
        time_mlp_config = resblock_config["time_mlp_config"]

        convblock_type = Conv1dBlock_dd

        self.blocks = nn.ModuleList(
            [
                convblock_type(
                    inp_channels, out_channels, kernel_size, mish, conv_zero_init=False
                ),
                convblock_type(
                    out_channels,
                    out_channels,
                    kernel_size,
                    mish,
                    conv_zero_init=conv_zero_init,
                ),
            ]
        )

        act_fn = nn.Mish() if mish else nn.SiLU()
        if time_mlp_config == 2:
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Linear(embed_dim, out_channels * 2),
                act_fn,
                nn.Linear(out_channels * 2, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif time_mlp_config == 3:
            self.time_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                act_fn,
                nn.Linear(embed_dim * 2, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif time_mlp_config == 0:  ## default setting, same as else below
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Linear(embed_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Linear(embed_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )

        if not force_residual_conv:
            self.residual_conv = (
                nn.Conv1d(inp_channels, out_channels, 1)
                if inp_channels != out_channels
                else nn.Identity()
            )
        else:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1)

    def forward(
        self,
        x,
        t,
    ):
        """
        pipeline:
        1. process x only
        2. process t only
        3. process (x + t) *zero init*
        4. process skip connection

        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        w : placeholder
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class HiResidualTemporalBlock(nn.Module):
    """Deeper time MLP variant used by TrajectoryTimeEncoder."""

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
