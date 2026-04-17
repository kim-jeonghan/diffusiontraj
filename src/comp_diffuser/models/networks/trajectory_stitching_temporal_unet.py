import torch
import torch.nn as nn

from ...utils.eval_utils import print_color
from .base_temporal_unet import BaseTemporalUNet
from .trajectory_time_encoder import TrajectoryTimeEncoder


class StitchingTemporalUNet(BaseTemporalUNet):
    """CompDiffuser UNet — additionally conditions on overlap trajectories and inpaint tokens."""

    def __init__(
        self,
        horizon,
        transition_dim,
        base_dim=32,
        dim_mults=(1, 2, 4, 8),
        time_dim=32,
        network_config={},
    ):
        self.st_ovlp_model_config = network_config["st_ovlp_model_config"]
        self.end_ovlp_model_config = network_config["end_ovlp_model_config"]
        self.ext_cond_dim = network_config["ext_cond_dim"]
        self.inpaint_token_dim = network_config["inpaint_token_dim"]
        self.inpaint_token_type = network_config["inpaint_token_type"]

        assert self.inpaint_token_type == "const"

        tot_cond_dim = time_dim + self.ext_cond_dim + 2 * self.inpaint_token_dim
        super().__init__(
            horizon=horizon,
            transition_dim=transition_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_dim=time_dim,
            network_config=network_config,
            tot_cond_dim=tot_cond_dim,
        )

        ovlp_model_type = network_config.get("ovlp_model_type", "unet")
        assert (
            ovlp_model_type == "unet"
        ), f"unsupported ovlp_model_type: {ovlp_model_type}"

        self.st_ovlp_model = TrajectoryTimeEncoder(**self.st_ovlp_model_config)
        self.end_ovlp_model = TrajectoryTimeEncoder(**self.end_ovlp_model_config)

        wall_embed_dim = self.st_ovlp_model.out_dim + self.end_ovlp_model.out_dim
        assert wall_embed_dim == self.ext_cond_dim

        self.st_inpaint_model = nn.Identity()
        self.end_inpaint_model = nn.Identity()

        self.register_buffer(
            "st_use_inpaint_token",
            torch.full((1, self.inpaint_token_dim), 1.0),
        )
        self.register_buffer(
            "st_no_inpaint_token",
            torch.full((1, self.inpaint_token_dim), 0.0),
        )
        self.register_buffer(
            "end_use_inpaint_token",
            torch.full((1, self.inpaint_token_dim), 1.0),
        )
        self.register_buffer(
            "end_no_inpaint_token",
            torch.full((1, self.inpaint_token_dim), 0.0),
        )

        print_color(
            f"[StitchingTemporalUNet] {tot_cond_dim=}, {self.ext_cond_dim=}", c="c"
        )

    def _build_cond_emb(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: dict,
        force_dropout: bool,
        half_fd: bool,
    ) -> torch.Tensor:
        b_size = x.shape[0]
        is_st_inpat = cond.get("is_st_inpat", cond.get("uses_start_inpaint"))
        is_end_inpat = cond.get("is_end_inpat", cond.get("uses_end_inpaint"))
        assert (
            is_st_inpat.shape[0] == b_size
            and is_st_inpat.ndim == 1
            and is_st_inpat.dtype == torch.bool
        )
        assert (
            is_end_inpat.shape[0] == b_size
            and is_end_inpat.ndim == 1
            and is_end_inpat.dtype == torch.bool
        )

        cond_emb = self.time_mlp(time)

        st_ovlp_is_drop = cond.get("st_ovlp_is_drop", cond.get("start_overlap_is_drop"))
        end_ovlp_is_drop = cond.get("end_ovlp_is_drop", cond.get("end_overlap_is_drop"))

        if st_ovlp_is_drop is not None:
            st_ovlp_feat = self.st_ovlp_model(
                cond.get("st_ovlp_traj", cond.get("start_overlap_traj")),
                time=cond.get("st_ovlp_t", cond.get("start_overlap_t")),
            )
            assert len(st_ovlp_is_drop) == len(st_ovlp_feat)
            assert st_ovlp_is_drop.dtype == torch.bool
            st_ovlp_feat[st_ovlp_is_drop] = 0.0
            assert not torch.logical_and(~st_ovlp_is_drop, is_st_inpat).any()
        else:
            st_ovlp_feat = torch.zeros(
                (b_size, self.st_ovlp_model.out_dim), device=x.device
            )

        if end_ovlp_is_drop is not None:
            end_ovlp_feat = self.end_ovlp_model(
                cond.get("end_ovlp_traj", cond.get("end_overlap_traj")),
                time=cond.get("end_ovlp_t", cond.get("end_overlap_t")),
            )
            end_ovlp_feat[end_ovlp_is_drop] = 0.0
            assert end_ovlp_is_drop.dtype == torch.bool
            assert not torch.logical_and(~end_ovlp_is_drop, is_end_inpat).any()
        else:
            end_ovlp_feat = torch.zeros(
                (b_size, self.end_ovlp_model.out_dim), device=x.device
            )

        st_token = torch.zeros(
            (b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device
        )
        num_st_inpt = torch.sum(is_st_inpat).item()
        st_token[is_st_inpat] = self.st_use_inpaint_token.repeat((num_st_inpt, 1))
        st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat(
            (b_size - num_st_inpt, 1)
        )

        end_token = torch.zeros(
            (b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device
        )
        num_end_inpt = torch.sum(is_end_inpat).item()
        end_token[is_end_inpat] = self.end_use_inpaint_token.repeat((num_end_inpt, 1))
        end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat(
            (b_size - num_end_inpt, 1)
        )

        st_token = self.st_inpaint_model(st_token)
        end_token = self.end_inpaint_model(end_token)

        if force_dropout:
            assert not self.training
            assert half_fd, "only half_fd dropout is supported"
            b_s = len(st_ovlp_feat)
            assert b_s % 2 == 0
            st_ovlp_feat[b_s // 2 :] = 0.0
            end_ovlp_feat[b_s // 2 :] = 0.0

        return torch.cat(
            [cond_emb, st_ovlp_feat, end_ovlp_feat, st_token, end_token], dim=-1
        )
