import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.arrays import to_np


class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (
            loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
        ).mean()
        return weighted_loss, {"a0_loss": a0_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()
        corr = (
            np.corrcoef(to_np(pred).squeeze(), to_np(targ).squeeze())[0, 1]
            if len(pred) > 1
            else np.NaN
        )
        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": corr,
        }
        return loss, info


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedLoss_L2_V2(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred, targ, ext_loss_w=1.0):
        loss = ext_loss_w * self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (
            loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
        ).mean()
        return weighted_loss, {"a0_loss": a0_loss}

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedLoss_L2_InvDyn_V3(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ, ext_loss_w=1.0):
        loss = ext_loss_w * self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {}

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedStateLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {"a0_loss": weighted_loss}


class WeightedStateL2(WeightedStateLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "l2_v2": WeightedLoss_L2_V2,
    "l2_inv_v3": WeightedLoss_L2_InvDyn_V3,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
    "state_l2": WeightedStateL2,
}
