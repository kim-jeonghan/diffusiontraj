import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.eval_utils import print_color


class MLPInvDyn(nn.Module):
    """MLP inverse dynamics model."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """hidden_dim (list): e.g. [512, 256, 128]"""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layer_dim = [input_dim] + hidden_dim + [output_dim]
        num_layer = len(layer_dim) - 1
        module_list = []
        for i_l in range(num_layer):
            module_list.append(nn.Linear(layer_dim[i_l], layer_dim[i_l + 1]))
            module_list.append(nn.ReLU())
        del module_list[-1]  # no relu at last

        self.encoder = nn.Sequential(*module_list)
        print_color(f"[MLPInvDyn]  {num_layer=}, {layer_dim=}")

    def forward(self, x):
        return self.encoder(x)


class MLPInvDynV2(nn.Module):
    """MLP inverse dynamics model with configurable activation and dropout."""

    def __init__(self, input_dim, output_dim, inv_m_config):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dim = inv_m_config["inv_hid_dims"]
        layer_dim = [input_dim] + hidden_dim + [output_dim]
        num_layer = len(layer_dim) - 1

        act_f = inv_m_config["act_f"]
        if act_f == "relu":
            act_fn_cls = nn.ReLU
        elif act_f == "Prelu":
            act_fn_cls = nn.PReLU
        else:
            raise ValueError(f"unsupported act_f: {act_f}")

        module_list = []
        for i_l in range(num_layer):
            module_list.append(nn.Linear(layer_dim[i_l], layer_dim[i_l + 1]))
            module_list.append(act_fn_cls())
            if inv_m_config["use_dpout"] and i_l < num_layer - 2:
                module_list.append(nn.Dropout(p=inv_m_config["prob_dpout"]))
        del module_list[-1]  # no activation at last

        self.encoder = nn.Sequential(*module_list)
        print_color(f"[MLPInvDynV2]  {num_layer=}, {layer_dim=}")

    def forward(self, x):
        return self.encoder(x)

    def loss(self, x_t, x_t_1, a_t):
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        pred_a_t = self.forward(x_comb_t)
        inv_loss = F.mse_loss(pred_a_t, a_t)
        return inv_loss, {}
