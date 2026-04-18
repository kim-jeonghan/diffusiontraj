import os
import random
import warnings

import h5py
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from tap import Tap

from comp_diffuser.data_generation.maze2d_constants import m2d_get_bottom_top_rows
from comp_diffuser.data_generation.maze2d_problem_utils import (
    m2d_rand_sample_probs,
    merge_prob_dicts,
)
from comp_diffuser.utils.eval_utils import print_color

warnings.simplefilter("always", ResourceWarning)


class ArgsParser(Tap):
    sub_conf: str = "config.maze2d"


def main():
    args = ArgsParser().parse_args()

    file_path = "configs/data_generation/maze2d_problem_sets.yaml"
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
        rs_cfg = config[args.sub_conf]

    seed_u = rs_cfg["seed_u"]
    random.seed(seed_u)
    np.random.seed(seed_u)
    torch.manual_seed(seed_u)

    el_name = rs_cfg["el_name"]
    assert el_name == "maze2d-large-v1", "not implement others yet"

    print_color(f"el_name: {el_name}")
    prob_dicts = []
    if rs_cfg["prob_type"] == "bottom_top_2way":
        bottom_row, top_row = m2d_get_bottom_top_rows(el_name)
        st_p_l_1 = bottom_row
        gl_p_l_1 = top_row
        prob_dicts.append(m2d_rand_sample_probs(st_p_l_1, gl_p_l_1, rs_cfg))

        st_p_l_2 = top_row
        gl_p_l_2 = bottom_row
        prob_dicts.append(m2d_rand_sample_probs(st_p_l_2, gl_p_l_2, rs_cfg))

        all_prob_dict = merge_prob_dicts(prob_dicts)
    else:
        raise NotImplementedError

    h5_root = "data/m2d/ev_probs"
    h5_save_path = f"{h5_root}/{args.sub_conf}.hdf5"

    with h5py.File(h5_save_path, "w") as file:
        file.create_dataset("start_state", data=all_prob_dict["start_state"])
        file.create_dataset("goal_pos", data=all_prob_dict["goal_pos"])
        file.attrs["env_seed"] = seed_u
        file.attrs["env_name"] = el_name

    os.chmod(h5_save_path, 0o444)
    print_color(f"[save to] {h5_save_path=}")


if __name__ == "__main__":
    main()
