import copy
import os.path as osp
from datetime import datetime

import torch

from comp_diffuser.models.stitching.trajectory_stitching_planner import (
    TrajectoryStitchingMazePlanner,
)
from comp_diffuser.utils.composition.trajectory_ranking import parse_seeds_str
from comp_diffuser.utils.eval_utils import print_color
from comp_diffuser.utils.serialization import (
    get_checkpoint_step,
    get_latest_epoch,
    list_saved_epochs,
)
from comp_diffuser.utils.setup import ArgsParser as BaseArgsParser

torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)


class ArgsParser(BaseArgsParser):
    dataset: str | None = None
    config: str | None = None
    ## should not put any existing var in config here
    pl_seeds: str = "-1"  # no seed
    # num_segments: int = 2
    plan_n_ep: int = -100  ## all if -100, auto parse to int
    # num_mcmc_steps: int = 10
    # var_temp = 0.5


def main(args_train, args):

    # ---------------------------------- setup ----------------------------------#
    # TODO:
    ld_config = dict(
        # diffusion_model_loadpath="artifacts/runs/maze2d-large-v1/diffusion/m2d_lg_cpV1_Trv2_bs32_Px0_T256/",
    )

    m2d_planner = TrajectoryStitchingMazePlanner(args_train, args=args)
    m2d_planner.setup_load(ld_config=ld_config)

    # pdb.set_trace()

    # ---------------------------- start planning -----------------------------#

    # seeds = None if args.pl_seeds == -1 else list(range(args.pl_seeds))
    # avg_result_dict = m2d_planner.plan_multi_run(seeds, num_ep=args.plan_n_ep)
    ##---
    # given_starts = np.array([[5, 6], [5, 6.5],
    #                          [5, 7], [5, 7.5],
    #                          [5, 8], [5, 8.5]], dtype=np.float32)
    ##---
    pl_seeds = args.pl_seeds
    from comp_diffuser.datasets.d4rl import Is_Gym_Robot_Env

    if len(pl_seeds) == 1:
        ## plan_n_ep
        if Is_Gym_Robot_Env:
            if pl_seeds[0] == -1:  ## no seed
                avg_result_dict = m2d_planner.ben_plan_once_parallel(
                    pl_seed=None,
                )
            else:
                avg_result_dict = m2d_planner.ben_plan_once_parallel(
                    pl_seed=pl_seeds[0]
                )
        else:
            if pl_seeds[0] == -1:  ## no seed
                # avg_result_dict = m2d_planner.plan_once(pl_seed=None,)
                avg_result_dict = m2d_planner.plan_once_parallel(
                    pl_seed=None,
                )
            else:
                # avg_result_dict = m2d_planner.plan_once(pl_seed=pl_seeds[0])
                avg_result_dict = m2d_planner.plan_once_parallel(pl_seed=pl_seeds[0])
    else:
        print_color(f"{args.pl_seeds=}")
        raise NotImplementedError("multi-seed planning is not implemented")

    return avg_result_dict


if __name__ == "__main__":
    ## training args
    args_train = ArgsParser().parse_args("diffusion")
    args = ArgsParser().parse_args("plan")
    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name

    args.pl_seeds = parse_seeds_str(args.pl_seeds)  ## a list of int
    args.n_batch_acc_probs = 10  ## A5000: 20=2.23it/s, 10=4.10it/s

    # pdb.set_trace()
    ### --- Hyper-parameters Setup ---
    from comp_diffuser.datasets.d4rl import Is_Gym_Robot_Env

    if Is_Gym_Robot_Env:  ## Ben
        if "-large-" in args_train.dataset:
            args.num_segments = 5  # ben large
            args.env_n_max_steps = 1000  ## ben large
        elif "-medium-" in args_train.dataset:
            args.num_segments = 5  # ben
            # args.num_segments = 6
            args.env_n_max_steps = 1000  ## ben
        elif "-umaze-" in args_train.dataset:
            args.num_segments = 5  # umaze h is only 40
            # args.num_segments = 4 #
            # args.num_segments = 6 ## test
            args.env_n_max_steps = 1000  #

    else:
        args.num_segments = 4
        args.env_n_max_steps = 600

    args.b_size_per_prob = 20
    args.b_size_per_prob = 40
    args.top_k = 5
    args.trajectory_selection = "first"
    args.blend_type = "exp"
    args.blend_exponential_beta = 2

    args.var_temp = 0.5
    args.cond_w = 2.0
    args.inference_schedule = "gsc"

    # pdb.set_trace()

    latest_e = get_latest_epoch(loadpath)
    available_epochs = list_saved_epochs(loadpath)
    if latest_e < 0:
        raise RuntimeError(
            "No trained trajectory stitching checkpoint found. "
            f"Available epochs at {osp.join(*loadpath)}: {available_epochs}. "
            "Run trajectory stitching training until it saves a checkpoint, "
            "or point planning at the correct trained artifact directory."
        )
    if get_checkpoint_step(loadpath, latest_e) <= 0:
        raise RuntimeError(
            "No trained trajectory stitching checkpoint found. "
            f"Available epochs at {osp.join(*loadpath)}: {available_epochs}. "
            "Run trajectory stitching training until the latest checkpoint "
            "records a positive training step, or point planning at the "
            "correct trained artifact directory."
        )
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e =
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()

    depoch_list = [
        latest_e,
    ]
    # depoch_list = [800000,] # 1M

    # sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S")}-nm{int(args.plan_n_ep)}'
    sub_dir = (
        f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}'
        + f"-nm{int(args.plan_n_ep)}-ncp{args.num_segments}"
        + f"-ems{args.env_n_max_steps}"
        + f"-{args.inference_schedule}"
        f"-evSd{','.join( [str(sd) for sd in args.pl_seeds] )}"
    )
    # pdb.set_trace()
    ## f'-vt{args.var_temp}'
    ## TODO:
    # args.is_vis_single = True
    args.is_vis_single = False

    if args.is_vis_single:
        sub_dir += "-vis"

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main(copy.deepcopy(args_train), copy.deepcopy(args))

        result_list.append(tmp)
