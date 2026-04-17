import os.path as osp

from comp_diffuser.utils.setup import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args
config_fn = osp.splitext(osp.basename(__file__))[0]

diffusion_args_to_watch = [
    ("prefix", ""),
    ("config_fn", config_fn),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
]


plan_args_to_watch = [
    ("prefix", ""),
    ("config_fn", config_fn),
    ##
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ##
    ("conditional", "cond"),
]

sm_horizon = 40
# len_ovlap = 16
tot_horizon = sm_horizon
time_dim = 64


base = {
    "dataset": "maze2d-umaze-v1",
    "dset_h5path": "data/smoke/maze2d-umaze-sparse-v1-smoke.hdf5",  #####
    "diffusion": {
        "config_fn": "",
        "sm_horizon": sm_horizon,
        "tot_horizon": tot_horizon,
        ##
        ## cnn model
        "model": "models.networks.maze_temporal_unet.MazeTemporalUNet",
        "base_dim": 96,
        "dim_mults": (1, 2, 4, 8),
        "time_dim": time_dim,
        "network_config": dict(
            t_seq_encoder_type="mlp",
            cat_t_w=True,
            resblock_ksize=5,
            time_mlp_config=3,
            ###
            inpaint_token_dim=32,
            inpaint_token_type="const",
        ),
        ## TODO:
        ## dd dfu
        "diffusion_model": "models.diffusion.maze_diffusion.MazeGaussianDiffusion",
        # 'diffusion_model': 'models.GaussianDiffusion',
        "n_diffusion_steps": 512,
        "action_weight": 1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,  ##
        "diff_config": dict(
            infer_deno_type="same",
            obs_manual_loss_weights={},
            w_loss_type="all",
            is_direct_train=True,
            ##
            # len_ovlp_cd=len_ovlap,
            # tr_1side_drop_prob=0.15,
            ## --- NEW ---
            # tr_inpat_prob=0.5,
            # tr_ovlp_prob=0.5,
        ),
        "trainer_dict": dict(),
        "renderer": "rendering.Maze2DRenderer",
        ## dataset
        "loader": "datasets.comp.CompositionalSequenceDataset",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["ben_maze2d_set_terminals"],  ####
        "clip_denoised": True,
        "use_padding": True,
        "max_path_length": 256,
        "max_n_episodes": 51500,
        "dataset_config": dict(
            obs_select_dim=(0, 1),
            dset_type="bens_pm_umaze",
            pad_type="first_last",
            extra_pad=40,
        ),
        ## serialization
        "logbase": "artifacts/runs",
        "prefix": "diffusion/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2_inv_v3",
        # 'loss_type': 'l2_v2',
        "n_train_steps": 2e6,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_freq": 4000,
        "sample_freq": 4000,
        "n_saves": 5,
        "n_reference": 60,
        "n_samples": 10,
        "device": "cuda",
    },
    "plan": {
        "config_fn": "",
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": tot_horizon,
        "n_diffusion_steps": 512,
        "normalizer": "LimitsNormalizer",
        ## serialization
        "vis_freq": 10,
        "logbase": "artifacts/runs",
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        ## loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}",
        "diffusion_epoch": "latest",
    },
}
