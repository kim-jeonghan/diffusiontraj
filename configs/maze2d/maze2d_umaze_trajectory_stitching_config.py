import os.path as osp

from comp_diffuser.utils.setup import watch

# ------------------------ base ------------------------#

config_fn = osp.splitext(osp.basename(__file__))[0]

diffusion_args_to_watch = [
    ("config_fn", config_fn),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
]

plan_args_to_watch = [
    ("config_fn", config_fn),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ("conditional", "cond"),
]

sm_horizon = 40
len_overlap = 16
tot_horizon = sm_horizon
time_dim = 64

ovlp_out_dim = 128
ovlp_model_config = dict(
    c_traj_hzn=len_overlap,
    in_dim=2,
    base_dim=32,
    dim_mults=(1, 2, 3, 4),
    time_dim=32,
    out_dim=ovlp_out_dim,
    tjti_enc_config=dict(
        t_seq_encoder_type="mlp",
        cnn_out_dim=128,
        final_mlp_dims=[640, 256, ovlp_out_dim],
        f_conv_ks=3,
    ),
)

base = {
    "dataset": "maze2d-umaze-v1",
    "dset_h5path": "data/m2d/maze2d-umaze-sparse-v1-smoke.hdf5",  #####
    "diffusion": {
        "config_fn": "",
        "sm_horizon": sm_horizon,
        "tot_horizon": tot_horizon,
        "model": "models.networks.trajectory_stitching_temporal_unet.StitchingTemporalUNet",
        "base_dim": 96,
        "dim_mults": (1, 2, 4, 8),
        "time_dim": time_dim,
        "network_config": dict(
            t_seq_encoder_type="mlp",
            cat_t_w=True,
            resblock_ksize=5,
            st_ovlp_model_config=ovlp_model_config,
            end_ovlp_model_config=ovlp_model_config,
            ext_cond_dim=2 * ovlp_out_dim,
            time_mlp_config=3,
            inpaint_token_dim=32,
            inpaint_token_type="const",
        ),
        "diffusion_model": "models.diffusion.trajectory_stitching_diffusion.StitchingDiffusion",
        "n_diffusion_steps": 512,
        "action_weight": 1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "diff_config": dict(
            infer_deno_type="same",
            obs_manual_loss_weights={},
            w_loss_type="all",
            is_direct_train=True,
            len_ovlp_cd=len_overlap,
            tr_1side_drop_prob=0.20,
            tr_inpat_prob=0.5,
            tr_ovlp_prob=0.5,
            tr_no_ovlp_none=False,
        ),
        "trainer_dict": dict(),
        "renderer": "rendering.Maze2DRenderer",
        "loader": "datasets.comp.CompositionalSequenceDataset",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["ben_maze2d_set_terminals"],
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
        "logbase": "artifacts",
        "prefix": "",
        "exp_name": watch(diffusion_args_to_watch),
        "n_steps_per_epoch": 10000,
        "loss_type": "l2_inv_v3",
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
        "horizon": tot_horizon,
        "n_diffusion_steps": 512,
        "normalizer": "LimitsNormalizer",
        "vis_freq": 10,
        "logbase": "artifacts",
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}",
        "diffusion_epoch": "latest",
    },
}
