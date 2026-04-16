import numpy as np
import torch
import wandb

from comp_diffuser.models.trajectory_stitching.trajectory_stitching_trainer import (
    TrajectoryStitchingTrainer,
)
from comp_diffuser.utils.arrays import batch_copy, batchify, report_parameters
from comp_diffuser.utils.config import Config
from comp_diffuser.utils.setup import ArgsParser as BaseArgsParser

np.set_printoptions(precision=3, suppress=True)
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=4, sci_mode=False)


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class ArgsParser(BaseArgsParser):
    dataset: str = None
    config: str


args = ArgsParser().parse_args("diffusion")


# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

# pdb.set_trace()

dataset_config = Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.tot_horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    ###
    max_n_episodes=getattr(args, "max_n_episodes", 10000),
    ###
    termination_penalty=args.termination_penalty,
    use_padding=args.use_padding,
    ## put a linnk to a smaller dataset for debugging purpose
    dset_h5path=getattr(args, "dset_h5path", None),
    dataset_config=args.dataset_config,
)

render_config = Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim  ## 2
action_dim = dataset.action_dim  ## 2

# test_sample = dataset[0]
# torch.set_printoptions(precision=10, sci_mode=False)
# pdb.set_trace() ## check horizon

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    ##
    device=args.device,
    ##
    horizon=args.sm_horizon,
    transition_dim=observation_dim,
    base_dim=args.base_dim,  ## new
    dim_mults=args.dim_mults,
    time_dim=args.time_dim,  ## new
    network_config=args.network_config,  ## new
)
model = model_config()

# pdb.set_trace()
## model to be input
diffusion_model_config = Config(
    args.diffusion_model,
    savepath=(args.savepath, "diffusion_model.pkl"),
    device=args.device,
    ##
    horizon=args.sm_horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_discount=args.loss_discount,
    loss_weights=args.loss_weights,
    ## ----- Luo -----
    diff_config=args.diff_config,
)

# pdb.set_trace()
## diffusion_model to be input
# comp_diffusion_config = utils.Config(
#     args.comp_diffusion,
#     savepath=(args.savepath, 'comp_diffusion.pkl'),
#     device=args.device,
#     ##
#     tot_horizon=args.tot_horizon,
#     len_overlap=args.len_overlap,
#     loss_type=args.loss_type,
#     tr_time_config=args.tr_time_config,
#     eval_time_config=args.eval_time_config,
#     comp_dfu_config=args.comp_dfu_config,
# )

#############
# pdb.set_trace()

trainer_config = Config(
    TrajectoryStitchingTrainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    ##
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    results_folder=args.savepath,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    trainer_dict=args.trainer_dict,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#


# model = model_config()
# from comp_diffuser.models.conditional_diffusion import Cd_Sml_GauDiffusion_InvDyn_V1
diffusion_model = diffusion_model_config(model=model)

trainer = trainer_config(
    diffusion_model=diffusion_model,
    dataset=dataset,
    renderer=renderer,
    device=args.device,
)

# pdb.set_trace()

# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

report_parameters(model)

print("Testing forward...", end=" ", flush=True)
batch = batchify(dataset[0])  # [1,380,2]
# batch = utils.batchify_seq( [dataset[0], dataset[1]] )
batch = batch_copy(batch, 4)
obs_trajs, act_trajs, stgl_cond = batch
# pdb.set_trace()
##---- can be delete
# for k,v in dict(diffusion_model.named_parameters()).items():
#     if 'diffusion_model' not in k:
#         print(k)
##----
loss, _ = diffusion_model.loss(x_clean=obs_trajs, cond_start_goal=stgl_cond)
loss.backward()
# pdb.set_trace()
print("✓")

# -----------------------------------------------------------------------------#
# --------------------------------- save config ---------------------------------#
# -----------------------------------------------------------------------------#


all_configs = dict(
    dataset_config=dataset_config._dict,
    render_config=render_config._dict,
    model_config=model_config._dict,
    diffusion_model_config=diffusion_model_config._dict,
    trainer_config=trainer_config._dict,
)

# print(args)
ckp_path = args.savepath
wandb.init(
    project="hierarchy-diffuser",
    name=args.logger_name,
    id=args.logger_id,
    dir=ckp_path,
    config=all_configs,  ## need to be a dict
    # resume="must",
    mode="online" if dataset_config.dset_h5path is None else "disabled",
)
# pdb.set_trace()

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=args.n_steps_per_epoch)
