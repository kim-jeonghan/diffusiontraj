import numpy as np
import torch
import wandb

from comp_diffuser.models.diffusion.maze_diffusion import MazeGaussianDiffusion
from comp_diffuser.trainers.maze_trainer import MazeTrainer
from comp_diffuser.utils.arrays import (
    batch_copy,
    batchify,
    report_parameters,
    set_device,
)
from comp_diffuser.utils.config import Config
from comp_diffuser.utils.setup import ArgsParser as BaseArgsParser

np.set_printoptions(precision=3, suppress=True)
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=4, sci_mode=False)


class ArgsParser(BaseArgsParser):
    dataset: str | None = None
    config: str


def main():
    args = ArgsParser().parse_args("diffusion")
    set_device(args.device)

    dataset_config = Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        env=args.dataset,
        horizon=args.tot_horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        max_path_length=args.max_path_length,
        max_n_episodes=getattr(args, "max_n_episodes", 10000),
        termination_penalty=args.termination_penalty,
        use_padding=args.use_padding,
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

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    model_config = Config(
        args.model,
        savepath=(args.savepath, "model_config.pkl"),
        device=args.device,
        horizon=args.sm_horizon,
        transition_dim=observation_dim,
        base_dim=args.base_dim,
        dim_mults=args.dim_mults,
        time_dim=args.time_dim,
        network_config=args.network_config,
    )
    model = model_config()

    diffusion_model_config = Config(
        args.diffusion_model,
        savepath=(args.savepath, "diffusion_model.pkl"),
        device=args.device,
        horizon=args.sm_horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=args.loss_discount,
        loss_weights=args.loss_weights,
        diff_config=args.diff_config,
    )

    trainer_config = Config(
        MazeTrainer,
        savepath=(args.savepath, "trainer_config.pkl"),
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

    diffusion_model = diffusion_model_config(model=model)
    if not isinstance(diffusion_model, MazeGaussianDiffusion):
        raise ValueError(
            "`train_maze_baseline.py` requires a baseline Maze config that builds "
            f"`MazeGaussianDiffusion`, but got `{type(diffusion_model).__name__}`. "
            "Do not pass a trajectory stitching config here."
        )

    trainer = trainer_config(
        diffusion_model=diffusion_model,
        dataset=dataset,
        renderer=renderer,
        device=args.device,
    )

    report_parameters(model)

    print("Testing forward...", end=" ", flush=True)
    batch = batchify(dataset[0])
    batch = batch_copy(batch, 4)
    obs_trajs, act_trajs, boundary_conditions = batch
    loss, _ = diffusion_model.loss(
        x_clean=obs_trajs, boundary_conditions=boundary_conditions
    )
    loss.backward()
    print("✓")

    all_configs = dict(
        dataset_config=dataset_config._dict,
        render_config=render_config._dict,
        model_config=model_config._dict,
        diffusion_model_config=diffusion_model_config._dict,
        trainer_config=trainer_config._dict,
    )

    wandb.init(
        project="hierarchy-diffuser",
        name=args.logger_name,
        id=args.logger_id,
        dir=args.savepath,
        config=all_configs,
        mode="online" if dataset_config.dset_h5path is None else "disabled",
    )

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch} / {n_epochs} | {args.savepath}")
        trainer.train(n_train_steps=args.n_steps_per_epoch)


if __name__ == "__main__":
    main()
