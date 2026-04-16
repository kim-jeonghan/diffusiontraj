import copy
import os

import einops
import torch
import wandb

# from comp_diffuser.models.diffusion import GaussianDiffusion
from ..models.diffusion.maze_diffusion import MazeGaussianDiffusionWithInverseDynamics
from ..models.helpers import apply_conditioning
from ..utils.arrays import apply_dict, to_device, to_device_tp, to_np
from ..utils.eval_utils import print_color
from ..utils.timer import Timer
from ..utils.train_utils import get_lr
from ..utils.training import EMA, cycle


class MazeTrainer(object):
    def __init__(
        self,
        diffusion_model: MazeGaussianDiffusionWithInverseDynamics,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        results_folder="./results",
        n_reference=8,
        n_samples=2,
        device="cuda",
        trainer_dict={},
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=train_batch_size,
                num_workers=6,
                shuffle=True,
                pin_memory=True,
            )
        )
        self.dataloader_vis = cycle(
            torch.utils.data.DataLoader(
                self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
            )
        )
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0
        self.device = device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for i_tr in range(n_train_steps):
            for i_ac in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                ## important: check what is inside the batch
                # pdb.set_trace()

                # batch = batch_to_device(batch)
                obs_trajs, act_trajs, boundary_conditions = to_device_tp(
                    *batch, device=self.device
                )

                if self.model.tr_cond_type == "no":
                    boundary_conditions = {}
                # loss, infos = self.model.loss(*batch)
                loss, infos = self.model.loss(
                    x_clean=obs_trajs, cond_start_goal=boundary_conditions
                )

                # pdb.set_trace()

                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                print(f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}")

                # pdb.set_trace()
                ## save to online
                metrics = {k: v.detach().item() for k, v in infos.items()}

                metrics["train/it"] = self.step
                metrics["train/loss"] = loss.detach().item()
                metrics["train/lr"] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.ema_model.eval()

                # self.render_samples(n_samples=self.n_samples, conditioning_mode=False)
                self.render_samples(
                    n_samples=self.n_samples, conditioning_mode="boundary_only"
                )

                self.ema_model.train()
                if self.step > 5e5:  ## less sampling, faster training
                    self.sample_freq = 30000

            self.step += 1

    def save(self, epoch):
        """
        saves model and ema to disk;
        """
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f"state_{epoch}.pt")
        torch.save(data, savepath)
        print_color(f"[ utils/training ] Saved model to {savepath}", c="y")

    def load4resume(self, loadpath):
        ## Dec 26
        data = torch.load(loadpath)
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
        self.step = data["step"]

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.logdir, f"state_{epoch}.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    # -----------------------------------------------------------------------------#
    # --------------------------------- rendering ---------------------------------#
    # -----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        """
        renders training points
        """

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        obs_trajs = to_np(batch.obs_trajs)

        ## [ batch_size x horizon x observation_dim (4 pos+vel) ]
        normed_observations = obs_trajs  # [:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(
            normed_observations, "observations"
        )

        observations = self.get_rowcol_obs_trajs(observations)

        savepath = os.path.join(self.logdir, "_sample-reference.png")

        is_non_keypt = None  # self.model.get_is_non_keypt(batch_size, None).numpy()

        self.renderer.composite(savepath, observations, is_non_keypt=is_non_keypt)

    def render_samples(self, batch_size=1, n_samples=2, conditioning_mode=None):
        """
        renders samples from (ema) diffusion model
        """
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            boundary_conditions = to_device(batch.conditions, "cuda:0")

            ## B,2
            ## repeat each item in conditions `n_samples` times
            boundary_conditions = apply_dict(
                einops.repeat,
                boundary_conditions,
                "b d -> (repeat b) d",
                repeat=n_samples,
            )

            # pdb.set_trace()

            ## old: [ n_samples x horizon x (action_dim + observation_dim) ]

            if conditioning_mode == "boundary_only":

                traj_full = einops.repeat(
                    batch.obs_trajs, "b h d -> (repeat b) h d", repeat=n_samples
                ).to("cuda:0")

                g_cond = dict(
                    conditioning_mode="boundary_only",
                    traj_full=traj_full,
                    t_type="rand",
                    boundary_conditions=boundary_conditions,
                )
                samples = self.ema_model.conditional_sample(g_cond=g_cond)
                samples = apply_conditioning(samples, boundary_conditions, 0)

            # elif do_cond in [None, False]:
            # samples = self.ema_model.sample_unCond(batch_size=len(boundary_conditions[0]))

            else:
                raise NotImplementedError

            ## (10, 380, 2)
            samples = to_np(samples)

            ##
            act_dim = 0 if self.model.uses_inverse_dynamics else self.dataset.action_dim
            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, act_dim:]
            # pdb.set_trace()

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:, None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            observations = self.get_rowcol_obs_trajs(observations)
            ##########

            sample_savedir = self.get_sample_savedir(self.step)
            self.debug_mode = False

            if self.debug_mode:
                sample_savedir = os.path.join(self.logdir, "debug-vis")
                if not os.path.isdir(sample_savedir):
                    os.makedirs(sample_savedir)

            savepath = os.path.join(
                sample_savedir, f"sample-{self.step}-{i}-{conditioning_mode}.png"
            )

            # is_non_keypt = self.model.get_is_non_keypt(n_samples, None).numpy()
            is_non_keypt = None

            # pdb.set_trace()

            is_cond = conditioning_mode not in [None, False]
            if is_cond:
                traj_full_unnorm = self.dataset.normalizer.unnormalize(
                    to_np(traj_full), "observations"
                )
                ## convert from Ben's format
                traj_full_unnorm = self.get_rowcol_obs_trajs(traj_full_unnorm)

                self.renderer.composite(
                    savepath,
                    observations,
                    is_non_keypt=is_non_keypt,
                )
            else:
                self.renderer.composite(
                    savepath, observations, is_non_keypt=is_non_keypt
                )

    def get_sample_savedir(self, i):
        div_freq = 100000
        subdir = str((i // div_freq) * div_freq)
        sample_savedir = os.path.join(self.logdir, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir

    def get_rowcol_obs_trajs(self, obs_trajs):
        ## special handling for xy in ben's dataset
        dset_type = self.dataset.dset_type
        if dset_type != "ours":
            assert "ben" in dset_type.lower()
            obs_trajs = utils.ben_xy_to_luo_rowcol(dset_type, obs_trajs)

        return obs_trajs
