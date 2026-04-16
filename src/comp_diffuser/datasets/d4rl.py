import collections
import os
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

import gym
import h5py
import numpy as np


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

_CONDA_ENV_NAME = os.getenv('CONDA_DEFAULT_ENV', '').split('/')[-1]
Is_Gym_Robot_Env = _CONDA_ENV_NAME in ['hi_diffuser_ben',]
with suppress_output():
    ## d4rl prints out a variety of warnings
    if Is_Gym_Robot_Env:
        ## NOTE: [Only in Eval for Maze2D Ben] 
        ## load from gymnasium
        import gymnasium as gym_na
    else:
        pass

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    if Is_Gym_Robot_Env and 'maze2d' in name:
        return load_env_gym_robo(name)
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    ## setup the name for choosing env specific value from dict later
    if 'hopper' in name or 'halfcheetah' in name or 'walker' in name:
        ## do not support this envs
        from comp_diffuser.locomo.loco_misc import get_loco_short_env_name
        env.short_name = get_loco_short_env_name(name)
    
    return env

def load_env_gym_robo(name_ori):
    """
    load a gym robotic env, for Ben
    """
    if type(name_ori) != str:
        ## name is already an environment
        return name_ori
    
    from comp_diffuser.datasets.gym_robo_utils import get_gym_robo_env_name
    ## get the gym robot version env name to load
    e_name = get_gym_robo_env_name(name_ori)
    with suppress_output():
        is_cont_tk = 'maze' not in e_name.lower() ## False for Maze
        wrapped_env = gym_na.make(
            e_name, continuing_task=is_cont_tk, render_mode="rgb_array")

    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = e_name
    ## important: no noise when reset, noise already added in prob
    # env.position_noise_range = 0.0 ## no need now, we have reset_given
    from comp_diffuser.datasets.comp.maze2d_constants import get_str_maze_spec
    env.str_maze_spec = get_str_maze_spec(e_name)

    return env





def get_dataset(env):
    # dataset = env.get_dataset()
    ## important: Luo update
    h5path = getattr(env, 'dset_h5path', None)
    from comp_diffuser.utils import print_color
    if h5path is not None:
        tmp_str = '!' * 50
        print_color(f'\n{tmp_str}\n')
        print_color(f'LuoTest: \n [Loading from LuoTest] {h5path}')
        print_color(f'\n{tmp_str}\n')
    print_color(f'd4rl.get_dataset: [Loading from LuoTest given h5path] {h5path}', c='y')
    dataset_url = getattr(env, '_dataset_url', None)
    print_color(f'd4rl.get_dataset: {dataset_url=}')

    if hasattr(env, 'get_dataset'):
        dataset = env.get_dataset(h5path=h5path)
    elif h5path is not None:
        dataset = load_h5_dataset(h5path)
    else:
        raise AttributeError('Environment does not provide get_dataset() and no h5path was set')

    return dataset


def load_h5_dataset(h5path):
    def _read_group(group, prefix=''):
        out = {}
        for key, value in group.items():
            name = f'{prefix}{key}'
            if isinstance(value, h5py.Dataset):
                out[name] = value[()]
            else:
                out.update(_read_group(value, prefix=f'{name}/'))
        return out

    with h5py.File(h5path, 'r') as dataset_file:
        return _read_group(dataset_file)






def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            
            if 'maze2d' in env.name and env.proc_m2d_ep:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
