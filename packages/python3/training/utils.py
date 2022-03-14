import os
import time
import gin
import yaml
import torch
import numpy as np

import dl
from dl import nest
from dl.rl import set_env_to_eval_mode
from sofa_envs.sofa_recorder import SofaRecorder
from sofa_envs.design_spaces import get_design_space


def unnorm_obs(obs, mean, std):
    def unnorm(item):
        ob, mean, std = item
        if mean is not None:
            return std * ob + mean
        else:
            return ob
    return nest.map_structure(unnorm, nest.zip_structure(obs, mean, std))


def unnorm_actions(actions, action_space):
    low = action_space.low
    high = action_space.high
    return (actions + 1) / 2 * (high - low) + low


def get_action_data(obs):
    leg_keys = sorted([k for k in obs if 'leg' in k])
    actions = np.array([obs[k]['obs'][0][-1] for k in leg_keys])
    return actions


def get_run_id(logdir):
    with open(os.path.join(logdir, 'singularity_config.yaml'), 'r') as f:
        data = yaml.load(f)
    return data['run_id']


def get_alg_name(logdir):
    try:
        return gin.query_parameter('train.algorithm').configurable.name
    except Exception:
        config = os.path.join(logdir, 'config.gin')
        dl.load_config(config)
        alg_name = gin.query_parameter('train.algorithm').configurable.name
        gin.clear_config()
        return alg_name


def get_alg(logdir, bindings):
    config = os.path.join(logdir, 'config.gin')
    dl.load_config(config, bindings)
    alg_configurable = gin.query_parameter('train.algorithm').configurable
    return alg_configurable.fn_or_cls(logdir)


def query_action_freq(logdir):
    config = os.path.join(logdir, 'config.gin')
    dl.load_config(config)
    steps_per_action = gin.query_parameter('sofa_make_env.steps_per_action')
    gin.clear_config()
    return steps_per_action


def query_design_space(logdir):
    config = os.path.join(logdir, 'config.gin')
    dl.load_config(config)
    design_space = gin.query_parameter('sofa_make_env.design_space')
    gin.clear_config()
    return get_design_space(design_space)


class EpisodeRunner():
    def __init__(self, logdir, ckpt, record=False, unreduced=False, bindings=[]):
        self.logdir = logdir
        self.alg_name = get_alg_name(logdir)

        self.steps_per_action = query_action_freq(logdir)
        self.design_space = query_design_space(logdir)

        bindings.extend(['coopt.SAC.nenv=1', 'sofa_make_env.steps_per_action=1'])
        if record:
            bindings.append('sofa_make_env.with_gui=True')
            self.recorder = SofaRecorder()  # This starts an x server if needed.
        else:
            bindings.append('sofa_make_env.with_gui=False')
            self.recorder = None
        if unreduced:
            bindings.append('sofa_make_env.reduction=None')

        self.alg = get_alg(logdir, bindings)
        self.ckpt = self.alg.load(ckpt)

        env = self.alg.sac.env
        self.pi = self.alg.sac.pi
        self.device = self.alg.sac.device
        env.init_scene_with_mode()
        self.env = env.venv.venv
        self.design_space = self.design_space(None, None)

    def run_episode(self):
        set_env_to_eval_mode(self.env)
        self.pi.eval()

        obs = self.env.reset()
        done = False
        reward = 0
        length = 0
        cumulative_rewards = []
        commanded_action = []
        smoothed_action = []
        while not done:
            obs = nest.map_structure(
                lambda x: torch.from_numpy(x).float().to(self.device),
                obs
            )
            with torch.no_grad():
                action = self.pi(obs).action
            action = nest.map_structure(lambda x: x.cpu().numpy(), action)
            for _ in range(self.steps_per_action):
                obs, r, done, _ = self.env.step(action)
                commanded_action.append(action[0])
                unnormed_obs = unnorm_obs(obs, self.env.mean, self.env.std)
                smoothed_action.append(get_action_data(unnormed_obs['observation']))
                design = unnormed_obs['design'][0].astype(np.int32)
                done = done[0]
                reward = reward + r[0]
                cumulative_rewards.append(reward)
                length += 1
                if done:
                    break
        commanded_action = np.array(commanded_action)
        smoothed_action = np.array(smoothed_action)
        if hasattr(self.design_space, 'expand_design_params') and len(design) <= 1:
            design = self.design_space.expand_design_params(int(design))
        return {
            'commanded_action': unnorm_actions(commanded_action, self.design_space.action_space),
            'smoothed_action': smoothed_action,
            'reward': reward,
            'cumulative_rewards': cumulative_rewards,
            'length': length,
            'design': design
        }

    def record_episodes(self, n, outfile, sleep_time=20):
        episode_data = []
        for _ in range(n):
            time.sleep(sleep_time)
            self.recorder.start_recording()
            time.sleep(sleep_time)
            episode_data.append(self.run_episode())
            self.recorder.stop_recording()
            time.sleep(sleep_time)
        self.recorder.merge_recordings(outfile)
        self.recorder.delete_recordings()
        return episode_data
