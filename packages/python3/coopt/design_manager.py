import numpy as np
import gym
from collections import deque

from dl import nest
from dl.rl import VecEnvWrapper
from dl import CatDist
from sofa_envs.design_spaces import get_design_space
import torch
import gin
import wandb


class DesignLogger:
    """Records and keeps the history of designs and rewards."""

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.designs = deque([], maxlen=maxlen)
        self.rewards = deque([], maxlen=maxlen)
        self.count = 0
        self.columns = ['count', 'design', 'reward']
        self.data = []
        wandb.define_metric("train/design_count", step_metric="train/step")
        wandb.define_metric("designs", step_metric="train/design_count")

    def log(self, design, reward):
        self.designs.append(design)
        self.rewards.append(reward)
        if self.count % 100 == 0:
            if isinstance(design, (list, tuple)):
                design_str = str([int(x) for x in design])
            else:
                design_str = str(int(design))
            self.data.append([self.count, design_str, float(reward)])
        if self.count % 1000 == 0:
            table = wandb.Table(data=self.data, columns=self.columns)
            wandb.log({"designs": table, "train/design_count": self.count})
        self.count += 1

    def state_dict(self):
        return {
            'designs': list(self.designs),
            'rewards': list(self.rewards),
            'count': self.count
        }

    def load_state_dict(self, state_dict):
        self.designs = deque(state_dict['designs'], maxlen=self.maxlen)
        self.rewards = deque(state_dict['rewards'], maxlen=self.maxlen)
        self.count = state_dict['count']

    def get_designs_and_rewards(self, n):
        return list(self.designs)[-n:], np.array(self.rewards)[-n:]

    def get_design_count(self):
        return self.count


class DesignManager(VecEnvWrapper):
    """Environment wrapper to handle design sampling/logging.

    This wrapper:
         - samples and sets design parameters at fixed intervals.
         - records and logs the performance of each design.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.steps_per_design = None
        self.design_count = 0
        self.steps = 0
        self.design_dist = None
        self.designs = None
        self.rewards = np.zeros(self.num_envs)
        self.design_space = get_design_space(
            gin.query_parameter('sofa_make_env.design_space')
        )(gin.query_parameter('sofa_make_env.ym'),
          gin.query_parameter('sofa_make_env.paper_ym'))

    def step_wait(self):
        return self.venv.step_wait()

    def set_design_dist(self, design_dist):
        self.design_dist = design_dist

    def get_design_count(self):
        return self.logger.get_design_count()

    def configure_design_manager(self, steps_per_design, maxlen):
        self.steps_per_design = steps_per_design
        self.logger = DesignLogger(maxlen)

    def get_designs_and_rewards(self, n):
        return self.logger.get_designs_and_rewards(n)

    def _unnorm(self, p):
        space = self.observation_space['design']
        if isinstance(space, gym.spaces.Box):
            return 0.5 * (p.clamp_(-1., 1.) + 1.) * (space.high - space.low) + space.low
        else:
            return p

    def _sample_design(self):
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy(), self.design_dist.sample()))

    def _sample_mode(self):
        with torch.no_grad():
            return self._unnorm(nest.map_structure(
                            lambda x: x.numpy(), self.design_dist.mode()))

    def init_scene_with_mode(self):
        self.designs = [self._sample_mode() for _ in range(self.num_envs)]
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.num_envs)

    def init_scene(self):
        self.designs = [self._sample_design() for _ in range(self.num_envs)]
        self.venv.set_designs(self.designs)
        self.rewards = np.zeros(self.num_envs)

    def reset(self, force=False):
        if self.designs is None:
            self.init_scene()
        return self.venv.reset(force=force)

    def step(self, action):
        ob, r, done, info = self.venv.step(action)
        self.steps += 1
        self.rewards += r
        if self.steps_per_design and self.steps % self.steps_per_design == 0:
            for design, reward in zip(self.designs, self.rewards):
                self.logger.log(design, reward)
            self.init_scene()
            done[:] = True
        return ob, r, done, info

    def state_dict(self):
        return {'designs': self.logger.state_dict()}

    def load_state_dict(self, state_dict):
        self.logger.load_state_dict(state_dict['designs'])
