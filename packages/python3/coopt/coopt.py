"""Joint optimization of design and control."""

import numpy as np
import torch
import gin
import os
import gym
from dl import Algorithm, Checkpointer, nest
from .sac import SAC
from .discrete_robot_optimizer import DiscreteDesignOptimizer


@gin.configurable(blacklist=['logdir'])
class CoOpt(Algorithm):
    def __init__(self,
                 logdir,
                 steps_per_design,  # Number of evironment steps each design takes
                 batch_size,        # number of designs used in each update
                 update_period,     # number of designs sampled between updates
                 ):
        self.logdir = logdir
        self.sac = SAC(logdir)
        self.steps_per_design = steps_per_design
        self.batch_size = batch_size
        self.update_period = update_period
        self.env = self.sac.env
        self.env.configure_design_manager(
            self.steps_per_design,
            self.batch_size
        )
        self.parameter_space = self.env.parameter_space
        self.design_optimizer = DiscreteDesignOptimizer(self.parameter_space)
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts_design_params'))
        self.t = self.sac.t
        self.design = None
        self.last_design_update = 0
        self._prev_design_grad = None
        self._prev_design_count = 0

    def step(self):
        self.env.set_design_dist(self.design_optimizer.get_design_dist())
        if self.t == 0:
            self.design_optimizer.log(self.t)
        dt = self.sac.step() - self.t
        self.t += dt

        design_count = self.env.get_design_count()
        designs_since_update = design_count - self.last_design_update
        if designs_since_update >= self.update_period:
            designs, rewards = self.env.get_designs_and_rewards(self.batch_size)
            designs = nest.map_structure(torch.from_numpy, designs)
            self.design_optimizer.update(designs, torch.from_numpy(rewards),
                                         self.t)
            self.design_optimizer.log(self.t)
            self.last_design_update = design_count
        return self.t

    def evaluate(self):
        pass

    def save(self):
        self.sac.save()
        state = self.design_optimizer.state_dict()
        self.ckptr.save(state, self.t)

    def load(self, t=None):
        state_dict = self.ckptr.load(t)
        if state_dict is not None:
            self.design_optimizer.load_state_dict(state_dict)
        self.env.set_design_dist(self.design_optimizer.get_design_dist())
        self.t = self.sac.load(t)
        design_count = self.env.get_design_count()
        self.last_design_update = design_count - (design_count % self.update_period)
        return self.t

    def close(self):
        self.env.close()
        self.sac.close()
