import torch
import torch.nn as nn
import numpy as np
import gym
import gin
from dl import CatDist
import wandb


class LinearSchedule():
    def __init__(self, start_val, end_val, start_step, end_step):
        self.sv = start_val
        self.ev = end_val
        self.ss = start_step
        self.es = end_step

    def __call__(self, t):
        if t <= self.ss:
            return self.sv
        if t >= self.es:
            return self.ev
        frac = (t - self.ss) / (self.es - self.ss)
        return self.ev * frac + self.sv * (1 - frac)


@gin.configurable(blacklist=['design_space'])
class DiscreteDesignOptimizer(nn.Module):
    def __init__(self,
                 design_space,
                 ent_decay_start,
                 ent_decay_end,
                 ):
        super().__init__()
        self.design_space = design_space
        self.scores = torch.nn.Parameter(torch.zeros((self.design_space.n,), dtype=torch.float32),
                                         requires_grad=False)
        self.target = LinearSchedule(np.log(self.design_space.n), 0, ent_decay_start, ent_decay_end)
        self.beta = 0.0
        self.beta_min = 0
        self.beta_max = 20
        wandb.define_metric("design/*", step_metric="train/step")

    def get_design_dist(self):
        return CatDist(logits=self.beta * self.scores)

    def set_beta(self, t):
        high = self.beta_max
        low = self.beta_min
        target = self.target(t)
        beta = (high + low) / 2.0
        ent = CatDist(logits=beta * self.scores).entropy()
        while np.abs(ent - target) > 0.01:
            if ent > target:
                low = beta
            else:
                high = beta
            beta = (high + low) / 2.0
            if beta > 0.99 * self.beta_max:
                beta = self.beta_max
                break
            ent = CatDist(logits=beta * self.scores).entropy()
        self.beta = beta

    def sample(self):
        """Return a sample from the design distribution."""
        return self.get_design_dist().sample()

    def update(self, designs, rewards, t):
        """Update the design distribution from the performance of a batch of designs."""
        self.scores[designs] = rewards.float()
        self.set_beta(t)

    def log(self, t):
        ent = self.get_design_dist().entropy()
        target = self.target(t)
        wandb.log({'design/beta': self.beta,
                   'design/ent': ent,
                   'design/ent_target': target,
                   'design/perplexity': np.exp(ent),
                   'design/perplexity_target': np.exp(target),
                   'train/step': t})

    def forward(self):
        return self.sample()

    def state_dict(self):
        """Return the state dict to save."""
        state_dict = {
            'params': super().state_dict(),
            'beta': self.beta
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """Load parameters from state_dict."""
        super().load_state_dict(state_dict['params'])
        self.beta = state_dict['beta']


if __name__ == '__main__':
    wandb.init()
    design_space = gym.spaces.Discrete(n=10000)
    opt = DiscreteDesignOptimizer(design_space, 0, 100)
    for t in range(100):
        designs = torch.stack([opt.sample() for _ in range(100)])
        rewards = 10 * torch.from_numpy(np.random.rand(100)).float()
        opt.update(designs, rewards, t)
        opt.log(t)
        if t % 10 == 0:
            state = opt.state_dict()
            opt = DiscreteDesignOptimizer(design_space, 0, 100)
            opt.load_state_dict(state)

