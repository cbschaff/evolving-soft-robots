"""https://arxiv.org/abs/1810.12894"""

from dl.modules import RunningNorm
from dl.rl import RewardForwardFilter
import torch
import torch.nn as nn


class RND(nn.Module):
    """Implementation of Random Network Distillation."""

    def __init__(self, net, opt, gamma, shape, device):
        """
        Args:
            net (function): a function mapping input shape to a pytorch module.
            opt : PyTorch optimizer
            shape (tuple) : input shape for networks
            device : The device to place the networks on
        """

        super().__init__()
        self.target_net = net(shape).to(device)
        self.prediction_net = net(shape).to(device)
        self.device = device
        self.opt = opt(self.prediction_net.parameters())
        self.ob_norm = RunningNorm(shape).to(device)
        self.err_norm = RunningNorm((1,)).to(device)
        self.reward_filter = RewardForwardFilter(gamma)

    def forward(self, obs, update_norm=False, updates=None):
        """Get intrinsic reward."""
        should_update = update_norm and (updates is None or updates.sum() > 0)
        if should_update:
            obs_to_update = obs if updates is None else obs[updates]
            if obs_to_update.shape[0] == 1:
                var = torch.zeros_like(obs[0])
            else:
                var = obs_to_update.var(dim=0)
            self.ob_norm.update(obs_to_update.mean(dim=0), var,
                                obs_to_update.shape[0])

        obs = torch.clamp(self.ob_norm(obs), -5, 5)
        with torch.no_grad():
            err = torch.norm(self.target_net(obs) - self.prediction_net(obs),
                             dim=1)
            if should_update:
                rets = self.reward_filter(err, updates)[updates]
                var = 0 if rets.shape[0] == 1 else rets.var()
                self.err_norm.update(rets.mean(), var, rets.shape[0])
            return err / (self.err_norm.std + 1e-5)

    def update(self, obs):
        obs = torch.clamp(self.ob_norm(obs), -5, 5)
        self.opt.zero_grad()
        loss = torch.norm(self.target_net(obs) - self.prediction_net(obs),
                          dim=1).mean()
        loss.backward()
        self.opt.step()
        return loss

    def state_dict(self, *args, **kwargs):
        return {
            'rnd': super().state_dict(*args, **kwargs),
            'opt': self.opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict['opt'])
        super().load_state_dict(state_dict['rnd'])


if __name__ == '__main__':
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.fc1 = nn.Linear(shape[0], 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    rnd = RND(Net, torch.optim.Adam, 0.99, (8,), 'cpu')

    obs = torch.randn((16, 8))
    print(rnd(obs, update_norm=True))
    for _ in range(100):
        rnd.update(obs)
        print(rnd(obs, update_norm=True))
