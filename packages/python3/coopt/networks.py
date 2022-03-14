"""Defines networks for Co-opt experiments."""
from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
from dl.rl.modules import Policy, QFunction
from dl.modules import TanhDiagGaussian, Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import gym


class MultiDiscreteDesignEmbedding(nn.Module):
    def __init__(self, ob_space):
        super().__init__()
        self.nvec = ob_space.nvec
        self.n = sum(self.nvec)

    def forward(self, x):
        design = x.long()
        design_emb = [F.one_hot(design[:, i], num_classes=n) for i, n in enumerate(self.nvec)]
        return torch.cat(design_emb, dim=1)


class OpenLoopObservationEmbedding(nn.Module):
    def __init__(self, ob_space, design_dim, n_out):
        super().__init__()
        if isinstance(ob_space['design'], gym.spaces.MultiDiscrete):
            self.design_emb = MultiDiscreteDesignEmbedding(ob_space['design'])
            n = self.design_emb.n
        else:
            self.design_emb = nn.Embedding(ob_space['design'].n, 8*design_dim,
                                           max_norm=1.)
            n = 8*design_dim
        n_a = ob_space['observation']['actions'].shape[0]
        self.fc = nn.Linear(n_a + n, n_out)

    def forward(self, x):
        d = x['design'].long()
        if len(d.shape) > 1:
            d = d.squeeze(dim=1)
        emb = torch.cat([
            self.design_emb(d),
            x['observation']['actions']
        ], dim=1)
        return F.relu(self.fc(emb))


class OpenLoopFeedForwardPolicyBase(PolicyBase):
    """Policy network."""
    def __init__(self, observation_space, action_space, units):
        self.units = units
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        self.emb = OpenLoopObservationEmbedding(
            self.observation_space,
            design_dim=3,
            n_out=self.units[0]
        )
        modules = []
        for i, n in enumerate(self.units[1:]):
            modules.append(nn.Linear(self.units[i], n))
            modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
        if isinstance(self.action_space, gym.spaces.Box):
            self.dist = TanhDiagGaussian(self.units[-1], self.action_space.shape[0])
        else:
            self.dist = Categorical(self.units[-1], self.action_space.n)

    def forward(self, x):
        """Forward."""
        x = self.emb(x)
        return self.dist(self.net(x))


class OpenLoopAppendActionFeedForwardQFBase(ContinuousQFunctionBase):
    """Q network."""
    def __init__(self, observation_space, action_space, units):
        self.units = units
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        self.emb = OpenLoopObservationEmbedding(
            self.observation_space,
            design_dim=3,
            n_out=self.units[0]
        )
        n_in = self.units[0] + self.action_space.shape[0]
        modules = [nn.Linear(n_in, self.units[1]), nn.ReLU()]
        for i, n in enumerate(self.units[2:]):
            modules.append(nn.Linear(self.units[i+1], n))
            modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
        self.qvalue = nn.Linear(self.units[-1], 1)

    def forward(self, x, a):
        """Forward."""
        x = torch.cat([self.emb(x), a], dim=1)
        return self.qvalue(self.net(x))


@gin.configurable(module='networks')
def open_loop_policy_fn(env, units=[64,64,64]):
    """Create a policy network."""
    return Policy(OpenLoopFeedForwardPolicyBase(env.observation_space,
                                                env.action_space, units))


@gin.configurable(module='networks')
def open_loop_qf_fn(env, units=[64,64,64]):
    """Create a qfunction network."""
    return QFunction(OpenLoopAppendActionFeedForwardQFBase(env.observation_space,
                                                           env.action_space, units))
