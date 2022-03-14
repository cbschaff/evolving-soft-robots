"""Defines networks for SAC experiments."""
from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
from dl.rl.modules import QFunction, Policy
from dl.modules import TanhDelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


class FeedForwardPolicyBase(PolicyBase):
    """Policy network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.dist = TanhDelta(32, self.action_space.shape[0])

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.dist(x)


class AppendActionFeedForwardQFBase(ContinuousQFunctionBase):
    """Q network."""

    def build(self):
        """Build."""
        nin = self.observation_space.shape[0] + self.action_space.shape[0]
        self.fc1 = nn.Linear(nin, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.qvalue = nn.Linear(32, 1)

    def forward(self, x, a):
        """Forward."""
        x = F.relu(self.fc1(torch.cat([x, a], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.qvalue(x)


@gin.configurable
def policy_fn(env):
    """Create a policy network."""
    return Policy(FeedForwardPolicyBase(env.observation_space,
                                        env.action_space))


@gin.configurable
def qf_fn(env):
    """Create a qfunction network."""
    return QFunction(AppendActionFeedForwardQFBase(env.observation_space,
                                                   env.action_space))
