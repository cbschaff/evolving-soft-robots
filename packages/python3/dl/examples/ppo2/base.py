"""Define networks for PPO2 experiments."""
from dl.rl.modules import PolicyBase, ValueFunctionBase, Policy, ValueFunction
from dl.modules import Categorical, DiagGaussian
import torch.nn.functional as F
import torch.nn as nn
import gin


class DiscretePolicyBase(PolicyBase):
    """Policy network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.dist = Categorical(128, self.action_space.n)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.dist(x)


class ContinuousPolicyBase(PolicyBase):
    """Policy network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.dist = DiagGaussian(128, self.action_space.shape[0])

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.dist(x)


class VFNet(ValueFunctionBase):
    """Value Function."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.vf = nn.Linear(128, 1)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.vf(x)


@gin.configurable
def discrete_policy_fn(env):
    """Create policy."""
    return Policy(DiscretePolicyBase(env.observation_space, env.action_space))


@gin.configurable
def continuous_policy_fn(env):
    """Create policy."""
    return Policy(ContinuousPolicyBase(env.observation_space, env.action_space))


@gin.configurable
def value_fn(env):
    """Create value function network."""
    return ValueFunction(VFNet(env.observation_space, env.action_space))
