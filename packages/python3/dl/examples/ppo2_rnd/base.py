"""Define networks for PPO2 experiments."""
from dl.rl.modules import PolicyBase, ValueFunctionBase, Policy, ValueFunction
from dl.modules import Categorical
from dl.rl.util import conv_out_shape
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gin


@gin.configurable
class RNDNet(nn.Module):
    """Deep network from https://www.nature.com/articles/nature14236."""

    def __init__(self, shape):
        """Build network."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        shape = shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 512)

    def forward(self, x):
        """Forward."""
        # Remove frame stack to get a function of only the current observation.
        x = F.relu(self.conv1(x[:, -1:]))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc(x.view(-1, self.nunits))


class NatureDQN(PolicyBase):
    """Deep network from https://www.nature.com/articles/nature14236."""

    def build(self):
        """Build network."""
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 512)
        self.dist = Categorical(512, self.action_space.n)

    def forward(self, x):
        """Forward."""
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        return self.dist(x)


class NatureDQNVF(ValueFunctionBase):
    """Deep network from https://www.nature.com/articles/nature14236."""

    def build(self):
        """Build network."""
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 512)
        self.vf = nn.Linear(512, 2)

    def forward(self, x):
        """Forward."""
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        return self.vf(x)


@gin.configurable
def policy_fn(env):
    """Create policy."""
    return Policy(NatureDQN(env.observation_space, env.action_space))


@gin.configurable
def value_fn(env):
    """Create value function network."""
    return ValueFunction(NatureDQNVF(env.observation_space, env.action_space))
