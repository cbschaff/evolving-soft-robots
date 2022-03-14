"""Defines network for DQN experiments."""
from dl.rl.modules import DiscreteQFunctionBase, QFunction
from dl.rl.util import conv_out_shape
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gin


class NatureDQN(DiscreteQFunctionBase):
    """Deep network from https://www.nature.com/articles/nature14236."""

    def build(self):
        """Build."""
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 512)
        self.qf = nn.Linear(512, self.action_space.n)

    def forward(self, x):
        """Forward."""
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        return self.qf(x)


@gin.configurable
def nature_dqn_fn(env):
    """Create nature dqn qfunction."""
    return QFunction(NatureDQN(env.observation_space, env.action_space))
