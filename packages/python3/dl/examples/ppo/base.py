"""Define networks for PPO experiments."""
from dl.rl.modules import ActorCriticBase, Policy
from dl.rl.util import conv_out_shape
from dl.modules import Categorical
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import numpy as np
import gin


class A3CCNN(ActorCriticBase):
    """Deep network from https://arxiv.org/abs/1602.01783."""

    def build(self):
        """Build."""
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2]:
            shape = conv_out_shape(shape, c)
        self.nunits = 32 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 256)
        self.vf = nn.Linear(256, 1)
        self.dist = Categorical(256, self.action_space.n)
        nn.init.orthogonal_(self.vf.weight.data, gain=1.0)
        nn.init.constant_(self.vf.bias.data, 0)

    def forward(self, x):
        """Forward."""
        x = (x.float() / 128.) - 1.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        return self.dist(x), self.vf(x)


class A3CRNN(ActorCriticBase):
    """Deep recurrent network from https://arxiv.org/abs/1602.01783."""

    def build(self):
        """Build."""
        self.conv1 = nn.Conv2d(1, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2]:
            shape = conv_out_shape(shape, c)
        self.nunits = 32 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 256)
        self.lstm = nn.LSTM(256, 256, 1)
        self.vf = nn.Linear(256, 1)
        self.dist = Categorical(256, self.action_space.n)
        nn.init.orthogonal_(self.vf.weight.data, gain=1.0)
        nn.init.constant_(self.vf.bias.data, 0)

    def forward(self, ob, state_in=None):
        """Forward."""
        if isinstance(ob, PackedSequence):
            x = ob.data
        else:
            x = ob
        x = (x.float() / 128.) - 1.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        if isinstance(ob, PackedSequence):
            x = PackedSequence(x, batch_sizes=ob.batch_sizes,
                               sorted_indices=ob.sorted_indices,
                               unsorted_indices=ob.unsorted_indices)
        else:
            x = x.unsqueeze(0)
        if state_in is None:
            x, state_out = self.lstm(x)
        else:
            x, state_out = self.lstm(x, state_in)
        if isinstance(x, PackedSequence):
            x = x.data
        else:
            x = x.squeeze(0)
        return self.dist(x), self.vf(x), state_out


@gin.configurable
def a3c_cnn_fn(env):
    """Create a3c conv net policy."""
    return Policy(A3CCNN(env.observation_space, env.action_space))


@gin.configurable
def a3c_rnn_fn(env):
    """Create a3c recurrent policy."""
    return Policy(A3CRNN(env.observation_space, env.action_space))
