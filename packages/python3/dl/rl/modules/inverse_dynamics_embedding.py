"""https://arxiv.org/pdf/1705.05363.pdf

Also: https://arxiv.org/abs/2002.06038"""

import torch
import torch.nn as nn


class InverseDynamicsEmbedding(nn.Module):
    """Embed states by optimizing an inverse dynamics loss."""

    def __init__(self, env, embedding_net, prediction_net, loss, opt, device):
        """Init"""

        super().__init__()
        self.embedding_net = embedding_net(env.observation_space).to(device)
        self.prediction_net = prediction_net(env.action_space).to(device)
        self.loss = loss()
        self.device = device
        self.opt = opt(self.parameters())

    def forward(self, obs):
        with torch.no_grad():
            return self.embedding_net(obs)

    def update(self, obs, next_obs, actions):
        self.opt.zero_grad()
        obs1 = self.embedding_net(obs)
        obs2 = self.embedding_net(next_obs)
        action_pred = self.prediction_net(obs1, obs2)
        loss = self.loss(action_pred, actions)
        loss.backward()
        self.opt.step()
        return loss

    def state_dict(self, *args, **kwargs):
        return {
            'ide': super().state_dict(*args, **kwargs),
            'opt': self.opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict['opt'])
        super().load_state_dict(state_dict['ide'])


if __name__ == '__main__':
    import torch.nn.functional as F
    from dl.modules import Categorical
    from dl.rl import make_env
    import numpy as np

    class EmbeddingNet(nn.Module):
        def __init__(self, observation_space):
            super().__init__()
            self.fc1 = nn.Linear(observation_space.shape[0], 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 32)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    class PredictionNet(nn.Module):
        def __init__(self, action_space):
            super().__init__()
            self.fc1 = nn.Linear(64, 32)
            self.dist = Categorical(32, action_space.n)

        def forward(self, x1, x2):
            x = torch.cat((x1, x2), dim=1)
            x = F.relu(self.fc1(x))
            return self.dist(x)

    class Loss(nn.Module):
        def forward(self, dist, actions):
            loss = F.cross_entropy(dist.logits, actions)
            print(loss)
            return loss

    env = make_env('LunarLander-v2')

    emb = InverseDynamicsEmbedding(env, EmbeddingNet, PredictionNet, Loss,
                                   torch.optim.Adam, 'cpu')

    obs = [env.reset()]
    acs = []
    for _ in range(100):
        acs.append([env.action_space.sample()])
        obs.append(env.step(acs[-1])[0])

    obs = torch.from_numpy(np.concatenate(obs, axis=0))
    acs = torch.from_numpy(np.concatenate(acs, axis=0))

    for _ in range(1000):
        emb.update(obs[:-1], obs[1:], acs)
    emb.load_state_dict(emb.state_dict())
