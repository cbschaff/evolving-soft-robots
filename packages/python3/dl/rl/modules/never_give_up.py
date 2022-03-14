"""Intrinsic Motivation module from the Never Give Up paper.

https://arxiv.org/abs/2002.06038
"""

import torch
import torch.nn as nn
import numpy as np
from dl.modules import RunningNorm
from dl.rl import RewardForwardFilter


class EpisodeBuffer(object):
    def __init__(self, capacity, nenv, dim, device):
        self.data = torch.zeros((capacity, nenv, dim), device=device)
        self.ind = torch.zeros((nenv,), device=device).long()
        self.n_in_buffer = torch.zeros((nenv,), device=device).long()
        self.capacity = capacity
        self.nenv = nenv
        self.device = device
        self.norm2 = torch.zeros((capacity, nenv), device=device)

    def reset(self, dones=None):
        if dones is None:
            self.ind[:] = 0
            self.n_in_buffer[:] = 0
        else:
            self.ind[dones] = 0
            self.n_in_buffer[dones] = 0

    def insert(self, obs):
        self.data[self.ind] = obs
        self.norm2[self.ind] = (obs * obs).sum(dim=1)
        self.ind = (self.ind + 1) % self.capacity
        self.n_in_buffer = torch.clamp(self.n_in_buffer + 1, max=self.capacity)

    def get_nearest_neighbors(self, obs, k):
        obs_norm2 = (obs * obs).sum(dim=1)
        dists = []
        for ind in range(self.nenv):
            if self.n_in_buffer[ind] < k:
                dists.append(torch.zeros((1, k), device=self.device))
            else:
                dot = torch.matmul(obs[ind:ind+1],
                                   self.data[:self.n_in_buffer[ind], ind].T)
                di = (obs_norm2[ind]
                      + self.norm2[:self.n_in_buffer[ind], ind]
                      - 2 * dot)
                dists.append(torch.sort(di, dim=1)[0][:, :k])
        return torch.cat(dists, dim=0)


class NGU(nn.Module):
    def __init__(self, rnd, ide, capacity, device, gamma=0.99, k=10, L=5,
                 eps=0.001, min_dist=0.008, max_similarity=4, c=1.0):
        super().__init__()
        self.rnd = rnd
        # Change the reward normalization of RND to normalize reward instead of
        # return. This will allow rnd to modify intrinsic rewards meaningfully.
        # (rewards will have std of 1, instead of approximately 1-gamma)
        self.rnd.reward_filter = lambda r, updates: r
        self.ide = ide
        self.capacity = capacity
        self.device = device
        self.k = k
        self.L = L
        self.eps = eps
        self.min_dist = min_dist
        self.max_similarity = np.sqrt(max_similarity)
        self.c = c
        self.buffer = None
        self.dist_running_avg = 0.
        self.dist_running_count = 0.
        self.reward_filter = RewardForwardFilter(gamma)
        self.err_norm = RunningNorm((1,)).to(device)

    def forward(self, obs, update_norm=False, updates=None):
        with torch.no_grad():
            embeddings = self.ide(obs)
            if self.buffer is None:
                self.buffer = EpisodeBuffer(self.capacity, embeddings.shape[0],
                                            embeddings.shape[1], self.device)

            dists = self.buffer.get_nearest_neighbors(embeddings, self.k)
            # Normalize distances by average of kth-neighbors distance
            self._update_dist_running_mean(dists[:, -1])
            if self.dist_running_avg > 1e-5:
                dists /= self.dist_running_avg

            # Set close distances to 0
            dists = torch.max(dists - self.min_dist, torch.zeros_like(dists))

            # Compute inverse kernel of distances
            k = self.eps / (dists + self.eps)

            # Compute similarity
            s = torch.sqrt(self.c + k.sum(dim=1))

            # Compute short term intrinsic reward
            r = torch.where(s > self.max_similarity, torch.zeros_like(s), 1 / s)

            # combine with long term intrinsic reward
            # RND divides error by a running estimate of error std.
            # NGU also subtracts the mean and adds 1.
            r_rnd = self.rnd(obs, update_norm, updates)
            rnd_mean = self.rnd.err_norm.mean / (self.rnd.err_norm.std + 1e-5)
            r_rnd += 1.0 - rnd_mean
            modifier = torch.clamp(r_rnd, 1, self.L)
            reward = r * modifier

            self.buffer.insert(embeddings)

            # update running norm
            if update_norm and (updates is None or updates.sum() > 0):
                rets = self.reward_filter(reward, updates)[updates]
                var = 0 if rets.shape[0] == 1 else rets.var()
                self.err_norm.update(rets.mean(), var, rets.shape[0])

            return reward / (self.err_norm.std + 1e-5)

    def _update_dist_running_mean(self, max_dists):
        inds = self.buffer.n_in_buffer >= self.k
        max_dists = max_dists[inds]
        if max_dists.shape[0] == 0:
            return
        d = max_dists.mean()
        c = len(max_dists)
        new_c = self.dist_running_count + c
        self.dist_running_avg *= self.dist_running_count / new_c
        self.dist_running_avg += (c / new_c) * d
        self.dist_running_count += c

    def reset(self, dones=None):
        if self.buffer:
            self.buffer.reset(dones)

    def update_rnd(self, obs):
        return self.rnd.update(obs)

    def update_ide(self, obs, next_obs, actions):
        return self.ide.update(obs, next_obs, actions)

    def state_dict(self):
        return {
            'rnd': self.rnd.state_dict(),
            'ide': self.ide.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.rnd.load_state_dict(state_dict['rnd'])
        self.ide.load_state_dict(state_dict['ide'])


if __name__ == '__main__':
    import torch.nn.functional as F
    from dl.modules import Categorical
    from dl.rl import make_env, RND, InverseDynamicsEmbedding

    class RNDNet(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.fc1 = nn.Linear(shape[0], 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

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
            return loss

    nenv = 2
    env = make_env('LunarLander-v2', nenv=nenv)
    rnd = RND(RNDNet, torch.optim.Adam, 0.99, env.observation_space.shape,
              'cpu')
    emb = InverseDynamicsEmbedding(env, EmbeddingNet, PredictionNet, Loss,
                                   torch.optim.Adam, 'cpu')
    ngu = NGU(rnd, emb, 50, 'cpu')

    for _ in range(100):
        ngu.reset()
        done = [False for _ in range(nenv)]
        obs = [env.reset()]
        acs = []
        while not np.any(done):
            ngu.reset(done)
            print(ngu(torch.from_numpy(obs[-1]), update_norm=True))
            acs.append([env.action_space.sample() for _ in range(nenv)])
            ob, _, done, _ = env.step(acs[-1])
            obs.append(ob)
        obs = torch.from_numpy(np.stack(obs, axis=0))
        acs = torch.from_numpy(np.stack(acs, axis=0))

        n = obs.shape
        for _ in range(10):
            ngu.update_rnd(obs)
            ngu.update_ide(obs[:-1].view(-1, obs.shape[-1]),
                           obs[1:].view(-1, obs.shape[-1]), acs.view(-1))
