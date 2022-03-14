
from dl.rl.util.vec_env import VecEnvWrapper
import torch
import numpy as np


class NGUVecEnv(VecEnvWrapper):
    """Augment rewards with intrinsic reward from Never Give Up."""

    def __init__(self, venv, ngu):
        super().__init__(venv)
        self.ngu = ngu
        self._eval = False
        self._dones = np.zeros(self.num_envs, dtype=np.bool)

    def reset(self, force=True):
        if force:
            self.ngu.reset()
        else:
            self.ngu.reset(self._dones)
        self._dones[:] = False
        return self.venv.reset(force=force)

    def step_wait(self):
        return self.venv.step_wait()

    def _to_torch(self, ob):
        return torch.from_numpy(ob).to(self.ngu.device).float()

    def step(self, action):
        """Step."""
        ob, reward, done, info = self.venv.step(action)
        updates = np.logical_not(self._dones)
        reward_intrinsic = self.ngu(self._to_torch(ob), updates=updates,
                                    update_norm=not self._eval).cpu().numpy()
        reward = np.stack([reward, reward_intrinsic], axis=1)
        self._dones = np.logical_or(done, self._dones)
        return ob, reward, done, info

    def train(self):
        self._eval = False

    def eval(self):
        self._eval = True


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import make_env
    from dl.rl import RND, InverseDynamicsEmbedding, NGU
    from dl.modules import Categorical
    import torch.nn as nn
    import torch.nn.functional as F

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
            print(loss)
            return loss

    class Test(unittest.TestCase):
        """Test."""

        def test_rnd_env(self):
            """Test vec frame stack wrapper."""
            nenv = 2
            env = make_env('LunarLander-v2', nenv=nenv)
            rnd = RND(RNDNet, torch.optim.Adam, 0.99,
                      env.observation_space.shape, 'cpu')
            emb = InverseDynamicsEmbedding(env, EmbeddingNet, PredictionNet,
                                           Loss, torch.optim.Adam, 'cpu')
            ngu = NGU(rnd, emb, 50, 'cpu')

            env = NGUVecEnv(env, ngu)
            env.reset()
            _, r, _, _ = env.step(np.array([env.action_space.sample()
                                            for _ in range(nenv)]))
            assert r.shape == (nenv, 2)
            for _ in range(1000):
                _, r, done, _ = env.step(np.array([env.action_space.sample()
                                                   for _ in range(nenv)]))
                assert r.shape == (nenv, 2)

                if np.any(done):
                    env.reset(force=False)

    unittest.main()
