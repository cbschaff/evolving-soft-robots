
from dl.rl.util.vec_env import VecEnvWrapper
import torch
import numpy as np


class RNDVecEnv(VecEnvWrapper):
    """Augment rewards with random network distillation."""

    def __init__(self, venv, rnd):
        super().__init__(venv)
        self.rnd = rnd
        self._eval = False
        self._dones = np.zeros(self.num_envs, dtype=np.bool)

    def reset(self, force=True):
        self._dones[:] = False
        return self.venv.reset(force=force)

    def step_wait(self):
        return self.venv.step_wait()

    def _to_torch(self, ob):
        return torch.from_numpy(ob).to(self.rnd.device).float()

    def step(self, action):
        """Step."""
        ob, reward, done, info = self.venv.step(action)
        updates = np.logical_not(self._dones)
        reward_intrinsic = self.rnd(self._to_torch(ob), updates=updates,
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
    from dl.rl.modules import RND
    import torch.nn.functional as F
    import torch.nn as nn

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

    def _env(nenv):
        """Create a training environment."""
        return make_env("LunarLander-v2", nenv)

    class Test(unittest.TestCase):
        """Test."""

        def test_rnd_env(self):
            """Test vec frame stack wrapper."""
            nenv = 4
            env = _env(nenv)
            env = RNDVecEnv(env, rnd)
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
