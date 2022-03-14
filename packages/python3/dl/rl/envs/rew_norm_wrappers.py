"""Environment Wrapper for normalizing observations."""
from dl.rl.util.vec_env import VecEnvWrapper
from dl.modules import RunningNorm
from dl.rl import RewardForwardFilter
import torch
import numpy as np


class VecRewardNormWrapper(VecEnvWrapper):
    """Reward normalization for vecorized environments."""

    def __init__(self, venv, gamma, eps=1e-5):
        """Init."""
        super().__init__(venv)
        self.rn = RunningNorm((1,), eps=eps)
        self.reward_filter = RewardForwardFilter(gamma)
        self._dones = np.zeros(self.num_envs, dtype=np.bool)
        self._eval = False

    def state_dict(self):
        """State dict."""
        return {
            'rn': self.rn.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.rn.load_state_dict(state_dict['rn'])

    def eval(self):
        """Set the environment to eval mode.

        Eval mode doesn't update the running norm of returns.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode."""
        self._eval = False

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.venv.step(action)
        if not self._eval:
            updates = np.logical_not(self._dones)
            rets = self.reward_filter(rews, updates)[updates]
            var = 0 if rets.shape[0] <= 1 else rets.var()
            self.rn.update(rets.mean(), var, rets.shape[0])
        self._dones = np.logical_or(dones, self._dones)
        if self.rn.std > self.rn.eps:
            rews = rews / (self.rn.std.numpy() + self.rn.eps)
        return obs, rews, dones, infos

    def reset(self, force=True):
        """Reset."""
        obs = self.venv.reset(force=force)
        self._dones[:] = False
        return obs

    def step_wait(self):
        return self.venv.step_wait()


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env

    class TestRewardNorm(unittest.TestCase):
        """Test."""

        def test_vec(self):
            """Test vec wrapper."""
            nenv = 10
            env = make_env('CartPole-v1', nenv)
            env = VecRewardNormWrapper(env, gamma=0.99)
            env.reset()
            for _ in range(5):
                out = env.step(np.array([env.action_space.sample()
                                         for _ in range(nenv)]))
                print(out[1])
            c = env.rn.count
            print(c)
            assert c == 5 * nenv
            state = env.state_dict()
            env.load_state_dict(state)
            assert c == env.rn.count

            env.eval()
            env.reset()
            for _ in range(10):
                env.step(np.array([env.action_space.sample()
                                   for _ in range(nenv)]))
            env.train()
            assert c == env.rn.count

    unittest.main()
