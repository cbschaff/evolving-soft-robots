"""Environment wrappers."""
from gym import ObservationWrapper, ActionWrapper
from dl.rl.util.vec_env import VecEnvWrapper
from gym.spaces import Box, Tuple
import numpy as np


class ImageTranspose(ObservationWrapper):
    """Change from HWC to CHW or vise versa."""

    def __init__(self, env):
        """Init."""
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 3
        self.observation_space = Box(
            self.observation_space.low.transpose(2, 0, 1),
            self.observation_space.high.transpose(2, 0, 1),
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        """Observation."""
        return obs.transpose(2, 0, 1)


class EpsilonGreedy(ActionWrapper):
    """Epsilon greedy wrapper."""

    def __init__(self, env, epsilon):
        """Init."""
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        """Wrap actions."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return action


class VecEpsilonGreedy(VecEnvWrapper):
    """Epsilon greedy wrapper for vectorized environments."""

    def __init__(self, venv, epsilon):
        """Init."""
        super().__init__(venv)
        self.epsilon = epsilon

    def step(self, actions):
        """Wrap actions."""
        if np.random.rand() < self.epsilon:
            actions = np.array([self.action_space.sample()
                                for _ in range(self.num_envs)])
        return self.venv.step(actions)

    def step_wait(self):
        """Step."""
        return self.venv.step_wait()

    def reset(self, force=True):
        """Reset."""
        return self.venv.reset(force=force)


if __name__ == '__main__':
    import unittest
    import gym

    class Test(unittest.TestCase):
        """Test."""

        def test_image_transpose(self):
            """Test image transpose wrapper."""
            env = gym.make('PongNoFrameskip-v4')
            s = env.observation_space.shape
            env = ImageTranspose(env)
            ob = env.reset()
            assert ob.shape == (s[2], s[0], s[1])
            ob, _, _, _ = env.step(env.action_space.sample())
            assert ob.shape == (s[2], s[0], s[1])

    unittest.main()


class VecActionRewardInObWrapper(VecEnvWrapper):
    """Add Action and Reward to Observation."""

    def __init__(self, venv, reward_shape=(1,)):
        super().__init__(venv)
        self._zero_action = self._get_zero_action(self.action_space)
        self._zero_reward = np.zeros([self.num_envs] + list(reward_shape),
                                     dtype=np.float32)

    def _get_zero_action(self, ac_space):
        if isinstance(ac_space, Tuple):
            return [self._get_zero_action(a) for a in ac_space.spaces]
        else:
            if hasattr(ac_space, 'n'):
                return np.zeros((self.num_envs, 1), dtype=np.float32)
            else:
                return np.zeros((self.num_envs, self.action_space.shape),
                                dtype=np.float32)

    def reset(self, force=True):
        ob = self.venv.reset(force=force)
        return {
            'ob': ob,
            'action': self._zero_action,
            'reward': self._zero_reward
        }

    def step(self, action):
        ob, r, done, info = self.venv.step(action)
        reward = r[:, None] if len(r.shape) == 1 else r
        ac = action[:, None] if len(action.shape) == 1 else action
        ob = {
            'ob': ob,
            'action': ac.astype(np.float32),
            'reward': reward.astype(np.float32)
        }
        return ob, r, done, info

    def step_wait(self):
        """Step."""
        return self.venv.step_wait()
