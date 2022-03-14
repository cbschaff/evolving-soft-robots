"""Environment Wrapper for normalizing observations."""
from dl.rl.util.vec_env import VecEnvWrapper
from dl import logger, nest
from dl.rl.util import get_ob_norm
import numpy as np
import time
import gin


@gin.configurable(blacklist=['norm'])
class VecObsNormWrapper(VecEnvWrapper):
    """Observation normalization for vecorized environments.

    Collects data from a random policy and computes a fixed normalization.
    Computing norm params is done lazily when normalization constants
    are needed and unknown.
    """

    def __init__(self, venv,
                 norm=True,
                 steps=10000,
                 mean=None,
                 std=None,
                 eps=1e-2,
                 log=True,
                 log_prob=0.01):
        """Init."""
        super().__init__(venv)
        self.steps = steps
        self.should_norm = norm
        self.eps = eps
        self.log = log
        self.log_prob = log_prob
        self.t = 0
        self._eval = False
        self.mean = None
        self.std = None
        self._dones = np.zeros(self.num_envs, dtype=np.bool)

        if mean is not None and std is not None:
            if not nest.has_same_structure(mean, std):
                raise ValueError("mean and std must have the same structure.")
            self.mean = mean
            self.std = nest.map_structure(
                        lambda x: np.maximum(x, self.eps), std)

    def _env(self):
        return self.env if hasattr(self, 'env') else self.venv

    def find_norm_params(self):
        """Calculate mean and std with a random policy to collect data."""
        self.mean, self.std = get_ob_norm(self._env(), self.steps, self.eps)

    def _normalize(self, obs):
        if not self.should_norm:
            return obs
        if self.mean is None or self.std is None:
            self.find_norm_params()
        obs = nest.map_structure(np.asarray, obs)
        obs = nest.map_structure(np.float32, obs)
        if not nest.has_same_structure(self.mean, obs):
            raise ValueError("mean and obs do not have the same structure!")

        def norm(item):
            ob, mean, std = item
            if mean is not None:
                return (ob - mean) / std
            else:
                return ob
        return nest.map_structure(norm, nest.zip_structure(obs, self.mean,
                                                           self.std))

    def state_dict(self):
        """State dict."""
        return {
            'mean': self.mean,
            'std': self.std,
            't': self.t,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.t = state_dict['t']

    def norm_and_log(self, obs):
        """Norm observations and log."""
        obs = self._normalize(obs)
        if not self._eval and self.log and self.log_prob > np.random.rand():
            flat_ob = np.concatenate([
                ob.ravel() for ob in nest.flatten(obs) if ob is not None
            ])
            percentiles = {
                '00': np.quantile(flat_ob, 0.0),
                '10': np.quantile(flat_ob, 0.1),
                '25': np.quantile(flat_ob, 0.25),
                '50': np.quantile(flat_ob, 0.5),
                '75': np.quantile(flat_ob, 0.75),
                '90': np.quantile(flat_ob, 0.9),
                '100': np.quantile(flat_ob, 1.0),
            }
            logger.add_scalars('ob_stats/percentiles', percentiles,
                               self.t, time.time())
        return obs

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables logging and stops counting steps.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode counts steps and logs obs distribution if self.log is True.
        """
        self._eval = False

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.venv.step(action)
        if not self._eval:
            self.t += np.sum(np.logical_not(self._dones))
        self._dones = np.logical_or(dones, self._dones)
        return self.norm_and_log(obs), rews, dones, infos

    def reset(self, force=True):
        """Reset."""
        obs = self.venv.reset(force=force)
        self._dones[:] = False
        return self._normalize(obs)

    def step_wait(self):
        return self.venv.step_wait()


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env
    from gym.spaces import Tuple, Discrete

    class NestedVecObWrapper(VecEnvWrapper):
        """Nest observations."""

        def __init__(self, venv):
            """Init."""
            super().__init__(venv)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space,
                                            Discrete(3)])

        def reset(self, force=True):
            """Reset."""
            ob = self.venv.reset(force=force)
            return (ob, ob, np.random.randint(3, size=(self.num_envs,)))

        def step_wait(self):
            """Step."""
            ob, r, done, info = self.venv.step_wait()
            return (ob, ob, np.random.randint(3, size=(self.num_envs))), r, done, info

    class TestObNorm(unittest.TestCase):
        """Test."""

        def test_vec(self):
            """Test vec wrapper."""
            logger.configure('./.test')
            nenv = 10
            env = make_env('CartPole-v1', nenv)
            env = VecObsNormWrapper(env, log_prob=1.)
            print(env.observation_space)
            env.reset()
            assert env.t == 0
            for _ in range(5):
                env.step(np.array([env.action_space.sample()
                                   for _ in range(nenv)]))
            state = env.state_dict()
            assert state['t'] == env.t
            assert np.allclose(state['mean'], env.mean)
            assert np.allclose(state['std'], env.std)
            state['t'] = 0
            env.load_state_dict(state)
            assert env.t == 0

            env.eval()
            env.reset()
            for _ in range(10):
                env.step(np.array([env.action_space.sample()
                                   for _ in range(nenv)]))
            assert env.t == 0
            env.train()
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

        def test_nested_observations(self):
            """Test nested observations."""
            logger.configure('./.test')
            env = make_env('CartPole-v1', 1)
            env = NestedVecObWrapper(env)
            env = NestedVecObWrapper(env)
            env = VecObsNormWrapper(env, log_prob=1.)
            print(env.observation_space)
            env.reset()
            assert env.t == 0
            for _ in range(100):
                _, _, done, _ = env.step(np.array([env.action_space.sample()
                                                   for _ in range(1)]))
                if done:
                    env.reset()
            assert env.t == 100
            state = env.state_dict()
            assert state['t'] == env.t
            state['t'] = 0
            env.load_state_dict(state)
            assert env.t == 0

            env.eval()
            env.reset()
            for _ in range(3):
                env.step(np.array([env.action_space.sample()]))
            assert env.t == 0
            env.train()
            for _ in range(3):
                env.step(np.array([env.action_space.sample()]))
            assert env.t == 3
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

    unittest.main()
