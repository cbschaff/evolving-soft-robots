"""Environment Wrappers for logging episode stats."""

from gym import Wrapper
from dl.rl.util.vec_env import VecEnvWrapper
from dl import logger
import numpy as np
import time


class EpisodeInfo(Wrapper):
    """Return episode information in step function.

    Pass episode stats through the info dict returned by step().
    If placed before wrappers which modify episode length and reward,
    this will provide easy access to unmodified episode stats.
    """

    def __init__(self, env):
        """Init."""
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0
        self.needs_reset = True

    def reset(self):
        """Reset."""
        self.episode_reward = 0
        self.episode_length = 0
        self.needs_reset = False
        return self.env.reset()

    def step(self, action):
        """Step."""
        assert not self.needs_reset, (
            "Can't step environment when the episode ends. "
            "Please call reset() first.")
        ob, r, done, info = self.env.step(action)
        if done:
            self.needs_reset = True
        self.episode_reward += r
        self.episode_length += 1
        assert 'episode_info' not in info, (
            f"Can't save episode data. Another EpisodeInfo Wrapper exists.")
        info['episode_info'] = {}
        info['episode_info']['reward'] = self.episode_reward
        info['episode_info']['length'] = self.episode_length
        info['episode_info']['done'] = done
        return ob, r, done, info


class VecEpisodeLogger(VecEnvWrapper):
    """EpisodeLogger for vecorized environments."""

    def __init__(self, venv, tstart=0):
        """Init."""
        super().__init__(venv)
        self.t = tstart
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)
        self._eval = False
        self._dones = np.zeros(self.num_envs, dtype=np.bool)

    def reset(self, force=True):
        """Reset."""
        obs = self.venv.reset(force=force)
        if force:
            self.rews[:] = 0
            self.lens[:] = 0
        else:
            self.rews[self._dones] = 0
            self.lens[self._dones] = 0
        self._dones[:] = False
        return obs

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.venv.step(action)
        if not self._eval:
            self.t += np.sum(np.logical_not(self._dones))
        for i, d in enumerate(self._dones):  # handle synced resets
            if not d:
                self.lens[i] += 1
                self.rews[i] += rews[i]
            else:
                assert dones[i]
        for i, done in enumerate(dones):
            if done and not self._dones[i]:
                if not self._eval:
                    logger.add_scalar('env/episode_length', self.lens[i],
                                      self.t, time.time())
                    logger.add_scalar('env/episode_reward', self.rews[i],
                                      self.t, time.time())
                self.lens[i] = 0
                self.rews[i] = 0.
        # log unwrapped episode stats if they exist
        if 'episode_info' in infos[0]:
            for i, info in enumerate(infos):
                epinfo = info['episode_info']
                if epinfo['done'] and not self._eval and not self._dones[i]:
                    logger.add_scalar('env/unwrapped_episode_length',
                                      epinfo['length'], self.t, time.time())
                    logger.add_scalar('env/unwrapped_episode_reward',
                                      epinfo['reward'], self.t, time.time())
        self._dones = np.logical_or(dones, self._dones)

        return obs, rews, dones, infos

    def step_wait(self):
        """Step wait."""
        return self.venv.step_wait()

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables logging and stops counting steps.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode counts steps and logs episode stats.
        """
        self._eval = False

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import SubprocVecEnv
    import gym

    class Test(unittest.TestCase):
        """Test."""

        def test_ep_info(self):
            """Test episode info."""
            env = gym.make('PongNoFrameskip-v4')
            env = EpisodeInfo(env)
            env.reset()
            rew = 0
            len = 0
            done = False
            while not done:
                _, r, done, info = env.step(env.action_space.sample())
                len += 1
                rew += r
                assert done == info['episode_info']['done']
                assert len == info['episode_info']['length']
                assert rew == info['episode_info']['reward']

        def test_vec_logger(self):
            """Test vec logger."""
            logger.configure('./.test')

            def env_fn(rank=0):
                env = gym.make('PongNoFrameskip-v4')
                env.seed(rank)
                return EpisodeInfo(env)

            def _env(rank):
                def _thunk():
                    return env_fn(rank=rank)
                return _thunk

            nenv = 4
            env = SubprocVecEnv([_env(i) for i in range(nenv)])
            env = VecEpisodeLogger(env)
            env.reset()
            for _ in range(5000):
                env.step(np.array([env.action_space.sample()
                                   for _ in range(nenv)]))
            state = env.state_dict()
            assert state['t'] == env.t
            state['t'] = 0
            env.load_state_dict(state)
            assert env.t == 0

            env.eval()
            env.reset()
            for _ in range(10):
                env.step(np.array([env.action_space.sample()
                                   for _ in range(nenv)]))
            assert env.t == 0
            assert np.allclose(env.lens, 10)
            env.train()
            for _ in range(10):
                env.step(np.array([env.action_space.sample()
                                   for _ in range(nenv)]))
            assert env.t == 10 * nenv
            assert np.allclose(env.lens, 20)
            logger.flush()
            shutil.rmtree('./.test')

    unittest.main()
