"""Set up gin.configurables for environment creation."""
from dl.rl.util import atari_wrappers
from dl.rl.envs.logging_wrappers import EpisodeInfo
from dl.rl.envs.misc_wrappers import ImageTranspose
from dl.rl.envs import VecFrameStack
from dl.rl.envs import VecObsNormWrapper
from dl.rl.envs import SubprocVecEnv
from dl.rl.envs import DummyVecEnv
from dl.rl.envs import ActionNormWrapper
import gin
import gym
from gym.wrappers import TimeLimit


class StepOnEndOfLifeEnv(gym.Wrapper):
    """Do a no-op step after loss of life in atari games.

    When not using the episodic life wrapper, the environment might not
    continue the game until the "fire" button is pressed.
    """

    def __init__(self, env):
        """Init."""
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def reset(self):
        """Reset."""
        obs = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action):
        """Step."""
        obs, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            if 'FIRE' in self.env.unwrapped.get_action_meanings():
                obs, _, _, _ = self.env.step(1)
            else:
                obs, _, _, _ = self.env.step(0)
        self.lives = lives
        return obs, reward, done, info


@gin.configurable(blacklist=['nenv'])
def make_atari_env(game_name, nenv=1, seed=0, sticky_actions=True,
                   timelimit=True, noop=False, frameskip=4, episode_life=False,
                   clip_rewards=True, frame_stack=1, scale=False,
                   timelimit_maxsteps=None):
    """Create an Atari environment."""
    id = game_name + 'NoFrameskip'
    id += '-v0' if sticky_actions else '-v4'

    def _env(rank):
        def _thunk():
            env = gym.make(id)
            if not timelimit:
                env = env.env
            elif timelimit_maxsteps:
                env = TimeLimit(env.env, timelimit_maxsteps)
            assert 'NoFrameskip' in env.spec.id
            if noop:
                env = atari_wrappers.NoopResetEnv(env, noop_max=30)
            env = atari_wrappers.MaxAndSkipEnv(env, skip=frameskip)
            env = StepOnEndOfLifeEnv(env)
            env = EpisodeInfo(env)
            env.seed(seed + rank)
            env = atari_wrappers.wrap_deepmind(
                env, episode_life=episode_life, clip_rewards=clip_rewards,
                frame_stack=False, scale=scale)
            env = ImageTranspose(env)
            return env
        return _thunk

    if nenv > 1:
        env = SubprocVecEnv([_env(i) for i in range(nenv)], context='fork')
    else:
        env = DummyVecEnv([_env(0)])

    if frame_stack > 1:
        env = VecFrameStack(env, frame_stack)
    return env


@gin.configurable(blacklist=['nenv'])
def make_env(env_id, nenv=1, seed=0, norm_observations=False,
             norm_actions=False):
    """Create an environment."""
    def _env(rank):
        def _thunk():
            env = gym.make(env_id)
            if norm_actions:
                env = ActionNormWrapper(env)
            env = EpisodeInfo(env)
            env.seed(seed + rank)
            return env
        return _thunk

    if nenv > 1:
        env = SubprocVecEnv([_env(i) for i in range(nenv)], context='fork')
    else:
        env = DummyVecEnv([_env(0)])

    if norm_observations:
        env = VecObsNormWrapper(env)
    return env


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestEnvFns(unittest.TestCase):
        """Test."""

        def test_atari(self):
            """Test atari fn."""
            env = make_atari_env('Pong', 1, 0, sticky_actions=True)
            assert env.spec.id == 'PongNoFrameskip-v0'
            assert env.observation_space.shape == (1, 84, 84)
            assert env.reset().shape == (1, 1, 84, 84)
            action = np.array([env.action_space.sample()])
            assert 'episode_info' in env.step(action)[3][0]
            env.close()
            env = make_atari_env('Breakout', 1, 0, sticky_actions=False)
            assert env.spec.id == 'BreakoutNoFrameskip-v4'
            assert env.observation_space.shape == (1, 84, 84)
            assert env.reset().shape == (1, 1, 84, 84)
            action = np.array([env.action_space.sample()])
            assert 'episode_info' in env.step(action)[3][0]
            env.close()
            env = make_atari_env('Breakout', 1, 0, sticky_actions=False,
                                 frame_stack=4)
            assert env.spec.id == 'BreakoutNoFrameskip-v4'
            assert env.observation_space.shape == (4, 84, 84)
            assert env.reset().shape == (1, 4, 84, 84)
            action = np.array([env.action_space.sample()])
            assert 'episode_info' in env.step(action)[3][0]
            env.close()

        def test_env(self):
            """Test env fn."""
            env = make_env('CartPole-v1', 1)
            env.reset()
            action = np.array([env.action_space.sample()])
            assert 'episode_info' in env.step(action)[3][0]

    unittest.main()
