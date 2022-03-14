"""Environment wrappers."""
from gym import ActionWrapper, spaces
import numpy as np


class ActionNormWrapper(ActionWrapper):
    """Normalize the range of continuous action spaces."""

    def __init__(self, env):
        """Init."""
        super().__init__(env)
        self.action_space = self.make_action_space(self.env.action_space)

    def make_action_space(self, ac_space):
        if isinstance(ac_space, spaces.Box):
            return spaces.Box(-np.ones_like(ac_space.low),
                              np.ones_like(ac_space.high), dtype=ac_space.dtype)
        elif isinstance(ac_space, spaces.Tuple):
            return spaces.Tuple([
                self.make_action_space(a_s) for a_s in ac_space
            ])
        elif isinstance(ac_space, spaces.Dict):
            return spaces.Dict({
                k: self.make_action_space(ac_space[k]) for k in ac_space.spaces
            })
        else:
            return ac_space

    def norm_action(self, ac_space, ac):
        if isinstance(ac_space, spaces.Box):
            return (ac + 1) * (ac_space.high - ac_space.low) / 2 + ac_space.low
        elif isinstance(ac_space, spaces.Tuple):
            return tuple([
                self.norm_action(a_s, a) for a_s, a in zip(ac_space, ac)
            ])
        elif isinstance(ac_space, spaces.Dict):
            return {
                k: self.norm_action(ac_space[k], ac[k]) for k in ac_space.spaces
            }
        else:
            return ac

    def action(self, action):
        return self.norm_action(self.env.action_space, action)


if __name__ == '__main__':

    import unittest
    import gym

    class StackActionWrapper(ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.action_space = spaces.Tuple([
                env.action_space,
                spaces.Dict({
                    'ac1': env.action_space,
                    'ac2': spaces.Discrete(5)
                })
            ])

        def action(self, action):
            return action[0]

    class TestActionNormWrapper(unittest.TestCase):
        """Test DummyVecEnv"""

        def test(self):
            env = ActionNormWrapper(StackActionWrapper(gym.make('CarRacing-v0')))

            env.reset()
            action = env.action_space.sample()
            print(action)
            print(env.action(action))
            print(env.action_space)
            print(env.env.action_space)
            print(env.env.env.action_space.low)
            print(env.env.env.action_space.high)

    unittest.main()
