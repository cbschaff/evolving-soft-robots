"""Frame stack environment wrappers."""
from dl.rl import VecEnvWrapper, pack_space, unpack_space
from gym.spaces import Box, Tuple, Dict
import numpy as np
from dl import nest


class VecFrameStack(VecEnvWrapper):
    """Frame stack wrapper for vectorized environments."""

    def __init__(self, venv, k):
        """Init."""
        super().__init__(venv)
        self.k = k
        self.observation_space, self.frames = self._get_ob_space_and_frames(
                                                        self.observation_space)
        self._dones = np.zeros(self.num_envs, dtype=np.bool)

    def _get_ob_space_and_frames(self, ob_space):
        nested_spaces = unpack_space(ob_space)

        def _get_frames(space):
            if isinstance(space, Box):
                shp = space.shape
                return np.zeros((self.num_envs, self.k*shp[0], *shp[1:]),
                                dtype=space.dtype)
            else:
                raise ValueError("Observation Space must contain only Box, "
                                 f"Dict, or Tuple. Found {type(space)}.")

        def _get_stacked_space(space):
            return Box(
                low=np.repeat(space.low, self.k, axis=0),
                high=np.repeat(space.high, self.k, axis=0),
                dtype=space.dtype
            )

        frames = nest.map_structure(_get_frames, nested_spaces)
        stacked_spaces = nest.map_structure(_get_stacked_space, nested_spaces)
        return pack_space(stacked_spaces), frames

    def _add_new_observation(self, frames, ob):
        shape = ob.shape[1]
        frames[:, :-shape] = frames[:, shape:]
        frames[:, -shape:] = ob
        return frames

    def reset(self, force=True):
        """Reset."""
        ob = self.venv.reset(force=force)

        def _zero_frames(frames):
            if force:
                frames[:] = 0
            else:
                frames[self._dones] = 0
            return frames

        def _add_ob(item):
            return self._add_new_observation(*item)

        self.frames = nest.map_structure(_zero_frames, self.frames)
        self.frames = nest.map_structure(_add_ob,
                                         nest.zip_structure(self.frames, ob))
        self._dones[:] = False
        return nest.map_structure(lambda x: x.copy(), self.frames)

    def step(self, action):
        """Step."""
        ob, reward, done, info = self.venv.step(action)

        def _zero_frames(frames):
            for i, d in enumerate(done):
                if d:
                    frames[i] = 0
            return frames

        def _add_ob(item):
            return self._add_new_observation(*item)

        self.frames = nest.map_structure(_zero_frames, self.frames)
        self.frames = nest.map_structure(_add_ob,
                                         nest.zip_structure(self.frames, ob))
        ob = nest.map_structure(lambda x: x.copy(), self.frames)
        self._dones = np.logical_or(done, self._dones)
        return ob, reward, done, info

    def step_wait(self):
        """Step wait."""
        return self.venv.step_wait()

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import make_atari_env

    def make_env(nenv):
        """Create a training environment."""
        return make_atari_env("Pong", nenv)

    class NestedVecObWrapper(VecEnvWrapper):
        """Nest observations."""

        def __init__(self, venv):
            """Init."""
            super().__init__(venv)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space])

        def reset(self, force=True):
            """Reset."""
            ob = self.venv.reset(force=force)
            return (ob, ob)

        def step_wait(self):
            """Step."""
            ob, r, done, info = self.venv.step_wait()
            return (ob, ob), r, done, info

    class Test(unittest.TestCase):
        """Test."""

        def test_vec_frame_stack(self):
            """Test vec frame stack wrapper."""
            nenv = 2
            env = make_env(nenv)
            s = env.observation_space.shape
            env = VecFrameStack(env, 4)
            ob = env.reset()
            assert ob.shape == (nenv, 4*s[0], s[1], s[2])
            ob, _, _, _ = env.step(np.array([env.action_space.sample()
                                             for _ in range(nenv)]))
            assert ob.shape == (nenv, 4*s[0], s[1], s[2])
            while True:
                ob, _, done, _ = env.step(np.array([env.action_space.sample()
                                                    for _ in range(nenv)]))
                assert ob.shape == (nenv, 4*s[0], s[1], s[2])

                for i, d in enumerate(done):
                    if d:
                        assert np.allclose(ob[i, :-1], 0)
                if np.any(done):
                    break

        def test_nested_vec_frame_stack(self):
            """Test vec frame stack wrapper."""
            nenv = 2
            env = make_env(nenv)
            env = NestedVecObWrapper(env)
            env = NestedVecObWrapper(env)
            s = env.observation_space.spaces[0].spaces[0].shape
            env = VecFrameStack(env, 4)
            ob = env.reset()
            assert ob[0][0].shape == (nenv, 4*s[0], s[1], s[2])
            assert ob[0][1].shape == (nenv, 4*s[0], s[1], s[2])
            assert ob[1][0].shape == (nenv, 4*s[0], s[1], s[2])
            assert ob[1][1].shape == (nenv, 4*s[0], s[1], s[2])
            while True:
                ob, _, done, _ = env.step(np.array([env.action_space.sample()
                                                    for _ in range(nenv)]))
                assert ob[0][0].shape == (nenv, 4*s[0], s[1], s[2])
                assert ob[0][1].shape == (nenv, 4*s[0], s[1], s[2])
                assert ob[1][0].shape == (nenv, 4*s[0], s[1], s[2])
                assert ob[1][1].shape == (nenv, 4*s[0], s[1], s[2])

                for i, d in enumerate(done):
                    if d:
                        assert np.allclose(ob[0][0][i, :-1], 0)
                        assert np.allclose(ob[1][0][i, :-1], 0)
                        assert np.allclose(ob[0][1][i, :-1], 0)
                        assert np.allclose(ob[1][1][i, :-1], 0)
                if np.any(done):
                    break

    unittest.main()
