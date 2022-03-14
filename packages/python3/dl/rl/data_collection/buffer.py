"""Replay buffer.

This file is apdated from
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
Edits were made to make the buffer more flexible in the data it could store.
"""
import numpy as np
import random
from dl import nest
from functools import partial


def sample_n_unique(sampling_f, n):
    """Sample n unique outputs from sampling_f.

    Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(object):
    """Replay Buffer."""

    def __init__(self, size, obs_history_len):
        """Memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each obs once rather than k times
              even if every observation normally consists of k last obs
            - store obs_t and obs_(t+1) in the same buffer.
        For the typical use case in Atari Deep RL buffer with 1M obs the
        total memory footprint of this buffer is
        10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning obs of zeros at the beginning
        of the episode, when there is less obs than `obs_history_len`,
        is acceptable.

        Warning! Observations are concatenated along the first dimension.
        For images, this means that the data format should be (C,H,W).
        Parameters

        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        obs_history_len: int
            Number of memories to be retried for each observation.

        """
        self.size = size
        self.obs_history_len = obs_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.data = {}
        self.required_keys = ['action', 'reward', 'done']

    def _init_obs_data(self, obs):
        dtype = np.float32 if obs.dtype == np.float64 else obs.dtype
        return np.empty([self.size] + list(obs.shape), dtype=dtype)

    def _init_replay_data(self, step_data):
        for k in self.required_keys:
            if k not in step_data:
                raise ValueError("action, reward, and done must be keys in the"
                                 "dict passed to buffer.store_effect.")

        def _make_buffer(x):
            x = np.asarray(x)
            return np.empty([self.size] + list(x.shape), dtype=np.float32)
        self.data = nest.map_structure(_make_buffer, step_data)

    def can_sample(self, batch_size):
        """Check if a batch_size can be sampled.

        Returns true if `batch_size` different transitions can be sampled
        from the buffer.
        """
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        batch = nest.map_structure(lambda x: x[idxes], self.data)

        def _batch(obs):
            return np.concatenate([ob[np.newaxis, :] for ob in obs], 0)

        obs = [self._encode_observation(idx) for idx in idxes]
        batch['obs'] = nest.map_structure(_batch, nest.zip_structure(*obs))
        next_obs = [self._encode_observation(idx + 1) for idx in idxes]
        batch['next_obs'] = nest.map_structure(_batch,
                                               nest.zip_structure(*next_obs))
        return batch

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        batched data: dict
            a dictionary containing batched observations, next_observations,
            action, reward, done, and other data stored in the replay buffer.

        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(
            lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `obs_history_len` obs.

        Returns
        -------
        observation: nest of np.array

        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):

        def _encode(ob, idx):
            end_idx = idx + 1  # make noninclusive
            start_idx = end_idx - self.obs_history_len
            # if there weren't enough obs ever in the buffer for context
            if start_idx < 0 and self.num_in_buffer != self.size:
                start_idx = 0
            for idx in range(start_idx, end_idx - 1):
                if self.data['done'][idx % self.size]:
                    start_idx = idx + 1
            missing_context = self.obs_history_len - (end_idx - start_idx)
            # if zero padding is needed for missing context
            # or we are on the boundry of the buffer
            if start_idx < 0 or missing_context > 0:
                obs = [
                    np.zeros_like(ob[0]) for _ in range(missing_context)
                ]
                for idx in range(start_idx, end_idx):
                    obs.append(ob[idx % self.size])
                return np.concatenate(obs, 0)
            else:
                # this optimization has potential to saves about 30% compute
                # time
                s = ob.shape[2:]
                return ob[start_idx:end_idx].reshape(-1, *s)

        return nest.map_structure(partial(_encode, idx=idx), self.obs)

    def store_observation(self, obs):
        """Store a single observation in the buffer at the next available index.

        Overwrites old observations if necessary.
        Parameters
        ----------
        obs: nest of np.array

        Returns
        -------
        idx: int
            Index at which the obs is stored. To be used for `store_effect`
            later.

        """
        if self.obs is None:
            self.obs = nest.map_structure(self._init_obs_data, obs)

        def _store_ob(item):
            buffer, ob = item
            buffer[self.next_idx] = ob

        nest.map_structure(_store_ob, nest.zip_structure(self.obs, obs))

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, step_data):
        """Store effects of action taken after obeserving obs stored at idx.

        The reason `store_observation` and `store_effect` is broken
        up into two functions is so that one can call
        `encode_recent_observation` in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recent observation
            (returned by `store_observation`).
        data: dict
            The data to store in the buffer.
        """
        if self.data == {}:
            self._init_replay_data(step_data)
        if not nest.has_same_structure(self.data, step_data):
            raise ValueError("The data passed to ReplayBuffer must the same"
                             " at all time steps.")

        def _insert(item):
            buffer, x = item
            buffer[idx] = x
        nest.map_structure(_insert, nest.zip_structure(self.data, step_data))

    def env_reset(self):
        """Update buffer based on early environment resest.

        Allow environment resets for the most recent transition after it has
        been stored. This is useful when loading a saved replay buffer.
        """
        if self.num_in_buffer > 0:
            self.data['done'][(self.next_idx-1) % self.size] = True

    def state_dict(self):
        """State dict."""
        return {
            'obs': self.obs,
            'data': self.data,
            'num_in_buffer': self.num_in_buffer,
            'next_idx': self.next_idx,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        try:
            self.obs = state_dict['obs'].item()
        except Exception:
            self.obs = state_dict['obs']
        if hasattr(state_dict['data'], 'item'):
            self.data = state_dict['data'].item()
        else:
            self.data = state_dict['data']
        self.num_in_buffer = state_dict['num_in_buffer']
        self.next_idx = state_dict['next_idx']


"""
Unit Tests
"""


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import make_atari_env as atari_env
    from gym import ObservationWrapper
    from gym.spaces import Tuple

    class NestedObWrapper(ObservationWrapper):
        """Nest observations."""

        def __init__(self, env):
            """Init."""
            super().__init__(env)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space])

        def observation(self, observation):
            """Duplicate observation."""
            return (observation, observation)

    class TestBuffer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            buffer = ReplayBuffer(10, 4)
            env = atari_env('Pong').envs[0]
            init_obs = env.reset()
            idx = buffer.store_observation(init_obs)
            assert np.allclose(buffer.encode_recent_observation()[:-3], 0)
            for i in range(10):
                ac = env.action_space.sample()
                obs, r, done, _ = env.step(ac)
                data = {'action': ac, 'reward': r, 'done': done, 'key1': 0}
                buffer.store_effect(idx, data)
                idx = buffer.store_observation(obs)

            # Check sample shapes
            s = buffer.sample(2)
            assert s['obs'].shape == (2, 4, 84, 84)
            assert s['next_obs'].shape == (2, 4, 84, 84)
            assert s['key1'].shape == (2,)
            s = buffer._encode_sample([4, 5])
            # Check observation stacking
            assert np.allclose(s['obs'][0][3], s['next_obs'][0][2])
            assert np.allclose(s['obs'][0][2], s['next_obs'][0][1])
            assert np.allclose(s['obs'][0][1], s['next_obs'][0][0])

            # Check sequential samples
            assert np.allclose(s['obs'][0][3], s['obs'][1][2])

            # Check for wrap around when buffer is full
            s = buffer._encode_sample([0])
            assert not np.allclose(s['obs'][0][:-3], 0)

            # Check env reset
            buffer.env_reset()
            assert buffer.data['done'][buffer.next_idx - 1 % buffer.size]

            # Check saving and loading
            state = buffer.state_dict()
            buffer2 = ReplayBuffer(10, 4)
            buffer2.load_state_dict(state)

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                assert np.allclose(s1[k], s2[k])

            for i in range(10):
                ac = env.action_space.sample()
                obs, r, done, _ = env.step(ac)
                data = {'action': ac, 'reward': r, 'done': done, 'key1': 0}
                buffer.store_effect(idx, data)
                buffer2.store_effect(idx, data)
                idx = buffer.store_observation(obs)
                idx2 = buffer2.store_observation(obs)
                assert idx == idx2

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                assert np.allclose(s1[k], s2[k])

        def test_nested_obs(self):
            """Test."""
            buffer = ReplayBuffer(10, 4)
            env = atari_env('Pong').envs[0]
            env = NestedObWrapper(env)
            env = NestedObWrapper(env)
            init_obs = env.reset()
            idx = buffer.store_observation(init_obs)
            assert np.allclose(buffer.encode_recent_observation()[:-3], 0)
            for i in range(10):
                ac = env.action_space.sample()
                obs, r, done, _ = env.step(ac)
                data = {'action': ac, 'reward': r, 'done': done, 'key1': 0}
                buffer.store_effect(idx, data)
                idx = buffer.store_observation(obs)

            # Check sample shapes
            s = buffer.sample(2)
            assert s['obs'][0][0].shape == (2, 4, 84, 84)
            assert s['obs'][0][1].shape == (2, 4, 84, 84)
            assert s['obs'][1][0].shape == (2, 4, 84, 84)
            assert s['obs'][1][1].shape == (2, 4, 84, 84)
            assert s['next_obs'][0][0].shape == (2, 4, 84, 84)
            assert s['next_obs'][0][1].shape == (2, 4, 84, 84)
            assert s['next_obs'][1][0].shape == (2, 4, 84, 84)
            assert s['next_obs'][1][1].shape == (2, 4, 84, 84)
            assert s['key1'].shape == (2,)
            s = buffer._encode_sample([4, 5])
            # Check observation stacking
            assert np.allclose(s['obs'][0][0][0][3], s['next_obs'][0][0][0][2])
            assert np.allclose(s['obs'][0][0][0][2], s['next_obs'][0][0][0][1])
            assert np.allclose(s['obs'][0][0][0][1], s['next_obs'][0][0][0][0])

            assert np.allclose(s['obs'][1][0][0][3], s['next_obs'][1][0][0][2])
            assert np.allclose(s['obs'][1][0][0][2], s['next_obs'][1][0][0][1])
            assert np.allclose(s['obs'][1][0][0][1], s['next_obs'][1][0][0][0])

            # Check sequential samples
            assert np.allclose(s['obs'][0][0][0][3], s['obs'][0][0][1][2])
            assert np.allclose(s['obs'][0][1][0][3], s['obs'][0][1][1][2])
            assert np.allclose(s['obs'][1][0][0][3], s['obs'][1][0][1][2])
            assert np.allclose(s['obs'][1][1][0][3], s['obs'][1][1][1][2])

            # Check for wrap around when buffer is full
            s = buffer._encode_sample([0])
            assert not np.allclose(s['obs'][0][0][0][:-3], 0)
            assert not np.allclose(s['obs'][0][1][0][:-3], 0)
            assert not np.allclose(s['obs'][1][0][0][:-3], 0)
            assert not np.allclose(s['obs'][1][1][0][:-3], 0)

            # Check env reset
            buffer.env_reset()
            assert buffer.data['done'][buffer.next_idx - 1 % buffer.size]

            # Check saving and loading
            state = buffer.state_dict()
            buffer2 = ReplayBuffer(10, 4)
            buffer2.load_state_dict(state)

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                if k == 'obs':
                    assert np.allclose(s1[k][0][0], s2[k][0][0])
                    assert np.allclose(s1[k][0][1], s2[k][0][1])
                    assert np.allclose(s1[k][1][0], s2[k][1][0])
                    assert np.allclose(s1[k][1][1], s2[k][1][1])
                else:
                    assert np.allclose(s1[k], s2[k])

            for i in range(10):
                ac = env.action_space.sample()
                obs, r, done, _ = env.step(ac)
                data = {'action': ac, 'reward': r, 'done': done, 'key1': 0}
                buffer.store_effect(idx, data)
                buffer2.store_effect(idx, data)
                idx = buffer.store_observation(obs)
                idx2 = buffer2.store_observation(obs)
                assert idx == idx2

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                if k == 'obs':
                    assert np.allclose(s1[k][0][0], s2[k][0][0])
                    assert np.allclose(s1[k][0][1], s2[k][0][1])
                    assert np.allclose(s1[k][1][0], s2[k][1][0])
                    assert np.allclose(s1[k][1][1], s2[k][1][1])
                else:
                    assert np.allclose(s1[k], s2[k])

    unittest.main()
