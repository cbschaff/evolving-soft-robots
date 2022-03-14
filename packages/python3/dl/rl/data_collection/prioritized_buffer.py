"""Prioritized Replay Buffer."""
from dl.rl.data_collection import ReplayBuffer
from dl.rl.util.segment_tree import SumSegmentTree, MinSegmentTree
from dl.rl.data_collection.buffer import sample_n_unique
import random
import numpy as np


class PrioritizedReplayBuffer(object):
    """Prioritized Reply Buffer.

    Implementation of https://arxiv.org/abs/1511.05952, using the
    "proportional sampling" algorithm.
    """

    def __init__(self, buffer, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__

        """
        assert alpha >= 0
        self._alpha = alpha
        self.buffer = buffer
        self.size = self.buffer.size
        self.num_in_buffer = self.buffer.num_in_buffer

        it_capacity = 1
        while it_capacity < buffer.size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def can_sample(self, batch_size):
        """Check if batch_size things can be sampled."""
        return self.buffer.can_sample(batch_size)

    def _sample_proportional(self):
        mass = random.random() * self._it_sum.sum(
            0, self.buffer.num_in_buffer - 2)
        return self._it_sum.find_prefixsum_idx(mass)

    def _encode_sample(self, idxes):
        return self.buffer._encode_sample(idxes)

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        batched data: dict
            a dictionary containing batched data. See buffer.py for details.
            Two keys are added to the dict:
                weights: np.array
                    Array of shape (batch_size,) and dtype np.float32
                    denoting importance weight of each sampled transition
                idxes: np.array
                    Array of shape (batch_size,) and dtype np.int32
                    idexes in buffer of sampled experiences

        """
        assert beta > 0
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(self._sample_proportional, batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.buffer.num_in_buffer) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.buffer.num_in_buffer) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        encoded_sample = self.buffer._encode_sample(idxes)
        encoded_sample['weights'] = weights
        encoded_sample['idxes'] = idxes
        return encoded_sample

    def encode_recent_observation(self):
        """Get last observation."""
        return self.buffer.encode_recent_observation()

    def store_observation(self, obs):
        """Store an observation."""
        idx = self.buffer.store_observation(obs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        self.num_in_buffer = self.buffer.num_in_buffer
        return idx

    def store_effect(self, *args, **kwargs):
        """Store effect of action."""
        return self.buffer.store_effect(*args, **kwargs)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.

        """
        priorities = np.minimum(priorities, self._max_priority)
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.buffer.num_in_buffer - 1
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def env_reset(self):
        """Modify buffer for early environment reset."""
        self.buffer.env_reset()

    def state_dict(self):
        """State dict."""
        state = self.buffer.state_dict()
        state['_max_priority'] = self._max_priority
        state['_it_sum'] = self._it_sum
        state['_it_min'] = self._it_min
        return state

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.buffer.load_state_dict(state_dict)
        self._max_priority = state_dict['_max_priority']
        if hasattr(state_dict['_it_sum'], 'item'):
            self._it_sum = state_dict['_it_sum'].item()
        else:
            self._it_sum = state_dict['_it_sum']
        if hasattr(state_dict['_it_min'], 'item'):
            self._it_min = state_dict['_it_min'].item()
        else:
            self._it_min = state_dict['_it_min']


"""
Unit Tests
"""


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import make_atari_env as atari_env

    class TestPrioritizedBuffer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            buffer = ReplayBuffer(10, 4)
            buffer = PrioritizedReplayBuffer(buffer, alpha=0.5)
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
            s = buffer.sample(2, beta=1.)
            assert len(s.keys()) == 8
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

            # check priorities
            buffer.update_priorities([4, 5], [0.5, 2])
            assert buffer._it_sum[4] == 0.5 ** buffer._alpha
            assert buffer._it_sum[5] == 1.0 ** buffer._alpha
            assert buffer._it_min[4] == 0.5 ** buffer._alpha
            assert buffer._it_min[5] == 1.0 ** buffer._alpha
            assert buffer._max_priority == 1.0

            # Check for wrap around when buffer is full
            s = buffer._encode_sample([0])
            assert not np.allclose(s['obs'][0][:-3], 0)

            # Check saving and loading
            state = buffer.state_dict()
            buffer2 = ReplayBuffer(10, 4)
            buffer2 = PrioritizedReplayBuffer(buffer2, alpha=0.5)
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
                assert buffer._max_priority == buffer2._max_priority

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                assert np.allclose(s1[k], s2[k])

            buffer.update_priorities([4, 5], [0.1, 0.9])
            buffer2.update_priorities([4, 5], [0.1, 0.9])
            assert buffer._max_priority == buffer2._max_priority
            assert buffer._it_sum[4] == buffer2._it_sum[4]
            assert buffer._it_sum[5] == buffer2._it_sum[5]

    unittest.main()
