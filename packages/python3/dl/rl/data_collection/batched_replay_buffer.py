from dl import nest
import numpy as np


class BatchedReplayBuffer(object):
    def __init__(self, *buffers):
        self.n = len(buffers)
        self.buffers = buffers
        self.num_in_buffer = 0
        self.size = sum([buf.size for buf in self.buffers])

    def _update_num_in_buffer(self):
        self.num_in_buffer = sum([buf.num_in_buffer for buf in self.buffers])

    def _get_sizes(self, batch_size):
        sizes = [batch_size // self.n for _ in range(self.n)]
        for i in range(batch_size - self.n * (batch_size // self.n)):
            sizes[i] += 1
        return sizes

    def can_sample(self, batch_size):
        sizes = self._get_sizes(batch_size)
        for buf, s in zip(self.buffers, sizes):
            if not buf.can_sample(s):
                return False
        return True

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
        sizes = self._get_sizes(batch_size)
        batches = [buffer.sample(s)
                   for buffer, s in zip(self.buffers, sizes) if s > 0]
        return nest.map_structure(
            lambda x: np.concatenate(x, axis=0), nest.zip_structure(*batches)
        )

    def encode_recent_observation(self):
        obs = [buf.encode_recent_observation() for buf in self.buffers]
        return nest.map_structure(np.stack, nest.zip_structure(*obs))

    def store_observation(self, obs):
        inds = []
        for i, buf in enumerate(self.buffers):
            inds.append(buf.store_observation(
                                    nest.map_structure(lambda x: x[i], obs)))
        self._update_num_in_buffer()
        return inds

    def store_effect(self, idx, step_data):
        for i, buf in enumerate(self.buffers):
            buf.store_effect(
                idx[i], nest.map_structure(lambda x: x[i], step_data)
            )

    def env_reset(self):
        for buf in self.buffers:
            buf.env_reset()

    def state_dict(self):
        state = {}
        for i, buf in enumerate(self.buffers):
            state[i] = buf.state_dict()
        return state

    def load_state_dict(self, state_dict):
        for i in range(self.n):
            self.buffers[i].load_state_dict(state_dict[i])
        self._update_num_in_buffer()


if __name__ == '__main__':
    from dl.rl import make_atari_env
    from dl.rl.data_collection.buffer import ReplayBuffer
    import unittest

    class TestBatchedBuffer(unittest.TestCase):
        """Test."""

        def test(self):
            nenv = 5
            buffers = [ReplayBuffer(1000, 1) for _ in range(nenv)]
            buffer = BatchedReplayBuffer(*buffers)
            env = make_atari_env('Pong', nenv=nenv)
            ob = env.reset()
            for _ in range(1000):
                ind = buffer.store_observation(ob)
                action = np.array([env.action_space.sample() for _ in range(nenv)])
                ob, r, done, _ = env.step(action)
                buffer.store_effect(ind, {'done': done, 'action': action,
                                          'reward': r})
                if np.all(done):
                    ob = env.reset()

            batch = buffer.sample(4)
            assert batch['obs'].shape == (4, 1, 84, 84)
            assert batch['next_obs'].shape == (4, 1, 84, 84)
            assert batch['reward'].shape == (4,)
            assert batch['done'].shape == (4,)
            assert batch['action'].shape == (4,)

            state = buffer.state_dict()
            buffer.load_state_dict(state)

    unittest.main()
