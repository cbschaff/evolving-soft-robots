from dl.rl.data_collection import ReplayBuffer
from dl.rl.data_collection.buffer import sample_n_unique
from dl import nest
import random
import numpy as np


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, size, obs_history_len, n):
        super().__init__(size, obs_history_len)
        self.n = n

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
        assert self.can_sample(batch_size + self.n - 1)
        idxes = sample_n_unique(
            lambda: random.randint(0, self.num_in_buffer - (self.n + 1)),
            batch_size)
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):

        def _batch(obs):
            return np.concatenate([ob[np.newaxis, :] for ob in obs], 0)

        batch = {}
        batch['obs'] = []
        for k in self.data.keys():
            batch[k] = []

        for i in range(self.n):
            obs = [self._encode_observation(idx + i) for idx in idxes]
            batch['obs'].append(nest.map_structure(_batch,
                                                   nest.zip_structure(*obs)))
            for k in self.data.keys():
                batch[k].append(self.data[k][[idx + i for idx in idxes]])
        obs = [self._encode_observation(idx + self.n) for idx in idxes]
        batch['obs'].append(nest.map_structure(_batch,
                                               nest.zip_structure(*obs)))
        return batch


if __name__ == '__main__':
    from dl.rl import make_atari_env

    buffer = NStepReplayBuffer(1000, 1, 5)
    env = make_atari_env('Pong', nenv=1)
    ob = env.reset()
    for _ in range(buffer.size):
        ind = buffer.store_observation(ob[0])
        action = np.array([env.action_space.sample()])
        ob, r, done, _ = env.step(action)
        buffer.store_effect(ind, {'done': done[0], 'action': action[0],
                                  'reward': r[0]})
        if np.all(done):
            ob = env.reset()

    import cv2
    batch = buffer.sample(4)
    for i in range(4):
        for j in range(5):
            cv2.imshow('ob', batch['obs'][j][i].transpose(1, 2, 0))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
