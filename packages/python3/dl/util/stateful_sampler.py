"""Modified from https://github.com/pytorch/pytorch/issues/11813."""

import numpy as np
import torch.utils.data.sampler as TorchSampler

# the initial permutation is preserved across restarts


class StatefulSampler(TorchSampler.Sampler):
    """Allows epoch progess to be saved and loaded."""

    # vanilla list is orders of magnitude faster for indexing
    # while np.array/torch.Tensor is 3x more storage efficient
    def __init__(self, data_source, shuffle=True):
        """Init."""
        self.shuffle = shuffle
        self.dset = data_source
        self._init_indices()

    def _init_indices(self):
        indices = None
        if self.shuffle:
            indices = np.random.permutation(len(self.dset))
        else:
            indices = range(len(self.dset))
        self.indices = list(indices)
        self.iter_counter = 0

    def __len__(self):
        """__len__."""
        return len(self.dset)

    def __iter__(self):
        """__iter__."""
        return self

    def __next__(self):
        """__next__."""
        if self.iter_counter == len(self.indices):
            self._init_indices()
            raise StopIteration()
        else:
            elem = self.indices[self.iter_counter]
            self.iter_counter += 1
            return elem

    def load_state_dict(self, state_dict):
        """Load sampler state."""
        self.indices = list(state_dict['indices'])
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, loader_iter=None):
        """Save sampler state."""
        prefetched_num = 0
        if loader_iter and loader_iter._num_workers > 0:
            batch_size = loader_iter._index_sampler.batch_size
            prefetched_num = \
                (loader_iter._send_idx - loader_iter._rcvd_idx) * batch_size
        return {
            'indices': np.array(self.indices),
            'iter_counter': self.iter_counter - prefetched_num
        }


if __name__ == '__main__':
    import unittest
    from torch.utils.data import Dataset, DataLoader

    class TestSampler(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            class Dset(Dataset):
                def __len__(self):
                    return 100

                def __getitem__(self, i):
                    return i

            data = Dset()
            sampler = StatefulSampler(data, shuffle=True)
            dl = DataLoader(data, sampler=sampler, batch_size=2, num_workers=2)
            used_inds = []
            diter = dl.__iter__()
            for _ in range(10):
                batch = diter.__next__()
                used_inds.extend(batch.tolist())
            state = sampler.state_dict(diter)

            sampler = StatefulSampler(data, shuffle=True)
            sampler.load_state_dict(state)
            dl = DataLoader(data, sampler=sampler, batch_size=2, num_workers=2)
            for batch in dl:
                used_inds.extend(batch.tolist())
            assert len(used_inds) == 100
            assert len(set(used_inds)) == 100

    unittest.main()
