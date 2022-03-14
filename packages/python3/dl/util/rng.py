"""Util for python random number generation."""
import random
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = None


def seed(seed):
    """Set seeds."""
    global SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    SEED = seed


def get_state(cuda=True):
    """Get random state."""
    s = {}
    s['torch'] = torch.get_rng_state()
    if torch.cuda.is_available() and cuda:
        s['torch_cuda'] = torch.cuda.get_rng_state_all()
    s['numpy'] = np.random.get_state()
    s['random'] = random.getstate()
    return s


def set_state(state):
    """Set random state."""
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


if __name__ == '__main__':
    import unittest

    class TestRandom(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            seed(0)
            state = get_state()
            r1 = torch.rand([10])
            r2 = np.random.rand(10)
            r3 = [random.random() for _ in range(10)]

            set_state(state)
            r1_copy = torch.rand([10])
            r2_copy = np.random.rand(10)
            r3_copy = [random.random() for _ in range(10)]

            assert np.allclose(r1.numpy(), r1_copy.numpy())
            assert np.allclose(r2, r2_copy)
            assert np.allclose(r3, r3_copy)

    unittest.main()
