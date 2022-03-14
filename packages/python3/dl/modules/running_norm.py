"""Running normalization.

https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
"""
import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Normalize with running estimates of mean and variance."""

    def __init__(self, ob_shape, eps=1e-5):
        """Init."""
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(size=ob_shape, dtype=torch.float),
                                 requires_grad=False)
        self.var = nn.Parameter(torch.ones(size=ob_shape, dtype=torch.float),
                                requires_grad=False)
        self.count = nn.Parameter(torch.zeros(size=[1], dtype=torch.float),
                                  requires_grad=False)
        self.eps = eps
        self.std = torch.sqrt(self.var)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        self.std = self.std.to(self.var.device)
        return out

    def forward(self, ob):
        """Forward."""
        return (ob.float() - self.mean) / (self.std + self.eps)

    def update(self, batch_mean, batch_var, batch_count):
        """Update meean and var."""
        delta = batch_mean - self.mean
        new_count = (self.count + batch_count)
        new_mean = self.mean + delta * (batch_count / new_count)
        new_var = self.count * self.var + batch_count * batch_var
        new_var += (delta**2) * self.count * batch_count / new_count
        new_var /= new_count
        self.count.copy_(new_count)
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.std = torch.sqrt(self.var)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.std = torch.sqrt(self.var)


if __name__ == '__main__':
    import unittest

    class TestRN(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            ron = RunningNorm([5, 4])

            ob = torch.ones([5, 4])
            assert torch.allclose(ob, ron(ob), atol=2e-5)  # eps is 1e-5

            def var(i):
                return torch.var(torch.arange(1, i+1).float(), unbiased=False)

            for i in range(1, 6):
                # obs arriving in batches of 2: ((2*i-1) * ob, 2*i * ob)
                batch_mean = (2*i - 0.5) * ob
                batch_var = 0.25 * ob
                batch_count = 2

                ron.update(batch_mean, batch_var, batch_count)
                assert torch.allclose(ron.mean, (i+0.5) * ob, atol=2e-5)
                assert torch.allclose(ron.var, var(2*i) * ob, atol=2e-5)
                assert ron.count == 2*i
                assert torch.allclose(
                    ron(ob[None]),
                    (ob[None] - (i+0.5) * ob[None]) / torch.sqrt(
                        var(2*i) * ob[None]), atol=2e-5)

            assert ron(ob[None]).shape == ob[None].shape

            assert not ron.mean.requires_grad
            assert not ron.var.requires_grad
            assert not ron.count.requires_grad

            assert len(ron.state_dict()) == 3

    unittest.main()
