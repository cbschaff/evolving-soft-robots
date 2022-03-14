"""Some simple networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


@gin.configurable
class FeedForwardNet(nn.Module):
    """Feed forward network."""

    def __init__(self, in_shape, units=[], activation_fn=F.relu,
                 activate_last=False):
        """Init.

        Creates a simple feed forward net.
        Example:
            net = FeedForwardNet(16, [32,32,1])
        Args:
            in_shape (int):
                number of channels of the input to the network.
            units (list):
                The number of units in each layer. The length of the list
                denotes the number of layers.
            activation_fn (callable):
                The activation function to apply after each layer.
            activate_last (bool):
                Whether or not to apply activation_fn to the output of the last
                layer.

        """
        super().__init__()
        ni = in_shape
        for i, no in enumerate(units):
            setattr(self, f'fc{i}', nn.Linear(ni, no))
            ni = no
        self.activation_fn = activation_fn
        self.nlayers = len(units)
        self.activate_last = activate_last

    def forward(self, x):
        """Forward."""
        for i in range(self.nlayers-1):
            x = self.activation_fn(getattr(self, f'fc{i}')(x))
        x = getattr(self, f'fc{self.nlayers-1}')(x)
        if self.activate_last:
            return self.activation_fn(x)
        return x


@gin.configurable
class Conv2dNet(nn.Module):
    """Convolutional network."""

    def __init__(self, in_channels=3, convs=[], activation_fn=F.relu,
                 activate_last=False):
        """Init.

        Creates a simple conv net.
        Example:
            net = Conv2dNet(3, [(16,3), (32,3,2), (64,3,1,1)])
        Args:
            in_channels (int):
                number of channels of the input to the network.
            convs      (list):
                Describes the conv layers. Each element of the
                list should contain the args for torch.nn.Conv2d
                in order. At minimum, the first two args must be
                specified (out_channels and kernel_size).
            activation_fn (callable):
                The activation function to apply after
                               each layer.
            activate_last (bool):
                Whether or not to apply activation_fn to the output of the last
                layer.

        """
        super().__init__()
        ci = in_channels
        for i, params in enumerate(convs):
            setattr(self, f'conv2d{i}', nn.Conv2d(ci, *params))
            ci = params[0]
        self.activation_fn = activation_fn
        self.nlayers = len(convs)
        self.activate_last = activate_last

    def forward(self, x):
        """Forward."""
        for i in range(self.nlayers-1):
            x = self.activation_fn(getattr(self, f'conv2d{i}')(x))
        x = getattr(self, f'conv2d{self.nlayers-1}')(x)
        if self.activate_last:
            return self.activation_fn(x)
        return x


if __name__ == '__main__':

    import os
    import unittest
    from dl import load_config

    class TestCore(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            config = \
                "FeedForwardNet.activation_fn = @F.relu \n \
                FeedForwardNet.activate_last = False \n \
                FeedForwardNet.units = [64,64,1] \n \
                FeedForwardNet.in_shape = 2304 \n \
                Conv2dNet.activation_fn = @F.relu \n \
                Conv2dNet.activate_last = True \n \
                Conv2dNet.in_channels = 3 \n \
                Conv2dNet.convs = [(16,3), (32,3,2), (64,3,1,1)] \n"
            with open('./test.gin', 'w') as f:
                f.write(config)
            load_config('./test.gin')

            import numpy as np

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = Conv2dNet()
                    self.ff = FeedForwardNet()

                def forward(self, x):
                    x = self.conv(x)
                    x = x.view(-1, 64 * 6 * 6)
                    return self.ff(x)
            net = Net()
            assert net.conv.conv2d0.kernel_size == (3, 3)
            assert net.conv.conv2d1.kernel_size == (3, 3)
            assert net.conv.conv2d2.kernel_size == (3, 3)

            assert net.conv.conv2d0.stride == (1, 1)
            assert net.conv.conv2d1.stride == (2, 2)
            assert net.conv.conv2d2.stride == (1, 1)

            assert net.conv.conv2d2.padding == (1, 1)

            assert net.ff.fc0.in_features == 2304
            assert net.ff.fc1.in_features == 64
            assert net.ff.fc2.in_features == 64
            assert net.ff.fc2.out_features == 1
            os.remove('./test.gin')

    unittest.main()
