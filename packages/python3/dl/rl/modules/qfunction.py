"""Implements a torch module for QFunctions."""
import torch.nn as nn
import gin
from dl.rl.modules.base import DiscreteQFunctionBase, ContinuousQFunctionBase
from collections import namedtuple


@gin.configurable(whitelist=['base'])
class QFunction(nn.Module):
    """Qfunction module."""

    def __init__(self, base):
        """Init."""
        super().__init__()
        self.base = base
        assert isinstance(self.base, (DiscreteQFunctionBase,
                                      ContinuousQFunctionBase))
        self.discrete = isinstance(self.base, DiscreteQFunctionBase)
        self.outputs = namedtuple('Outputs', ['action', 'value', 'max_a',
                                              'max_q', 'qvals', 'state_out'])

    def _run_base(self, ob, action=None, state_in=None):
        if action is None or self.discrete:
            assert self.discrete, (
                "You must provide an action for a continuous action space")
            if state_in is None:
                outs = self.base(ob)
            else:
                outs = self.base(ob, state_in=state_in)
            if isinstance(outs, tuple):
                qvals, state_out = outs
            else:
                qvals, state_out = outs, None
            return qvals, state_out

        else:
            if state_in is None:
                outs = self.base(ob, action)
            else:
                outs = self.base(ob, action, state_in)
            if isinstance(outs, tuple):
                qval, state_out = outs
            else:
                qval, state_out = outs, None
            return qval.squeeze(1), state_out

    def forward(self, ob, action=None, state_in=None):
        """Compute Q-value.

        Returns:
            out (namedtuple):
                out.action: If an action is specified, out.action is the same,
                            otherwise it is the argmax of the Q-values
                out.value:  The q value of (x, out.action)
                out.max_a:  The argmax of the Q-values
                            (only available for discrete action spaces)
                out.max_q:  The max of the Q-values
                            (only available for discrete action spaces)
                out.qvals:  The Q-value for each action
                            (only available for discrete action spaces)
                out.state_out:  The temporal state of the model

        """
        if action is None:
            qvals, state_out = self._run_base(ob, state_in=state_in)
            maxq, maxa = qvals.max(dim=-1)
            return self.outputs(action=maxa, value=maxq, max_a=maxa, max_q=maxq,
                                qvals=qvals, state_out=state_out)
        elif self.discrete:
            qvals, state_out = self._run_base(ob, state_in=state_in)
            maxq, maxa = qvals.max(dim=-1)
            if len(action.shape) == 1:
                action = action.long().unsqueeze(1)
            else:
                action = action.long()
            value = qvals.gather(1, action).squeeze(1)
            return self.outputs(action=action.squeeze(1), value=value,
                                max_a=maxa, max_q=maxq, qvals=qvals,
                                state_out=state_out)
        else:
            value, state_out = self._run_base(ob, action, state_in)
            return self.outputs(action=action, value=value, max_a=None,
                                max_q=None, qvals=None, state_out=state_out)


if __name__ == '__main__':
    import gym
    import torch
    import unittest
    import numpy as np

    class TestQF(unittest.TestCase):
        """Test."""

        def test_discrete(self):
            """Test discsrete qfunction."""
            class Base(DiscreteQFunctionBase):
                def forward(self, ob):
                    return torch.from_numpy(np.random.rand(ob.shape[0],
                                                           self.action_space.n))

            env = gym.make('CartPole-v1')
            q = QFunction(Base(env.observation_space, env.action_space))
            ob = env.reset()

            outs = q(ob[None])
            assert np.allclose(outs.action, outs.max_a)
            assert np.allclose(outs.value, outs.max_q)
            assert outs.action.shape == (1,)

            outs = q(ob[None], torch.from_numpy(np.array([0])))
            assert np.allclose(outs.action, 0)
            assert np.allclose(outs.value, outs.qvals[:, 0])
            assert outs.action.shape == (1,)

            outs = q(ob[None], torch.from_numpy(np.array([[0]])))
            assert np.allclose(outs.action, 0)
            assert np.allclose(outs.value, outs.qvals[:, 0])
            assert outs.action.shape == (1,)

        def test_continuous(self):
            """Test continuous qfunction."""
            class Base(ContinuousQFunctionBase):
                def forward(self, ob, ac):
                    return torch.from_numpy(np.random.rand(ob.shape[0], 1))

            env = gym.make('CartPole-v1')
            q = QFunction(Base(env.observation_space, env.action_space))
            ob = env.reset()

            ac = torch.from_numpy(np.array([[1]]))
            outs = q(ob[None], ac)
            assert outs.max_a is None
            assert outs.max_q is None
            assert outs.action.shape == (1, 1)

    unittest.main()
