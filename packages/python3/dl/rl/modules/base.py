"""Define the interface between torch modules and RL modules."""
import torch.nn as nn


class RLBase(nn.Module):
    """Base interface."""

    def __init__(self, observation_space, action_space):
        """Init."""
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.build()

    def build(self):
        """Create model here."""
        pass


class ValueFunctionBase(RLBase):
    """ValueFunction Interface."""

    def forward(self, ob, state_in=None):
        """Return the value of ob."""
        raise NotImplementedError


class DiscreteQFunctionBase(RLBase):
    """Discsrete QFunction Interface."""

    def forward(self, ob, state_in=None):
        """Return the qvalue of ob for each action."""
        raise NotImplementedError


class ContinuousQFunctionBase(RLBase):
    """Continuous QFunction Interface."""

    def forward(self, ob, ac, state_in=None):
        """Return the qvalue of ob and ac."""
        raise NotImplementedError


class PolicyBase(RLBase):
    """Policy Interface."""

    def forward(self, ob, state_in=None):
        """Forward.

        Args:
            ob: Observation,
            state_in: Hidden_states for recurrent policies
        Returns:
            dist: Action distribution,
            state_out: next hidden states of the model (optional)

        """
        raise NotImplementedError


class ActorCriticBase(RLBase):
    """Actor Critic interface."""

    def forward(self, ob, state_in=None):
        """forward.

        Args:
            ob: Observation,
            state_in: Hidden_states for recurrent policies
        Returns:
            dist: Action distribution,
            value: The value of ob,
            state_out: next hidden states of the model (optional)

        """
        raise NotImplementedError
