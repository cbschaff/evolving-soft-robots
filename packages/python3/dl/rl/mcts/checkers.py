from dl.rl.mcts.game import Game
import numpy as np


class Checkers(Game):
    """Checkers."""

    def reset(self):
        """Return start state and player id."""
        state = np.zeros((2, 8, 8))
        state[0, 0][[1, 3, 5, 7]] = 1
        state[0, 1][[0, 2, 4, 6]] = 1
        state[0, 2][[1, 3, 5, 7]] = 1
        state[0, -1][[1, 3, 5, 7]] = -1
        state[0, -2][[0, 2, 4, 6]] = -1
        state[0, -3][[1, 3, 5, 7]] = -1
        return state, 1

    def get_canonical_state(self, state, player):
        """Return cononical view of board and canonical player id."""
        return player * state, 1

    def action_space(self):
        """Return the action space of the game."""
        raise NotImplementedError

    def get_valid_actions(self, state, player):
        """Return the valid actions for the current state/player."""
        raise NotImplementedError

    def move(self, state, player, action):
        """Return next state and next player id."""
        raise NotImplementedError

    def game_over(self, state, player):
        """Return a bool for game_over and the score of player."""
        raise NotImplementedError

    def to_string(self, state, player):
        """Return a string representation of the state of the game."""
        raise NotImplementedError
