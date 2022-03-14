"""Interface for 2-player zero-sum games."""
import numpy as np


class Game(object):
    """Game base class."""

    def reset(self):
        """Return start state and player id."""
        raise NotImplementedError

    def get_canonical_state(self, state, player):
        """Return cononical view of board and canonical player id."""
        raise NotImplementedError

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


class TicTacToe(Game):
    """Tic Tac Toe game."""

    def reset(self):
        """Reset."""
        return np.zeros((3, 3)), 1

    def get_canonical_state(self, state, player):
        """Return canonical state/player."""
        return player * state, 1

    def action_space(self):
        """Return the shape of the action space."""
        return (9,)

    def get_valid_actions(self, state, player):
        """Return the valid actions."""
        return state.ravel() == 0

    def move(self, state, player, action):
        """Move."""
        y, x = action // 3, action % 3
        if state[y][x] != 0:
            raise ValueError("Invalid Move!!")
        state = state.copy()
        state[y][x] = player
        return state, -player

    def game_over(self, state, player):
        """Check for game over."""
        rows = np.sum(state, axis=0)
        if np.any(rows == 3):
            return True, player
        if np.any(rows == -3):
            return True, -player

        cols = np.sum(state, axis=1)
        if np.any(cols == 3):
            return True, player
        if np.any(cols == -3):
            return True, -player

        diag1 = state[0, 0] + state[1, 1] + state[2, 2]
        if diag1 == 3:
            return True, player
        if diag1 == -3:
            return True, -player

        diag2 = state[0, 2] + state[1, 1] + state[2, 0]
        if diag2 == 3:
            return True, player
        if diag2 == -3:
            return True, -player

        if not np.any(state == 0):
            return True, 0

        return False, 0

    def to_string(self, state, player):
        """To string."""
        x = []
        for xx in state.ravel():
            if xx == -1:
                x.append("-1")
            elif xx == 1:
                x.append(" 1")
            else:
                x.append("  ")
        s = f"Player: {player} \n"
        s += f" ______________ \n"
        s += f"| {x[0]} | {x[1]} | {x[2]} |\n"
        s += f" ______________ \n"
        s += f"| {x[3]} | {x[4]} | {x[5]} |\n"
        s += f" ______________ \n"
        s += f"| {x[6]} | {x[7]} | {x[8]} |\n"
        s += f" ______________ \n"
        return s


if __name__ == '__main__':
    g = TicTacToe()
    s, p = g.reset()
    print(g.to_string(s, p))
    while not g.game_over(s, p)[0]:
        acs = g.get_valid_actions(s, p)
        inds = np.where(acs)[0]
        s, p = g.move(s, p, np.random.choice(inds))
        print(g.to_string(s, p))
