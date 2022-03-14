import numpy as np


class MCTSNode(object):
    """Search Tree."""

    def __init__(self, state, player_id, action_probs, legal_actions, over,
                 score):
        self.state = state
        self.player_id = player_id
        self.p = action_probs
        self.n = 0
        self.na = np.zeros(len(legal_actions))
        self.q = np.zeros(len(legal_actions))
        self.legal_actions = legal_actions
        self.illegal_actions = np.logical_not(legal_actions)
        self.children = [None for _ in range(len(legal_actions))]
        self.game_over = over
        self.score = score

    def update(self, ac, value):
        np1 = self.na[ac] + 1
        self.q[ac] = (self.na[ac] / np1) * self.q[ac] + (1 / np1) * value
        self.n += 1
        self.na[ac] += 1

    def add_dirichlet_noise(self, phantom_moves, eps):
        """Add dirchlet noise to prior probs."""
        n_legal_ac = self.legal_actions.sum()
        alpha = (phantom_moves / n_legal_ac) * self.legal_actions + 1e-8
        self.p = (1. - eps) * self.p + eps * np.random.dirichlet(alpha)

    def ucb(self, cpuct):
        u = self.q + cpuct * self.p * np.sqrt(self.n) / (self.na + 1)
        u[self.illegal_actions] = -np.inf
        return u

    def sample_action(self, argmax):
        if self.game_over:
            return None
        if argmax:
            return np.argmax(self.na)
        else:
            return np.random.choice(np.where(self.legal_actions)[0],
                                    p=self.na[self.legal_actions]/self.n)


class MCTS(object):
    """MCTS."""

    def __init__(self, game, pi, cpuct=5.0, phantom_moves=10, eps=0.25):
        """init."""
        self.game = game
        self.pi = pi
        self.cpuct = cpuct
        self.phantom_moves = phantom_moves
        self.eps = 0.25
        self.reset()

    def reset(self):
        s, p = self.game.reset()
        legal_actions = self.game.get_valid_actions(s, p)
        cs, cp = self.game.get_canonical_state(s, p)
        probs, _ = self.pi(cs, legal_actions)
        self.tree = MCTSNode(s, p, probs, legal_actions, False, None)
        self.tree.add_dirichlet_noise(self.phantom_moves, self.eps)

    def get_state(self):
        return self.game.get_canonical_state(self.tree.state,
                                             self.tree.player_id)[0]

    def get_action_probs(self):
        return self.tree.na / self.tree.n

    def state_string(self):
        return self.game.to_string(self.tree.state, self.tree.player_id)

    def game_over(self):
        return self.tree.game_over

    def score(self):
        return self.tree.score

    def act(self, argmax):
        action = self.tree.sample_action(argmax)
        self.tree = self.tree.children[action]
        if not self.tree.game_over:
            self.tree.add_dirichlet_noise(self.phantom_moves, self.eps)
        return action, self.tree.state, self.tree.player_id

    def search(self, n_sims):
        for _ in range(n_sims):
            self.simulate(self.tree)

    def simulate(self, node):
        """Perform Search."""
        if node.game_over:
            return -node.score
        s, p = node.state, node.player_id
        ac = np.argmax(node.ucb(self.cpuct))

        if node.children[ac] is None:  # Leaf Node
            next_s, next_p = self.game.move(s, p, ac)
            legal_actions = self.game.get_valid_actions(next_s, next_p)
            over, score = self.game.game_over(next_s, next_p)
            if over:  # Check for game over
                node.children[ac] = MCTSNode(next_s, next_p, None,
                                             legal_actions, True, score)
                return -score

            cs, _ = self.game.get_canonical_state(next_s, next_p)
            probs, v = self.pi(cs, legal_actions)
            node.children[ac] = MCTSNode(next_s, next_p, probs, legal_actions,
                                         False, None)
            return -v

        elif isinstance(node.children[ac], MCTSNode):  # Continue simulating
            v = self.simulate(node.children[ac])
            node.update(ac, v)
            return -v

        else:  # next state is game over
            return -node.children[ac]


if __name__ == '__main__':

    class Pi(object):
        """Random policy."""

        def __call__(self, s, valid_actions):
            """predict."""
            return np.ones(9) * valid_actions / np.sum(valid_actions), 0

    from game import TicTacToe
    mcts = MCTS(TicTacToe(), Pi())
    print(mcts.state_string())

    while not mcts.game_over():
        mcts.search(1000)
        mcts.act(True)
        print(mcts.state_string())
