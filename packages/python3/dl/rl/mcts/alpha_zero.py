"""Alpha Zero implementation."""

from dl import logger, Checkpointer, nest
from dl.rl.mcts import MCTS
import gin
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GameReplay(object):
    """Storage for game data."""

    def __init__(self, max_size):
        """Init."""
        self.n = max_size
        self.data = None

    def _init_data(self, data):
        self.data = {}
        for k, v in data.items():
            dtype = np.float32 if v.dtype == np.float64 else v.dtype
            shape = [self.n] + list(v.shape[1:])
            self.data[k] = np.zeros(shape, dtype=dtype)
        self.it = 0
        self.max_it = 0
        self.sample_max = 0

    def insert(self, data):
        """Insert game data."""
        if self.data is None:
            self._init_data(data)

        batch_size = data['state'].shape[0]
        end_it = (self.it + batch_size) % self.n
        for k, v in data.items():
            if end_it == 0:
                self.data[k][self.it:] = v
            elif end_it < self.it:
                self.data[k][self.it:] = v[:-end_it]
                self.data[k][:end_it] = v[-end_it:]
            else:
                self.data[k][self.it:end_it] = v
        self.max_it = min(self.max_it + batch_size, self.n)
        self.it = end_it

    def full(self):
        """Return if the buffer is full."""
        return self.max_it == self.n

    def sample(self, batch_size):
        """Sample a batch of game data."""
        if self.data is None:
            return None
        if self.max_it < batch_size:
            return None

        inds = np.random.choice(range(self.max_it), size=batch_size,
                                replace=False)
        return {k: v[inds] for k, v in self.data.items()}

    def state_dict(self):
        """Save buffer."""
        return {'data': self.data,
                'it': self.it,
                'max_it': self.max_it}

    def load_state_dict(self, state_dict):
        """Load buffer."""
        self.data = state_dict['data']
        self.it = state_dict['it']
        self.max_it = state_dict['max_it']


class SelfPlayManager(object):
    """Generate and store self play data."""

    def __init__(self, pi, game, game_buffer, device):
        """Init."""
        self.game = game

        def policy(state, valid_actions):
            with torch.no_grad():
                outs = pi(torch.from_numpy(state).to(device).unsqueeze(0))
                value = outs.value.squeeze(0).cpu().numpy()
                logits = outs.dist.logits.squeeze(0)
                probs = F.softmax(logits[valid_actions], dim=0)
                p = torch.zeros_like(logits)
                p[valid_actions] = probs
                return p.cpu().numpy(), value

        self.pi = policy
        self.buffer = game_buffer
        self.device = device

        self.mcts = MCTS(game, policy)

    def play_game(self, n_sims):
        """Play a self play game."""
        data = {'state': [], 'prob': [], 'value': []}

        self.mcts.reset()

        while not self.mcts.game_over():
            self.mcts.search(n_sims)
            data['state'].append(self.mcts.get_state())
            data['prob'].append(self.mcts.get_action_probs())
            self.mcts.act(argmax=False)
        score = self.mcts.score()

        values = []
        game_length = len(data['state'])
        for _ in range(game_length):
            score = -score
            values.append(score)
        data['value'] = np.asarray(values[::-1]).astype(np.float32)
        data['state'] = np.asarray(data['state'])
        data['prob'] = np.asarray(data['prob'])
        self.buffer.insert(data)
        return game_length

    def sample(self, batch_size):
        """Sample a batch of self play data."""
        batch = self.buffer.sample(batch_size)

        def _to_torch(x):
            return torch.from_numpy(x).to(self.device)
        return nest.map_structure(_to_torch, batch)


@gin.configurable
class AlphaZero(object):
    """Alpha Zero Agent."""

    def __init__(self,
                 logdir,
                 game,
                 policy,
                 optimizer=torch.optim.Adam,
                 n_simulations=100,
                 buffer_size=200,
                 batch_size=64,
                 batches_per_game=1,
                 gpu=True):
        """Init."""
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.game = game
        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')

        self.game = game
        self.n_sims = n_simulations
        self.batch_size = batch_size
        self.batches_per_game = batches_per_game

        self.pi = policy.to(self.device)
        self.opt = optimizer(self.pi.parameters(), lr=1e-2, weight_decay=1e-4)

        self.buffer = GameReplay(buffer_size)
        self.data_manager = SelfPlayManager(self.pi, self.game, self.buffer,
                                            self.device)

        self.mse = nn.MSELoss()

        self.t = 0

    def loss(self, batch):
        """Compute Loss."""
        loss = {}
        outs = self.pi(batch['state'])
        log_probs = F.log_softmax(outs.dist.logits, dim=1)
        loss['pi'] = -(batch['prob'] * log_probs).sum(dim=1).mean()
        loss['value'] = self.mse(batch['value'], outs.value)
        loss['total'] = loss['pi'] + loss['value']
        return loss

    def step(self):
        """Step alpha zero."""
        self.pi.train()
        self.t += self.data_manager.play_game(self.n_sims)

        # fill replay buffer if needed
        while not self.buffer.full():
            self.t += self.data_manager.play_game(self.n_sims)

        for _ in range(self.batches_per_game):
            batch = self.data_manager.sample(self.batch_size)
            self.opt.zero_grad()
            loss = self.loss(batch)
            loss['total'].backward()
            self.opt.step()
            for k, v in loss.items():
                logger.add_scalar(f'loss/{k}', v.detach().cpu().numpy(), self.t,
                                  time.time())
        return self.t

    def evaluate(self):
        """Evaluate."""
        for _ in range(10):
            print("Play against random!")
            state, id = self.game.reset()

            self.pi.eval()
            while not self.game.game_over(state, id)[0]:
                print(self.game.to_string(state, id))
                cs, _ = self.game.get_canonical_state(state, id)
                valid_actions = self.game.get_valid_actions(state, id)
                with torch.no_grad():
                    p, v = self.data_manager.pi(cs, valid_actions)
                print(f"ALPHAZero thinks the value is: {v * id}")
                state, id = self.game.move(state, id, np.argmax(p))

                #############################################
                # Play against random agent. Comment out for self play
                if self.game.game_over(state, id)[0]:
                    break
                print(self.game.to_string(state, id))
                valid_actions = self.game.get_valid_actions(state, id)
                acs = [i for i in range(len(valid_actions)) if valid_actions[i]]
                ac = np.random.choice(acs)
                state, id = self.game.move(state, id, ac)
                #############################################
            print(self.game.to_string(state, id))
            _, score = self.game.game_over(state, id)
            print(f"SCORE: {-score}")

    def save(self):
        """State dict."""
        state_dict = {
            'pi': self.pi.state_dict(),
            'opt': self.opt.state_dict(),
            't': self.t
        }
        buffer_dict = self.buffer.state_dict()
        state_dict['buffer_format'] = nest.get_structure(buffer_dict)
        self.ckptr.save(state_dict, self.t)

        # save buffer seperately and only once (because it can be huge)
        np.savez(os.path.join(self.ckptr.ckptdir, 'buffer.npz'),
                 **{f'{i:04d}': x for i, x in
                    enumerate(nest.flatten(buffer_dict))})

    def load(self, t=None):
        """Load state dict."""
        state_dict = self.ckptr.load(t)
        if state_dict is None:
            self.t = 0
            return self.t
        self.pi.load_state_dict(state_dict['pi'])
        self.opt.load_state_dict(state_dict['opt'])
        self.t = state_dict['t']

        buffer_format = state_dict['buffer_format']
        buffer_state = dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                 'buffer.npz')))
        buffer_state = nest.flatten(buffer_state)
        self.buffer.load_state_dict(nest.pack_sequence_as(buffer_state,
                                                          buffer_format))
        return self.t

    def close(self):
        """Close."""
        pass


if __name__ == '__main__':

    import unittest
    import shutil
    from dl import train
    from dl.rl.modules import ActorCriticBase, Policy
    from dl.modules import Categorical
    from functools import partial
    from dl.rl.mcts import TicTacToe

    class TicTacToeNet(ActorCriticBase):
        """Tic Tac Toe Net."""

        def __init__(self):
            """Build network."""
            super().__init__(None, None)
            self.fc1 = nn.Linear(9, 32)
            self.fc2 = nn.Linear(32, 32)
            self.vf = nn.Linear(32, 1)
            self.dist = Categorical(32, 9)

        def forward(self, x):
            """Forward."""
            x = F.relu(self.fc1(x.float().view(-1, 9)))
            x = F.relu(self.fc2(x))
            return self.dist(x), self.vf(x)

    class TestAlphaZero(unittest.TestCase):
        """Test case."""

        def test(self):
            """Test."""
            def policy_fn():
                return Policy(TicTacToeNet())

            a0 = partial(AlphaZero, game=TicTacToe(), policy=policy_fn())
            train('test', a0, maxt=100000, eval=True, eval_period=1000)
            # shutil.rmtree('test')

    unittest.main()
