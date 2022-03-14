"""DQN algorithm.

https://www.nature.com/articles/nature14236
"""
from dl.rl.data_collection import ReplayBufferDataManager, ReplayBuffer, BatchedReplayBuffer
from dl import logger, nest, Algorithm, Checkpointer
import gin
import os
import time
import torch
import torch.nn as nn
import numpy as np
from dl.rl.util import rl_evaluate, rl_record, misc, LinearSchedule
from dl.rl.envs import VecFrameStack, VecEpsilonGreedy, VecEpisodeLogger


class EpsilonGreedyActor(object):
    """Epsilon Greedy actor."""

    def __init__(self, qf, epsilon_schedule, action_space, nenv):
        """Init."""
        self.qf = qf
        self.eps = epsilon_schedule
        self.action_space = action_space
        self.nenv = nenv
        self.t = 0

    def __call__(self, obs):
        """Epsilon greedy action."""
        if self.eps.value(self.t) > np.random.rand():
            action = torch.from_numpy(
                np.array([self.action_space.sample() for _ in range(self.nenv)])
            )
        else:
            action = self.qf(obs).action
        self.t += 1
        return {'action': action}

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']


@gin.configurable(blacklist=['logdir'])
class DQN(Algorithm):
    """DQN algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 qf_fn,
                 nenv=1,
                 optimizer=torch.optim.RMSprop,
                 buffer_size=100000,
                 frame_stack=1,
                 learning_starts=10000,
                 update_period=1,
                 gamma=0.99,
                 huber_loss=True,
                 exploration_timesteps=1000000,
                 final_eps=0.1,
                 eval_eps=0.05,
                 target_update_period=10000,
                 batch_size=32,
                 gpu=True,
                 eval_num_episodes=1,
                 record_num_episodes=1,
                 log_period=10):
        """Init."""
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.env_fn = env_fn
        self.nenv = nenv
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.gamma = gamma
        self.frame_stack = frame_stack
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_period = update_period
        self.eval_eps = eval_eps
        self.target_update_period = target_update_period - (
            target_update_period % self.update_period)
        self.log_period = log_period
        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')

        self.env = VecEpisodeLogger(env_fn(nenv=nenv))
        stacked_env = VecFrameStack(env_fn(nenv=nenv), self.frame_stack)

        self.qf = qf_fn(stacked_env).to(self.device)
        self.qf_targ = qf_fn(stacked_env).to(self.device)
        self.opt = optimizer(self.qf.parameters())
        if huber_loss:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = torch.nn.MSELoss(reduction='none')
        self.eps_schedule = LinearSchedule(exploration_timesteps, final_eps,
                                           1.0)
        self._actor = EpsilonGreedyActor(self.qf, self.eps_schedule,
                                         self.env.action_space, self.nenv)

        self.buffer = BatchedReplayBuffer(*[
            ReplayBuffer(buffer_size, frame_stack) for _ in range(self.nenv)
        ])
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    self._actor,
                                                    self.device,
                                                    self.learning_starts,
                                                    self.update_period)
        self.t = 0

    def _compute_target(self, rew, next_ob, done):
        qtarg = self.qf_targ(next_ob).max_q
        return rew + (1.0 - done) * self.gamma * qtarg

    def _get_batch(self):
        return self.data_manager.sample(self.batch_size)

    def loss(self, batch):
        """Compute loss."""
        q = self.qf(batch['obs'], batch['action']).value

        with torch.no_grad():
            target = self._compute_target(batch['reward'], batch['next_obs'],
                                          batch['done'])

        assert target.shape == q.shape
        loss = self.criterion(target, q).mean()
        if self.t % self.log_period < self.update_period:
            logger.add_scalar('alg/maxq', torch.max(q).detach().cpu().numpy(),
                              self.t, time.time())
            logger.add_scalar('alg/loss', loss.detach().cpu().numpy(), self.t,
                              time.time())
            logger.add_scalar('alg/epsilon',
                              self.eps_schedule.value(self._actor.t),
                              self.t, time.time())
        return loss

    def step(self):
        """Step."""
        self.t += self.data_manager.step_until_update()
        if self.t % self.target_update_period == 0:
            self.qf_targ.load_state_dict(self.qf.state_dict())

        self.opt.zero_grad()
        loss = self.loss(self._get_batch())
        loss.backward()
        self.opt.step()
        return self.t

    def evaluate(self):
        """Evaluate."""
        eval_env = VecEpsilonGreedy(VecFrameStack(self.env, self.frame_stack),
                                    self.eval_eps)
        self.qf.eval()
        misc.set_env_to_eval_mode(eval_env)

        # Eval policy
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval',
                               self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(eval_env, self.qf, self.eval_num_episodes,
                            outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'],
                          self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'],
                          self.t, time.time())

        # Record policy
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video',
                               self.ckptr.format.format(self.t) + '.mp4')
        rl_record(eval_env, self.qf, self.record_num_episodes, outfile,
                  self.device)

        self.qf.train()
        misc.set_env_to_train_mode(self.env)
        self.data_manager.manual_reset()

    def save(self):
        """Save."""
        state_dict = {
            'qf': self.qf.state_dict(),
            'qf_targ': self.qf.state_dict(),
            'opt': self.opt.state_dict(),
            '_actor': self._actor.state_dict(),
            'env': misc.env_state_dict(self.env),
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
        """Load."""
        state_dict = self.ckptr.load(t)
        if state_dict is None:
            self.t = 0
            return self.t
        self.qf.load_state_dict(state_dict['qf'])
        self.qf_targ.load_state_dict(state_dict['qf_targ'])
        self.opt.load_state_dict(state_dict['opt'])
        self._actor.load_state_dict(state_dict['_actor'])
        misc.env_load_state_dict(self.env, state_dict['env'])
        self.t = state_dict['t']

        buffer_format = state_dict['buffer_format']
        buffer_state = dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                 'buffer.npz')))
        buffer_state = nest.flatten(buffer_state)
        self.buffer.load_state_dict(nest.pack_sequence_as(buffer_state,
                                                          buffer_format))
        self.data_manager.manual_reset()
        return self.t

    def close(self):
        """Close environment."""
        try:
            self.env.close()
        except Exception:
            pass


if __name__ == '__main__':
    import unittest
    import shutil
    from dl import train
    from dl.rl.envs import make_atari_env
    from dl.rl.modules import DiscreteQFunctionBase
    from dl.rl.modules import QFunction
    from dl.rl.util import conv_out_shape
    import torch.nn.functional as F
    from functools import partial

    class NatureDQN(DiscreteQFunctionBase):
        """Deep network from https://www.nature.com/articles/nature14236."""

        def build(self):
            """Build."""
            self.conv1 = nn.Conv2d(4, 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            shape = self.observation_space.shape[1:]
            for c in [self.conv1, self.conv2, self.conv3]:
                shape = conv_out_shape(shape, c)
            self.nunits = 64 * np.prod(shape)
            self.fc = nn.Linear(self.nunits, 512)
            self.qf = nn.Linear(512, self.action_space.n)

        def forward(self, x):
            """Forward."""
            x = x.float() / 255.
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc(x.view(-1, self.nunits)))
            return self.qf(x)

    class TestDQN(unittest.TestCase):
        """Test case."""

        def test_ql(self):
            """Test."""
            def env_fn(nenv):
                return make_atari_env('Pong', nenv, frame_stack=1)

            def qf_fn(env):
                return QFunction(NatureDQN(env.observation_space,
                                           env.action_space))

            ql = partial(DQN,
                         env_fn=env_fn,
                         qf_fn=qf_fn,
                         learning_starts=100,
                         buffer_size=200,
                         update_period=4,
                         frame_stack=4,
                         exploration_timesteps=500,
                         target_update_period=100)
            train('logs', ql, maxt=1000, eval=True, eval_period=1000)
            alg = ql('logs')
            alg.load()
            assert np.allclose(alg.eps_schedule.value(alg.t), 0.1)
            shutil.rmtree('logs')

    unittest.main()
