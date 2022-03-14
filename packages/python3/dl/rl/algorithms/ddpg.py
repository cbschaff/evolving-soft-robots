"""DDPG algorithm.

https://arxiv.org/abs/1509.02971
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
from dl.rl.envs import VecFrameStack, VecEpisodeLogger


def soft_target_update(target_net, net, tau):
    """Soft update totarget network."""
    for tp, p in zip(target_net.parameters(), net.parameters()):
        tp.data.copy_((1. - tau) * tp.data + tau * p.data)


class OrnsteinUhlenbeck(object):
    """Ornstein-Uhlenbeck process for directed exploration.

    Default parameters are chosen from the DDPg paper.
    """

    def __init__(self, shape, device, theta, sigma):
        """Init."""
        self.theta = theta
        self.sigma = sigma
        self.x = torch.zeros(shape, device=device, dtype=torch.float32,
                             requires_grad=False)
        self.dist = torch.distributions.Normal(
                torch.zeros(shape, device=device, dtype=torch.float32),
                torch.ones(shape, device=device, dtype=torch.float32))

    def __call__(self):
        """Sample."""
        with torch.no_grad():
            self.x = ((1. - self.theta) * self.x
                      + self.sigma * self.dist.sample())
        return self.x


class DDPGActor(object):
    """DDPG actor."""

    def __init__(self, pi, action_space, theta=0.15, sigma=0.2):
        """Init."""
        self.pi = pi
        self.noise = None
        self.action_space = action_space
        self.theta = theta
        self.sigma = sigma

    def __call__(self, obs):
        """Act."""
        with torch.no_grad():
            action = self.pi(obs, deterministic=True).action
            return {'action': self.add_noise_to_action(action)}

    def add_noise_to_action(self, action):
        """Add exploration noise."""
        if self.noise is None:
            self.noise = OrnsteinUhlenbeck(action.shape, action.device,
                                           self.theta, self.sigma)
            self.low = torch.from_numpy(self.action_space.low).to(
                                                            action.device)
            self.high = torch.from_numpy(self.action_space.high).to(
                                                            action.device)
        return torch.max(torch.min(action + self.noise(), self.high), self.low)

    def update_sigma(self, sigma):
        """Update noise standard deviation."""
        self.sigma = sigma
        if self.noise:
            self.noise.sigma = sigma


@gin.configurable(blacklist=['logdir'])
class DDPG(Algorithm):
    """DDPG algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 policy_fn,
                 qf_fn,
                 nenv=1,
                 optimizer=torch.optim.Adam,
                 buffer_size=10000,
                 frame_stack=1,
                 learning_starts=1000,
                 update_period=1,
                 batch_size=256,
                 policy_lr=1e-4,
                 qf_lr=1e-3,
                 qf_weight_decay=0.01,
                 gamma=0.99,
                 noise_theta=0.15,
                 noise_sigma=0.2,
                 noise_sigma_final=0.01,
                 noise_decay_period=10000,
                 target_update_period=1,
                 target_smoothing_coef=0.005,
                 reward_scale=1,
                 gpu=True,
                 eval_num_episodes=1,
                 record_num_episodes=1,
                 log_period=1000):
        """Init."""
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.env_fn = env_fn
        self.nenv = nenv
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.frame_stack = frame_stack
        self.learning_starts = learning_starts
        self.update_period = update_period
        self.batch_size = batch_size
        if target_update_period < self.update_period:
            self.target_update_period = self.update_period
        else:
            self.target_update_period = target_update_period - (
                                target_update_period % self.update_period)
        self.reward_scale = reward_scale
        self.target_smoothing_coef = target_smoothing_coef
        self.log_period = log_period

        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')
        self.t = 0

        self.env = VecEpisodeLogger(env_fn(nenv=nenv))
        self.policy_fn = policy_fn
        self.qf_fn = qf_fn
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.pi = policy_fn(eval_env)
        self.qf = qf_fn(eval_env)
        self.target_pi = policy_fn(eval_env)
        self.target_qf = qf_fn(eval_env)

        self.pi.to(self.device)
        self.qf.to(self.device)
        self.target_pi.to(self.device)
        self.target_qf.to(self.device)

        self.optimizer = optimizer
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.qf_weight_decay = qf_weight_decay
        self.opt_pi = optimizer(self.pi.parameters(), lr=policy_lr)
        self.opt_qf = optimizer(self.qf.parameters(), lr=qf_lr,
                                weight_decay=qf_weight_decay)

        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_qf.load_state_dict(self.qf.state_dict())

        self.noise_schedule = LinearSchedule(noise_decay_period,
                                             noise_sigma_final, noise_sigma)
        self._actor = DDPGActor(self.pi, self.env.action_space, noise_theta,
                                self.noise_schedule.value(self.t))
        self.buffer = BatchedReplayBuffer(*[
            ReplayBuffer(buffer_size, frame_stack) for _ in range(self.nenv)
        ])
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    self._actor,
                                                    self.device,
                                                    self.learning_starts,
                                                    self.update_period)

        self.qf_criterion = torch.nn.MSELoss()
        if self.env.action_space.__class__.__name__ == 'Discrete':
            raise ValueError("Action space must be continuous!")

    def loss(self, batch):
        """Loss function."""
        # compute QFunction loss.
        with torch.no_grad():
            target_action = self.target_pi(batch['next_obs']).action
            target_q = self.target_qf(batch['next_obs'], target_action).value
            qtarg = self.reward_scale * batch['reward'].float() + (
                    (1.0 - batch['done']) * self.gamma * target_q)

        q = self.qf(batch['obs'], batch['action']).value
        assert qtarg.shape == q.shape
        qf_loss = self.qf_criterion(q, qtarg)

        # compute policy loss
        action = self.pi(batch['obs'], deterministic=True).action
        q = self.qf(batch['obs'], action).value
        pi_loss = -q.mean()

        # log losses
        if self.t % self.log_period < self.update_period:
            logger.add_scalar('loss/qf', qf_loss, self.t, time.time())
            logger.add_scalar('loss/pi', pi_loss, self.t, time.time())
        return pi_loss, qf_loss

    def step(self):
        """Step optimization."""
        self._actor.update_sigma(self.noise_schedule.value(self.t))
        self.t += self.data_manager.step_until_update()
        if self.t % self.target_update_period == 0:
            soft_target_update(self.target_pi, self.pi,
                               self.target_smoothing_coef)
            soft_target_update(self.target_qf, self.qf,
                               self.target_smoothing_coef)

        if self.t % self.update_period == 0:
            batch = self.data_manager.sample(self.batch_size)

            pi_loss, qf_loss = self.loss(batch)

            # update
            self.opt_pi.zero_grad()
            pi_loss.backward()
            self.opt_pi.step()

            self.opt_qf.zero_grad()
            qf_loss.backward()
            self.opt_qf.step()

        return self.t

    def evaluate(self):
        """Evaluate."""
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.pi.eval()
        misc.set_env_to_eval_mode(eval_env)

        # Eval policy
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval',
                               self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(eval_env, self.pi, self.eval_num_episodes,
                            outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'],
                          self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'],
                          self.t, time.time())

        # Record policy
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video',
                               self.ckptr.format.format(self.t) + '.mp4')
        rl_record(eval_env, self.pi, self.record_num_episodes, outfile,
                  self.device)

        self.pi.train()
        misc.set_env_to_train_mode(self.env)
        self.data_manager.manual_reset()

    def save(self):
        """Save."""
        state_dict = {
            'pi': self.pi.state_dict(),
            'qf': self.qf.state_dict(),
            'target_pi': self.target_pi.state_dict(),
            'target_qf': self.target_qf.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_qf': self.opt_qf.state_dict(),
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
        self.pi.load_state_dict(state_dict['pi'])
        self.qf.load_state_dict(state_dict['qf'])
        self.target_pi.load_state_dict(state_dict['target_pi'])
        self.target_qf.load_state_dict(state_dict['target_qf'])

        self.opt_pi.load_state_dict(state_dict['opt_pi'])
        self.opt_qf.load_state_dict(state_dict['opt_qf'])
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
    from dl.rl.envs import make_env, ActionNormWrapper
    from dl.rl.modules import QFunction
    from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
    from dl.rl.modules import Policy
    from dl.modules import Delta
    import torch.nn.functional as F
    from functools import partial

    class PiBase(PolicyBase):
        """Policy network."""

        def build(self):
            """Build Network."""
            self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.dist = Delta(32, self.action_space.shape[0])

        def forward(self, x):
            """Forward."""
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.dist(x)

    class QFBase(ContinuousQFunctionBase):
        """Q network."""

        def build(self):
            """Build Network."""
            nin = self.observation_space.shape[0] + self.action_space.shape[0]
            self.fc1 = nn.Linear(nin, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.qvalue = nn.Linear(32, 1)

        def forward(self, x, a):
            """Forward."""
            x = F.relu(self.fc1(torch.cat([x, a], dim=1)))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.qvalue(x)

    def env_fn(nenv):
        """Environment function."""
        return ActionNormWrapper(make_env('LunarLanderContinuous-v2', nenv=nenv))

    def policy_fn(env):
        """Create a policy."""
        return Policy(PiBase(env.observation_space, env.action_space))

    def qf_fn(env):
        """Create a qfunction."""
        return QFunction(QFBase(env.observation_space, env.action_space))

    class TestDDPG(unittest.TestCase):
        """Test case."""

        def test_sac(self):
            """Test."""
            ddpg = partial(DDPG,
                           env_fn=env_fn,
                           policy_fn=policy_fn,
                           qf_fn=qf_fn,
                           learning_starts=300,
                           eval_num_episodes=1,
                           buffer_size=500)
            train('logs', ddpg, maxt=1000, eval=False, eval_period=1000)
            alg = ddpg('logs')
            assert alg.load() == 1000
            shutil.rmtree('logs')

    unittest.main()
