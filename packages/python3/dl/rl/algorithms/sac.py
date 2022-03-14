"""SAC algorithm.

https://arxiv.org/abs/1801.01290
"""
from dl.rl.data_collection import ReplayBufferDataManager, ReplayBuffer, BatchedReplayBuffer
from dl.modules import TanhNormal
from dl import logger, nest, Algorithm, Checkpointer
import gin
import os
import time
import torch
import torch.nn as nn
import numpy as np
from dl.rl.util import rl_evaluate, rl_record, misc
from dl.rl.envs import VecFrameStack, VecEpisodeLogger


def soft_target_update(target_net, net, tau):
    """Soft update totarget network."""
    for tp, p in zip(target_net.parameters(), net.parameters()):
        tp.data.copy_((1. - tau) * tp.data + tau * p.data)


class SACActor(object):
    """SAC actor."""

    def __init__(self, pi):
        """Init."""
        self.pi = pi

    def __call__(self, obs):
        """Act."""
        return {'action': self.pi(obs).action}


@gin.configurable(blacklist=['logdir'])
class SAC(Algorithm):
    """SAC algorithm."""

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
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 gamma=0.99,
                 target_update_period=1,
                 policy_update_period=1,
                 target_smoothing_coef=0.005,
                 alpha=0.2,
                 automatic_entropy_tuning=True,
                 target_entropy=None,
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
        if policy_update_period < self.update_period:
            self.policy_update_period = self.update_period
        else:
            self.policy_update_period = policy_update_period - (
                                policy_update_period % self.update_period)
        self.target_smoothing_coef = target_smoothing_coef
        self.log_period = log_period

        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')

        self.env = VecEpisodeLogger(env_fn(nenv=nenv))
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.pi = policy_fn(eval_env)
        self.qf1 = qf_fn(eval_env)
        self.qf2 = qf_fn(eval_env)
        self.target_qf1 = qf_fn(eval_env)
        self.target_qf2 = qf_fn(eval_env)

        self.pi.to(self.device)
        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.target_qf1.to(self.device)
        self.target_qf2.to(self.device)

        self.opt_pi = optimizer(self.pi.parameters(), lr=policy_lr)
        self.opt_qf1 = optimizer(self.qf1.parameters(), lr=qf_lr)
        self.opt_qf2 = optimizer(self.qf2.parameters(), lr=qf_lr)

        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.buffer = BatchedReplayBuffer(*[
            ReplayBuffer(buffer_size, frame_stack) for _ in range(self.nenv)
        ])
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    SACActor(self.pi),
                                                    self.device,
                                                    self.learning_starts,
                                                    self.update_period)

        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                target_entropies = nest.map_structure(
                    lambda space: -np.prod(space.shape).item(),
                    misc.unpack_space(self.env.action_space)
                )
                self.target_entropy = sum(nest.flatten(target_entropies))

            self.log_alpha = torch.tensor(np.log([self.alpha]),
                                          requires_grad=True,
                                          device=self.device,
                                          dtype=torch.float32)
            self.opt_alpha = optimizer([self.log_alpha], lr=policy_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.opt_alpha = None

        self.mse_loss = torch.nn.MSELoss()

        self.t = 0

    def loss(self, batch):
        """Loss function."""
        pi_out = self.pi(batch['obs'], reparameterization_trick=True)
        logp = pi_out.dist.log_prob(pi_out.action)
        q1 = self.qf1(batch['obs'], batch['action']).value
        q2 = self.qf2(batch['obs'], batch['action']).value

        # alpha loss
        if self.automatic_entropy_tuning:
            ent_error = logp + self.target_entropy
            alpha_loss = -(self.log_alpha * ent_error.detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
            alpha_loss = 0

        # qf loss
        with torch.no_grad():
            next_pi_out = self.pi(batch['next_obs'])
            next_ac_logp = next_pi_out.dist.log_prob(next_pi_out.action)
            q1_next = self.target_qf1(batch['next_obs'], next_pi_out.action).value
            q2_next = self.target_qf2(batch['next_obs'], next_pi_out.action).value
            qnext = torch.min(q1_next, q2_next) - alpha * next_ac_logp
            qtarg = batch['reward'] + (1.0 - batch['done']) * self.gamma * qnext

        assert qtarg.shape == q1.shape
        assert qtarg.shape == q2.shape
        qf1_loss = self.mse_loss(q1, qtarg)
        qf2_loss = self.mse_loss(q2, qtarg)

        # pi loss
        pi_loss = None
        if self.t % self.policy_update_period == 0:
            q1_pi = self.qf1(batch['obs'], pi_out.action).value
            q2_pi = self.qf2(batch['obs'], pi_out.action).value
            min_q_pi = torch.min(q1_pi, q2_pi)
            assert min_q_pi.shape == logp.shape
            pi_loss = (alpha * logp - min_q_pi).mean()

            # log pi loss about as frequently as other losses
            if self.t % self.log_period < self.policy_update_period:
                logger.add_scalar('loss/pi', pi_loss, self.t, time.time())

        if self.t % self.log_period < self.update_period:
            if self.automatic_entropy_tuning:
                logger.add_scalar('alg/log_alpha',
                                  self.log_alpha.detach().cpu().numpy(), self.t,
                                  time.time())
                scalars = {"target": self.target_entropy,
                           "entropy": -torch.mean(
                                        logp.detach()).cpu().numpy().item()}
                logger.add_scalars('alg/entropy', scalars, self.t, time.time())
            else:
                logger.add_scalar(
                        'alg/entropy',
                        -torch.mean(logp.detach()).cpu().numpy().item(),
                        self.t, time.time())
            logger.add_scalar('loss/qf1', qf1_loss, self.t, time.time())
            logger.add_scalar('loss/qf2', qf2_loss, self.t, time.time())
            logger.add_scalar('alg/qf1', q1.mean().detach().cpu().numpy(), self.t, time.time())
            logger.add_scalar('alg/qf2', q2.mean().detach().cpu().numpy(), self.t, time.time())
        return pi_loss, qf1_loss, qf2_loss

    def step(self):
        """Step optimization."""
        self.t += self.data_manager.step_until_update()
        if self.t % self.target_update_period == 0:
            soft_target_update(self.target_qf1, self.qf1,
                               self.target_smoothing_coef)
            soft_target_update(self.target_qf2, self.qf2,
                               self.target_smoothing_coef)

        if self.t % self.update_period == 0:
            batch = self.data_manager.sample(self.batch_size)

            pi_loss, qf1_loss, qf2_loss = self.loss(batch)

            # update
            if pi_loss:
                self.opt_pi.zero_grad()
                pi_loss.backward()
                self.opt_pi.step()

            self.opt_qf1.zero_grad()
            qf1_loss.backward()
            self.opt_qf1.step()

            self.opt_qf2.zero_grad()
            qf2_loss.backward()
            self.opt_qf2.step()

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
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'target_qf1': self.target_qf1.state_dict(),
            'target_qf2': self.target_qf2.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_qf1': self.opt_qf1.state_dict(),
            'opt_qf2': self.opt_qf2.state_dict(),
            'log_alpha': (self.log_alpha if self.automatic_entropy_tuning
                          else None),
            'opt_alpha': (self.opt_alpha.state_dict()
                          if self.automatic_entropy_tuning else None),
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
        self.qf1.load_state_dict(state_dict['qf1'])
        self.qf2.load_state_dict(state_dict['qf2'])
        self.target_qf1.load_state_dict(state_dict['target_qf1'])
        self.target_qf2.load_state_dict(state_dict['target_qf2'])

        self.opt_pi.load_state_dict(state_dict['opt_pi'])
        self.opt_qf1.load_state_dict(state_dict['opt_qf1'])
        self.opt_qf2.load_state_dict(state_dict['opt_qf2'])

        if state_dict['log_alpha']:
            with torch.no_grad():
                self.log_alpha.copy_(state_dict['log_alpha'])
            self.opt_alpha.load_state_dict(state_dict['opt_alpha'])
        misc.env_load_state_dict(self.env, state_dict['env'])
        self.t = state_dict['t']

        buffer_format = state_dict['buffer_format']
        buffer_state = dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                 'buffer.npz'),
                                    allow_pickle=True))
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
    from dl.rl.envs import make_env
    from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
    from dl.rl.modules import QFunction, Policy
    from dl.modules import TanhDiagGaussian
    import torch.nn.functional as F
    from functools import partial

    class PiBase(PolicyBase):
        """Policy network."""

        def build(self):
            """Build Network."""
            self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.dist = TanhDiagGaussian(32, self.action_space.shape[0])

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
        return make_env('LunarLanderContinuous-v2', nenv=nenv)

    def policy_fn(env):
        """Create a policy."""
        return Policy(PiBase(env.observation_space, env.action_space))

    def qf_fn(env):
        """Create a qfunction."""
        return QFunction(QFBase(env.observation_space, env.action_space))

    class TestSAC(unittest.TestCase):
        """Test case."""

        def test_sac(self):
            """Test."""
            sac = partial(SAC,
                          env_fn=env_fn,
                          policy_fn=policy_fn,
                          qf_fn=qf_fn,
                          learning_starts=300,
                          eval_num_episodes=1,
                          buffer_size=500,
                          target_update_period=100)
            train('logs', sac, maxt=1000, eval=False, eval_period=1000)
            alg = sac('logs')
            alg.load()
            shutil.rmtree('logs')

    unittest.main()
