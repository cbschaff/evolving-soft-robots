"""SAC algorithm.

https://arxiv.org/abs/1801.01290
"""
from dl.modules import CatDist
from dl import logger
import gin
import time
import torch
import torch.nn as nn
from dl.rl.algorithms.sac import SAC


@gin.configurable(blacklist=['logdir'])
class SACDiscrete(SAC):
    """Discerete action space version of the SAC algorithm."""

    def loss(self, batch):
        """Loss function."""
        dist = self.pi(batch['obs']).dist
        q1 = self.qf1(batch['obs'], batch['action']).value
        q2 = self.qf2(batch['obs'], batch['action']).value

        # alpha loss
        if self.automatic_entropy_tuning:
            ent_error = dist.entropy() - self.target_entropy
            alpha_loss = self.log_alpha * ent_error.detach().mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
            alpha_loss = 0

        # qf loss
        with torch.no_grad():
            next_dist = self.pi(batch['next_obs']).dist
            q1_next = self.target_qf1(batch['next_obs']).qvals
            q2_next = self.target_qf2(batch['next_obs']).qvals
            qmin = torch.min(q1_next, q2_next)
            # explicitly compute the expectation over next actions
            qnext = torch.sum(qmin * next_dist.probs, dim=1) + alpha * next_dist.entropy()
            qtarg = batch['reward'] + (1.0 - batch['done']) * self.gamma * qnext

        assert qtarg.shape == q1.shape
        assert qtarg.shape == q2.shape
        qf1_loss = self.mse_loss(q1, qtarg)
        qf2_loss = self.mse_loss(q2, qtarg)

        # pi loss
        pi_loss = None
        if self.t % self.policy_update_period == 0:
            with torch.no_grad():
                q1_pi = self.qf1(batch['obs']).qvals
                q2_pi = self.qf2(batch['obs']).qvals
                min_q_pi = torch.min(q1_pi, q2_pi)
            assert min_q_pi.shape == dist.logits.shape
            target_dist = CatDist(logits=min_q_pi)
            pi_dist = CatDist(logits=alpha * dist.logits)
            pi_loss = pi_dist.kl(target_dist).mean()

            # log pi loss about as frequently as other losses
            if self.t % self.log_period < self.policy_update_period:
                logger.add_scalar('loss/pi', pi_loss, self.t, time.time())

        if self.t % self.log_period < self.update_period:
            if self.automatic_entropy_tuning:
                logger.add_scalar('alg/log_alpha',
                                  self.log_alpha.detach().cpu().numpy(), self.t,
                                  time.time())
                scalars = {
                    "target": self.target_entropy,
                    "entropy": dist.entropy().mean().detach().cpu().numpy().item()
                }
                logger.add_scalars('alg/entropy', scalars, self.t, time.time())
            else:
                logger.add_scalar(
                        'alg/entropy',
                        dist.entropy().mean().detach().cpu().numpy().item(),
                        self.t, time.time())
            logger.add_scalar('loss/qf1', qf1_loss, self.t, time.time())
            logger.add_scalar('loss/qf2', qf2_loss, self.t, time.time())
            logger.add_scalar('alg/qf1', q1.mean().detach().cpu().numpy(), self.t, time.time())
            logger.add_scalar('alg/qf2', q2.mean().detach().cpu().numpy(), self.t, time.time())
        return pi_loss, qf1_loss, qf2_loss


if __name__ == '__main__':
    import unittest
    import shutil
    from dl import train
    from dl.rl.envs import make_env
    from dl.rl.modules import PolicyBase, DiscreteQFunctionBase
    from dl.rl.modules import QFunction, Policy
    from dl.modules import Categorical
    import torch.nn.functional as F
    from functools import partial

    class PiBase(PolicyBase):
        """Policy network."""

        def build(self):
            """Build Network."""
            self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.dist = Categorical(32, self.action_space.n)

        def forward(self, x):
            """Forward."""
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.dist(x)

    class QFBase(DiscreteQFunctionBase):
        """Q network."""

        def build(self):
            """Build Network."""
            nin = self.observation_space.shape[0]
            self.fc1 = nn.Linear(nin, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.qvalue = nn.Linear(32, self.action_space.n)

        def forward(self, x):
            """Forward."""
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.qvalue(x)

    def env_fn(nenv):
        """Environment function."""
        return make_env('LunarLander-v2', nenv=nenv)

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
            sac = partial(SACDiscrete,
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
