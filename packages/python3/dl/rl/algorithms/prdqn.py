"""Prioritized Replay DQN algorithm.

https://arxiv.org/abs/1511.05952
"""

from dl.rl.algorithms import DoubleDQN
from dl.rl.data_collection import PrioritizedReplayBuffer
from dl.rl.util import LinearSchedule
import gin
import torch
import time
from dl import logger


@gin.configurable(blacklist=['logdir'])
class PrioritizedReplayDQN(DoubleDQN):
    """Prioritized Replay DQN."""

    def __init__(self,
                 logdir,
                 replay_alpha=0.6,
                 replay_beta=0.4,
                 t_beta_max=int(1e7),
                 **kwargs):
        """Init."""
        super().__init__(logdir, **kwargs)
        self.buffer.buffers = [
            PrioritizedReplayBuffer(buf, alpha=replay_alpha)
            for buf in self.buffer.buffers
        ]
        self.beta_schedule = LinearSchedule(t_beta_max, 1.0, replay_beta)

    def _get_batch(self):
        beta = self.beta_schedule.value(self.t)
        return self.data_manager.sample(self.batch_size, beta)

    def loss(self, batch):
        """Loss."""
        q = self.qf(batch['obs'], batch['action']).value

        with torch.no_grad():
            target = self._compute_target(batch['reward'], batch['next_obs'],
                                          batch['done'])

        assert target.shape == q.shape
        err = self.criterion(target, q)
        self.buffer.update_priorities(batch['idxes'],
                                      err.detach().cpu().numpy() + 1e-6)
        assert err.shape == batch['weights'].shape
        err = batch['weights'] * err
        loss = err.mean()

        if self.t % self.log_period < self.update_period:
            logger.add_scalar('alg/maxq', torch.max(q).detach().cpu().numpy(),
                              self.t, time.time())
            logger.add_scalar('alg/loss', loss.detach().cpu().numpy(), self.t,
                              time.time())
            logger.add_scalar('alg/epsilon',
                              self.eps_schedule.value(self._actor.t),
                              self.t, time.time())
            logger.add_scalar('alg/beta', self.beta_schedule.value(self.t),
                              self.t, time.time())
        return loss
