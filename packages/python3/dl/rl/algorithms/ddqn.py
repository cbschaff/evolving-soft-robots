"""Double DQN algorithm.

https://arxiv.org/abs/1509.06461
"""

from dl.rl.algorithms import DQN
import gin


@gin.configurable(blacklist=['logdir'])
class DoubleDQN(DQN):
    """Double DQN."""

    def _compute_target(self, rew, next_ob, done):
        next_ac = self.qf(next_ob).max_a
        qtarg = self.qf_targ(next_ob, next_ac).value
        return rew + (1.0 - done) * self.gamma * qtarg
