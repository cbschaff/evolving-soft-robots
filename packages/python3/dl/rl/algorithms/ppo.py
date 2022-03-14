"""PPO RL algorithm.

https://arxiv.org/abs/1707.06347
"""
from dl.rl.envs import VecEpisodeLogger, VecRewardNormWrapper
from dl.rl.data_collection import RolloutDataManager
from dl.rl.modules import Policy
from dl.rl.util import rl_evaluate, rl_record, misc
from dl import logger, Algorithm, Checkpointer, nest
import gin
import os
import time
import torch
import torch.nn as nn
import numpy as np


class PPOActor(object):
    """Actor."""

    def __init__(self, pi):
        """Init."""
        self.pi = pi

    def __call__(self, ob, state_in=None):
        """Produce decision from model."""
        outs = self.pi(ob, state_in)
        data = {'action': outs.action,
                'value': outs.value,
                'logp': outs.dist.log_prob(outs.action),
                'dist': outs.dist.to_tensors()}
        if outs.state_out is not None:
            data['state'] = outs.state_out
        return data


@gin.configurable(blacklist=['logdir'])
class PPO(Algorithm):
    """PPO algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 policy_fn,
                 nenv=1,
                 optimizer=torch.optim.Adam,
                 batch_size=32,
                 rollout_length=None,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False,
                 epochs_per_rollout=10,
                 max_grad_norm=None,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 clip_param=0.2,
                 eval_num_episodes=1,
                 record_num_episodes=1,
                 gpu=True):
        """Init."""
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.env_fn = env_fn
        self.nenv = nenv
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.epochs_per_rollout = epochs_per_rollout
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_param = clip_param
        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')

        self.env = VecEpisodeLogger(VecRewardNormWrapper(env_fn(nenv=nenv),
                                                         gamma))

        self.pi = policy_fn(self.env).to(self.device)
        self.opt = optimizer(self.pi.parameters())
        self.data_manager = RolloutDataManager(
            self.env,
            PPOActor(self.pi),
            self.device,
            batch_size=batch_size,
            rollout_length=rollout_length,
            gamma=gamma,
            lambda_=lambda_,
            norm_advantages=norm_advantages)

        self.mse = nn.MSELoss(reduction='none')

        self.t = 0

    def compute_kl(self):
        """Compute KL divergence of new and old policies."""
        kl = 0
        n = 0
        for batch in self.data_manager.sampler():
            outs = self.pi(batch['obs'])
            old_dist = outs.dist.from_tensors(batch['dist'])
            k = old_dist.kl(outs.dist).mean()
            s = nest.flatten(batch['action'])[0].shape[0]
            kl = (n / (n + s)) * kl + (s / (n + s)) * k
            n += s
        return kl

    def loss(self, batch):
        """Compute loss."""
        outs = self.pi(batch['obs'])
        loss = {}

        # compute policy loss
        logp = outs.dist.log_prob(batch['action'])
        assert logp.shape == batch['logp'].shape
        ratio = torch.exp(logp - batch['logp'])
        assert ratio.shape == batch['atarg'].shape
        ploss1 = ratio * batch['atarg']
        ploss2 = torch.clamp(ratio, 1.0-self.clip_param,
                             1.0+self.clip_param) * batch['atarg']
        pi_loss = -torch.min(ploss1, ploss2).mean()
        loss['pi'] = pi_loss

        # compute value loss
        vloss1 = 0.5 * self.mse(outs.value, batch['vtarg'])
        vpred_clipped = batch['vpred'] + (
            outs.value - batch['vpred']).clamp(-self.clip_param,
                                               self.clip_param)
        vloss2 = 0.5 * self.mse(vpred_clipped, batch['vtarg'])
        vf_loss = torch.max(vloss1, vloss2).mean()
        loss['value'] = vf_loss

        # compute entropy loss
        ent_loss = outs.dist.entropy().mean()
        loss['entropy'] = ent_loss

        tot_loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
        loss['total'] = tot_loss
        return loss

    def step(self):
        """Compute rollout, loss, and update model."""
        self.pi.train()
        self.t += self.data_manager.rollout()
        losses = {}
        for _ in range(self.epochs_per_rollout):
            for batch in self.data_manager.sampler():
                self.opt.zero_grad()
                loss = self.loss(batch)
                if losses == {}:
                    losses = {k: [] for k in loss}
                for k, v in loss.items():
                    losses[k].append(v.detach().cpu().numpy())
                loss['total'].backward()
                if self.max_grad_norm:
                    norm = nn.utils.clip_grad_norm_(self.pi.parameters(),
                                                    self.max_grad_norm)
                    logger.add_scalar('alg/grad_norm', norm, self.t,
                                      time.time())
                    logger.add_scalar('alg/grad_norm_clipped',
                                      min(norm, self.max_grad_norm),
                                      self.t, time.time())
                self.opt.step()
        for k, v in losses.items():
            logger.add_scalar(f'loss/{k}', np.mean(v), self.t, time.time())

        data = self.data_manager.storage.get_rollout()
        value_error = data['vpred'].data - data['q_mc'].data
        logger.add_scalar('alg/value_error_mean',
                          value_error.mean().cpu().numpy(), self.t, time.time())
        logger.add_scalar('alg/value_error_std',
                          value_error.std().cpu().numpy(), self.t, time.time())

        logger.add_scalar('alg/kl', self.compute_kl(), self.t, time.time())
        return self.t

    def evaluate(self):
        """Evaluate model."""
        self.pi.eval()
        misc.set_env_to_eval_mode(self.env)

        # Eval policy
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval',
                               self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(self.env, self.pi, self.eval_num_episodes,
                            outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'],
                          self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'],
                          self.t, time.time())

        # Record policy
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video',
                               self.ckptr.format.format(self.t) + '.mp4')
        rl_record(self.env, self.pi, self.record_num_episodes, outfile,
                  self.device)

        self.pi.train()
        misc.set_env_to_train_mode(self.env)

    def save(self):
        """State dict."""
        state_dict = {
            'pi': self.pi.state_dict(),
            'opt': self.opt.state_dict(),
            'env': misc.env_state_dict(self.env),
            't': self.t
        }
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        """Load state dict."""
        state_dict = self.ckptr.load(t)
        if state_dict is None:
            self.t = 0
            return self.t
        self.pi.load_state_dict(state_dict['pi'])
        self.opt.load_state_dict(state_dict['opt'])
        misc.env_load_state_dict(self.env, state_dict['env'])
        self.t = state_dict['t']
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
    from dl.rl.modules import ActorCriticBase
    from dl.rl.util import conv_out_shape
    from dl.modules import Categorical
    import torch.nn.functional as F
    from functools import partial

    class NatureDQN(ActorCriticBase):
        """Deep network from https://www.nature.com/articles/nature14236."""

        def build(self):
            """Build network."""
            self.conv1 = nn.Conv2d(4, 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            shape = self.observation_space.shape[1:]
            for c in [self.conv1, self.conv2, self.conv3]:
                shape = conv_out_shape(shape, c)
            self.nunits = 64 * np.prod(shape)
            self.fc = nn.Linear(self.nunits, 512)
            self.vf = nn.Linear(512, 1)
            self.dist = Categorical(512, self.action_space.n)

        def forward(self, x):
            """Forward."""
            x = x.float() / 255.
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc(x.view(-1, self.nunits)))
            return self.dist(x), self.vf(x)

    class TestPPO(unittest.TestCase):
        """Test case."""

        def test_feed_forward_ppo(self):
            """Test feed forward ppo."""
            def env_fn(nenv):
                return make_atari_env('Pong', nenv, frame_stack=4)

            def policy_fn(env):
                return Policy(NatureDQN(env.observation_space,
                                        env.action_space))

            ppo = partial(PPO, env_fn=env_fn, policy_fn=policy_fn)
            train('test', ppo, maxt=1000, eval=True, eval_period=1000)
            shutil.rmtree('test')

    unittest.main()
