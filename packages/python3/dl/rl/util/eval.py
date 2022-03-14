"""Evaluation for RL Environments."""
import os
import json
import tempfile
import time
import numpy as np
from imageio import imwrite
import subprocess as sp
from dl import logger
from dl import nest
from dl.rl.util import ensure_vec_env
import torch


class Actor(object):
    """Wrap actor to convert actions to numpy."""

    def __init__(self, net, device):
        """Init."""
        self.net = net
        self.state = None
        self.device = device

    def __call__(self, ob):
        """__call__."""
        with torch.no_grad():
            def _to_torch(o):
                return torch.from_numpy(o).to(self.device)
            ob = nest.map_structure(_to_torch, ob)
            if self.state is None:
                out = self.net(ob)
            else:
                out = self.net(ob, self.state)
            if hasattr(out, 'state_out'):
                self.state = out.state_out
            return nest.map_structure(lambda x: x.cpu().numpy(), out.action)


def rl_evaluate(env, actor, nepisodes, outfile=None, device='cpu',
                save_info=False):
    """Compute episode stats for an environment and actor.

    If the environment has an EpisodeInfo Wrapper, rl_record will use that
    to determine episode termination.
    Args:
        env: A Gym environment
        actor: A torch.nn.Module whose input is an observation and output has a
               '.action' attribute.
        nepisodes: The number of episodes to run.
        outfile: Where to save results (if provided)
        device: The device which contains the actor.
        save_info: Save the info dict with results.
    Returns:
        A dict of episode stats

    """
    if nepisodes == 0:
        return
    env = ensure_vec_env(env)
    ep_lengths = []
    ep_rewards = []
    all_infos = []
    actor = Actor(actor, device)
    while len(ep_lengths) < nepisodes:
        _dones = np.zeros(env.num_envs, dtype=np.bool)
        all_infos.extend([[] for _ in range(env.num_envs)])
        obs = env.reset()
        while not np.all(_dones):
            obs, rs, dones, infos = env.step(actor(obs))
            for i, _ in enumerate(dones):
                dones[i] = infos[i]['episode_info']['done']
                if not _dones[i] and save_info:
                    all_infos[-i-1].append(infos[i])
            _dones = np.logical_or(dones, _dones)

        # save results
        for i, info in enumerate(infos):
            ep_lengths.append(infos[i]['episode_info']['length'])
            ep_rewards.append(infos[i]['episode_info']['reward'])

    outs = {
        'episode_lengths': ep_lengths,
        'episode_rewards': ep_rewards,
        'mean_length': np.mean(ep_lengths),
        'mean_reward': np.mean(ep_rewards),
    }
    if save_info:
        outs['info'] = all_infos
    if outfile:
        torch.save(outs, outfile)
    return outs


def rl_record(env, actor, nepisodes, outfile, device='cpu', fps=30):
    """Compute episode stats for an environment and actor.

    If the environment has an EpisodeInfo Wrapper, rl_record will use that to
    determine episode termination.
    Args:
        env: A Gym environment
        actor: A callable whose input is an observation and output has a
               '.action' attribute.
        nepisodes: The number of episodes to run.
        outfile: Where to save the video.
        device: The device which contains the actor.
        fps: The frame rate of the video.
    Returns:
        A dict of episode stats

    """
    if nepisodes == 0:
        return
    env = ensure_vec_env(env)
    tmpdir = os.path.join(tempfile.gettempdir(),
                          'video_' + str(time.monotonic()))
    os.makedirs(tmpdir)
    id = 0
    actor = Actor(actor, device)
    episodes = 0

    def write_ims(ims, id):
        for im in ims:
            imwrite(os.path.join(tmpdir, '{:05d}.png'.format(id)), im)
            id += 1
        return id

    while episodes < nepisodes:
        obs = env.reset()
        nenv = min(env.num_envs, nepisodes)
        _dones = np.zeros(nenv, dtype=np.bool)
        ims = [[] for _ in range(nenv)]

        # collect images
        try:
            rgbs = env.get_images()
        except Exception as e:
            logger.log(e)
            logger.log("Error while rendering.")
            return
        for i in range(nenv):
            ims[i].append(rgbs[i])

        # rollout episodes
        while not np.all(_dones):
            obs, r_, dones, infos = env.step(actor(obs))
            for i, _ in enumerate(dones):
                if 'episode_info' in infos[i]:
                    dones[i] = infos[i]['episode_info']['done']

            # collect images
            try:
                rgbs = env.get_images()
            except Exception:
                logger.log("Error while rendering.")
                return
            for i in range(nenv):
                if not _dones[i]:
                    ims[i].append(rgbs[i])
            _dones = np.logical_or(dones[:nenv], _dones)

        # save images
        for i in range(nenv):
            if episodes < nepisodes:
                id = write_ims(ims[i], id)
                ims[i] = []
                episodes += 1

    sp.call(['ffmpeg', '-r', str(fps), '-f', 'image2', '-i',
             os.path.join(tmpdir, '%05d.png'), '-vcodec', 'libx264',
             '-pix_fmt', 'yuv420p', os.path.join(tmpdir, 'out.mp4')])
    sp.call(['mv', os.path.join(tmpdir, 'out.mp4'), outfile])
    sp.call(['rm', '-rf', tmpdir])


if __name__ == '__main__':
    import unittest
    import gym
    from dl.rl.envs import EpisodeInfo
    from collections import namedtuple

    class Test(unittest.TestCase):
        """Test."""

        def test_eval(self):
            """Test."""
            env = EpisodeInfo(gym.make('CartPole-v1'))

            def actor(ob):
                ac = torch.from_numpy(np.array(env.action_space.sample()))[None]
                return namedtuple('test', ['action', 'state_out'])(
                    action=ac, state_out=None)

            stats = rl_evaluate(env, actor, 10, outfile='./out.pt',
                                save_info=True)
            assert len(stats['episode_lengths']) >= 10
            assert len(stats['episode_rewards']) >= 10
            assert len(stats['episode_rewards']) == len(
                    stats['episode_lengths'])
            assert np.mean(stats['episode_lengths']) == stats['mean_length']
            assert np.mean(stats['episode_rewards']) == stats['mean_reward']
            env.close()
            os.remove('./out.pt')

        def test_record(self):
            """Test record."""
            env = EpisodeInfo(gym.make('CartPole-v1'))

            def actor(ob):
                ac = torch.from_numpy(np.array(env.action_space.sample()))[None]
                return namedtuple('test', ['action', 'state_out'])(
                    action=ac, state_out=None)

            rl_record(env, actor, 10, './video.mp4')
            os.remove('./video.mp4')
            env.close()

    unittest.main()
