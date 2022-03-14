"""Environment Wrapper for normalizing observations."""
from dl.rl.util.vec_env import VecEnvWrapper
from dl import nest
from dl.rl.util import get_ob_norm
from gym import Wrapper
import numpy as np
import gin
import wandb


@gin.configurable(blacklist=['norm'])
class VecObsNormWrapper(VecEnvWrapper):
    """Observation normalization for vecorized environments.

    Collects data from a random policy and computes a fixed normalization.
    Computing norm params is done lazily when normalization constants
    are needed and unknown.
    """

    def __init__(self, venv,
                 norm=True,
                 steps=10000,
                 mean=None,
                 std=None,
                 eps=1e-2,
                 log=True,
                 log_prob=0.01):
        """Init."""
        super().__init__(venv)
        self.steps = steps
        self.should_norm = norm
        self.eps = eps
        self.log = log
        self.log_prob = log_prob
        self.t = 0
        self._eval = False
        self.mean = None
        self.std = None
        self._dones = np.zeros(self.num_envs, dtype=np.bool)

        if mean is not None and std is not None:
            if not nest.has_same_structure(mean, std):
                raise ValueError("mean and std must have the same structure.")
            self.mean = mean
            self.std = nest.map_structure(
                        lambda x: np.maximum(x, self.eps), std)

    def _env(self):
        return self.env if hasattr(self, 'env') else self.venv

    def find_norm_params(self):
        """Calculate mean and std with a random policy to collect data."""
        self.mean, self.std = get_ob_norm(self._env(), self.steps, self.eps)

    def _normalize(self, obs):
        if not self.should_norm:
            return obs
        if self.mean is None or self.std is None:
            self.find_norm_params()
        obs = nest.map_structure(np.asarray, obs)
        obs = nest.map_structure(np.float32, obs)
        if not nest.has_same_structure(self.mean, obs):
            raise ValueError("mean and obs do not have the same structure!")

        def norm(item):
            ob, mean, std = item
            if mean is not None:
                return (ob - mean) / std
            else:
                return ob
        return nest.map_structure(norm, nest.zip_structure(obs, self.mean,
                                                           self.std))

    def state_dict(self):
        """State dict."""
        return {
            'mean': self.mean,
            'std': self.std,
            't': self.t,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.t = state_dict['t']

    def norm_and_log(self, obs):
        """Norm observations and log."""
        obs = self._normalize(obs)
        if not self._eval and self.log and self.log_prob > np.random.rand():
            flat_ob = np.concatenate([
                ob.ravel() for ob in nest.flatten(obs) if ob is not None
            ])
            wandb.log({'train/observation': wandb.Histogram(flat_ob),
                       'train/step': self.t})
        return obs

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables logging and stops counting steps.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode counts steps and logs obs distribution if self.log is True.
        """
        self._eval = False

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.venv.step(action)
        if not self._eval:
            self.t += np.sum(np.logical_not(self._dones))
        self._dones = np.logical_or(dones, self._dones)
        return self.norm_and_log(obs), rews, dones, infos

    def reset(self, force=True):
        """Reset."""
        obs = self.venv.reset(force=force)
        self._dones[:] = False
        return self._normalize(obs)

    def step_wait(self):
        return self.venv.step_wait()


class EpisodeInfo(Wrapper):
    """Return episode information in step function.

    Pass episode stats through the info dict returned by step().
    If placed before wrappers which modify episode length and reward,
    this will provide easy access to unmodified episode stats.
    """

    def __init__(self, env):
        """Init."""
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0
        self.needs_reset = True

    def reset(self):
        """Reset."""
        self.episode_reward = 0
        self.episode_length = 0
        self.needs_reset = False
        return self.env.reset()

    def step(self, action):
        """Step."""
        assert not self.needs_reset, (
            "Can't step environment when the episode ends. "
            "Please call reset() first.")
        ob, r, done, info = self.env.step(action)
        if done:
            self.needs_reset = True
        self.episode_reward += r
        self.episode_length += 1
        assert 'episode_info' not in info, (
            f"Can't save episode data. Another EpisodeInfo Wrapper exists.")
        info['episode_info'] = {}
        info['episode_info']['reward'] = self.episode_reward
        info['episode_info']['length'] = self.episode_length
        info['episode_info']['done'] = done
        return ob, r, done, info


class VecEpisodeLogger(VecEnvWrapper):
    """EpisodeLogger for vecorized environments."""

    def __init__(self, venv, tstart=0):
        """Init."""
        super().__init__(venv)
        self.t = tstart
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)
        self._eval = False
        self._dones = np.zeros(self.num_envs, dtype=np.bool)
        wandb.define_metric('train/episode_reward', summary='max', step_metric="train/step")
        wandb.define_metric('train/episode_reward', summary='last', step_metric="train/step")

    def reset(self, force=True):
        """Reset."""
        obs = self.venv.reset(force=force)
        if force:
            self.rews[:] = 0
            self.lens[:] = 0
        else:
            self.rews[self._dones] = 0
            self.lens[self._dones] = 0
        self._dones[:] = False
        return obs

    def step(self, action):
        """Step."""
        obs, rews, dones, infos = self.venv.step(action)
        if not self._eval:
            self.t += np.sum(np.logical_not(self._dones))
        for i, d in enumerate(self._dones):  # handle synced resets
            if not d:
                self.lens[i] += 1
                self.rews[i] += rews[i]
            else:
                assert dones[i]
        for i, done in enumerate(dones):
            if done and not self._dones[i]:
                if not self._eval:
                    data = {
                        'train/episode_reward': self.rews[i],
                        'train/episode_length': self.lens[i]
                    }
                    data['train/step'] = self.t
                    wandb.log(data)
                self.lens[i] = 0
                self.rews[i] = 0.
        # log unwrapped episode stats if they exist
        if 'episode_info' in infos[0]:
            for i, info in enumerate(infos):
                epinfo = info['episode_info']
                if epinfo['done'] and not self._eval and not self._dones[i]:
                    data = {
                        'train/unwrapped_episode_reward': epinfo['reward'],
                        'train/unwrapped_episode_length': epinfo['length']
                    }
                    data['train/step'] = self.t
                    wandb.log(data)
        self._dones = np.logical_or(dones, self._dones)

        return obs, rews, dones, infos

    def step_wait(self):
        """Step wait."""
        return self.venv.step_wait()

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables logging and stops counting steps.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode counts steps and logs episode stats.
        """
        self._eval = False

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']
