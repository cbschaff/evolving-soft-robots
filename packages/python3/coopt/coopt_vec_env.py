"""SubprocVecEnv from cbschaff/dl but with additional interface for sending design parmaters."""

import multiprocessing as mp

import numpy as np
from dl.rl.util.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
from dl import rng, nest
from dl.rl import env_state_dict, env_load_state_dict


def worker(remote, parent_remote, env_fn_wrapper, seed):

    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        return ob, reward, done, info

    env = env_fn_wrapper.x()
    parent_remote.close()
    rng.seed(seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(step_env(env, data))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((env.observation_space, env.action_space,
                                                env.parameter_space, env.spec)))
            elif cmd == 'get_rng':
                remote.send(rng.get_state(cuda=False))
            elif cmd == 'get_state':
                remote.send(env_state_dict(env))
            elif cmd == 'set_rng':
                rng.set_state(data)
            elif cmd == 'set_state':
                env_load_state_dict(env, data)
            elif cmd == 'set_design':
                env.init_scene(data)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
        # allow main process to save state if needed.
        count = 0.
        while count < 2:
            cmd, data = remote.recv()
            if cmd == 'get_rng':
                count += 1
                remote.send(rng.get_state(cuda=False))
            elif cmd == 'get_state':
                count += 1
                remote.send(env_state_dict(env))
            elif cmd == 'close':
                remote.close()
                break
            elif count == 0:
                continue

    finally:
        env.close()


class CoOptSubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.reward_range = None
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), ind))
                   for ind, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, parameter_space, self.spec = self.remotes[0].recv().x
        self.parameter_space = parameter_space
        self.viewer = None
        self._dones = [False for _ in range(nenvs)]
        self._last_transitions = [None for _ in range(nenvs)]
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def set_designs(self, designs):
        for i, remote in enumerate(self.remotes):
            remote.send(('set_design', designs[i]))
        self._dones = [True for _ in range(self.nenvs)]

    def step_async(self, actions):
        self._assert_not_closed()

        def _numpy_check(ac):
            if not isinstance(ac, np.ndarray):
                raise ValueError("You must pass actions as nested numpy arrays"
                                 " to SubprocVecEnv.")
        nest.map_structure(_numpy_check, actions)
        for i, remote in enumerate(self.remotes):
            if self._dones[i]:
                continue
            action = nest.map_structure(lambda ac: ac[i], actions)
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = []
        for i, remote in enumerate(self.remotes):
            if not self._dones[i]:
                self._last_transitions[i] = remote.recv()
            results.append(self._last_transitions[i])
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        for i, info in enumerate(infos):
            info['active'] = self._dones[i]
            self._dones[i] = self._last_transitions[i][2]
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, force=True):
        if not force:
            return self._reset_done_envs()
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        self._dones = [False for _ in range(self.nenvs)]
        self._last_transitions = [None for _ in range(self.nenvs)]
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs)

    def _reset_done_envs(self):
        self._assert_not_closed()
        for i, remote in enumerate(self.remotes):
            if self._dones[i] or self._last_transitions[i] is None:
                remote.send(('reset', None))
        obs = []
        for i, remote in enumerate(self.remotes):
            if self._dones[i] or self._last_transitions[i] is None:
                obs.append(remote.recv())
                self._dones[i] = False
                self._last_transitions[i] = None
            else:
                obs.append(self._last_transitions[i][0])
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    return nest.map_structure(np.stack, nest.zip_structure(*obs))
