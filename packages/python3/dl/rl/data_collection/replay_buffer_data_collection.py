"""Collect data from an environment and store it in a replay buffer."""
from dl.rl.data_collection import ReplayBuffer, BatchedReplayBuffer
from dl.rl.util import ensure_vec_env
from dl import nest
import torch
import numpy as np


class ReplayBufferDataManager(object):
    """Collects data from environments and stores it in a RolloutStorage.

    The resposibilities of this class are:
        - Replay Buffer creation, saving, and loading.
        - Adding data to the replay buffer.
        - Stepping the environment.

    act_fn:
        A callable which takes in the observation and returns:
            - a dictionary with the data to store in the replay buffer.
              'action' must be in the dict.
    """

    def __init__(self,
                 buffer,
                 env,
                 act_fn,
                 device,
                 learning_starts=1000,
                 update_period=1):
        """Init."""
        self.env = ensure_vec_env(env)
        if self.env.num_envs > 1 and not isinstance(buffer, BatchedReplayBuffer):
            raise ValueError("when num_envs > 1, you must pass a BatchedReplayBuffer"
                             " to the ReplayBufferDataManager.")
        if not isinstance(buffer, BatchedReplayBuffer):
            buffer = BatchedReplayBuffer(buffer)
        if self.env.num_envs != buffer.n:
            raise ValueError(f"Found {self.env.num_envs} envs and {buffer.n} "
                             "buffers. The number of envs must be equal to the "
                             "number of buffers!")
        self.act = act_fn
        self.buffer = buffer
        self.device = device
        self.learning_starts = learning_starts
        self.update_period = update_period
        self._ob = None

    def manual_reset(self):
        """Update buffer on manual environment reset."""
        self.buffer.env_reset()
        self._ob = self.env.reset()

    def env_step_and_store_transition(self):
        """Step env and store transition in replay buffer."""
        if self._ob is None:
            self.manual_reset()

        def _to_torch(ob):
            return torch.from_numpy(ob).to(self.device)

        idx = self.buffer.store_observation(self._ob)
        ob = self.buffer.encode_recent_observation()
        with torch.no_grad():
            data = self.act(nest.map_structure(_to_torch, ob))
            data = nest.map_structure(lambda x: x.cpu().numpy(), data)
        self._ob, r, done, _ = self.env.step(data['action'])
        data['reward'] = r
        data['done'] = done
        self.buffer.store_effect(idx, data)
        if np.any(done):
            self._ob = self.env.reset(force=False)

    def step_until_update(self):
        """Step env untiil update."""
        t = 0
        for _ in range(self.update_period):
            self.env_step_and_store_transition()
            t += self.env.num_envs
        while self.buffer.num_in_buffer < min(self.learning_starts,
                                              self.buffer.size):
            self.env_step_and_store_transition()
            t += self.env.num_envs
        return t

    def sample(self, *args, **kwargs):
        """Sample batch from replay buffer."""
        batch = self.buffer.sample(*args, **kwargs)

        def _to_torch(data):
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.device)
            else:
                return data
        return nest.map_structure(_to_torch, batch)


if __name__ == '__main__':
    import unittest
    from dl.rl.modules import QFunction, DiscreteQFunctionBase
    from dl.rl.envs import make_env
    from dl.modules import FeedForwardNet
    from gym.spaces import Tuple
    from dl.rl.util.vec_env import VecEnvWrapper

    class FeedForwardBase(DiscreteQFunctionBase):
        """Feed forward network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32, self.action_space.n])

        def forward(self, ob):
            """Forward."""
            if isinstance(ob, list):
                ob = ob[0]
            return self.net(ob.float())

    class BufferActor(object):
        """Actor."""

        def __init__(self, pi):
            """init."""
            self.pi = pi

        def __call__(self, ob):
            """act."""
            outs = self.pi(ob)
            data = {'action': outs.action}
            if isinstance(ob, (list, tuple)):
                data['key1'] = torch.zeros_like(ob[0])
            else:
                data['key1'] = torch.zeros_like(ob)
            return data

    class NestedVecObWrapper(VecEnvWrapper):
        """Nest observations."""

        def __init__(self, venv):
            """Init."""
            super().__init__(venv)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space])

        def reset(self):
            """Reset."""
            ob = self.venv.reset()
            return (ob, ob)

        def step_wait(self):
            """Step."""
            ob, r, done, info = self.venv.step_wait()
            return (ob, ob), r, done, info

    class TestReplayBufferDataManager(unittest.TestCase):
        """Test case."""

        def test(self):
            """Test."""
            env = make_env('CartPole-v1')
            qf = QFunction(FeedForwardBase(env.observation_space,
                                           env.action_space))
            buffer = ReplayBuffer(2000, 1)
            data_manager = ReplayBufferDataManager(
                    buffer, env, act_fn=BufferActor(qf),
                    device='cpu', learning_starts=50, update_period=2)

            for _ in range(11):
                data_manager.step_until_update()
            assert buffer.num_in_buffer == 70

            batch = data_manager.sample(32)
            data_manager.act(batch['obs'])
            assert batch['action'].shape == batch['reward'].shape
            assert batch['action'].shape == batch['done'].shape
            if isinstance(batch['obs'], list):
                assert batch['obs'][0].shape == batch['next_obs'][0].shape
                assert len(batch['obs'][0].shape) == 2
            else:
                assert batch['obs'].shape == batch['next_obs'].shape
                assert len(batch['obs'].shape) == 2
            assert len(batch['action'].shape) == 1

        def test_nested_ob(self):
            """Test."""
            env = make_env('CartPole-v1')
            qf = QFunction(FeedForwardBase(env.observation_space,
                                           env.action_space))
            env = NestedVecObWrapper(env)
            buffer = ReplayBuffer(2000, 1)
            data_manager = ReplayBufferDataManager(
                    buffer, env, act_fn=BufferActor(qf),
                    device='cpu', learning_starts=50, update_period=2)

            for _ in range(11):
                data_manager.step_until_update()
            assert buffer.num_in_buffer == 70

            batch = data_manager.sample(32)
            data_manager.act(batch['obs'])
            assert batch['action'].shape == batch['reward'].shape
            assert batch['action'].shape == batch['done'].shape
            if isinstance(batch['obs'], list):
                assert batch['obs'][0].shape == batch['next_obs'][0].shape
                assert len(batch['obs'][0].shape) == 2
            else:
                assert batch['obs'].shape == batch['next_obs'].shape
                assert len(batch['obs'].shape) == 2
            assert len(batch['action'].shape) == 1

    unittest.main()
