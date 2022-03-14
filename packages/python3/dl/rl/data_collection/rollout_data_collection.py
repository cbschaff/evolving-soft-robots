"""Code for storing and iterating over rollout data."""
from dl.rl.data_collection import RolloutStorage
from dl.rl.util import ensure_vec_env
from dl import nest
import torch


class RolloutDataManager(object):
    """Collects data from environments and stores it in a RolloutStorage.

    The resposibilities of this class are:
        - Handle storage of rollout data
        - Handle computing rollouts
        - Handle batching and iterating over rollout data

    act_fn:
        A callable which takes in the observation, recurrent state and returns:
            - a dictionary with the data to store in the rollout. 'action'
              and 'value' must be in the dict. Recurrent states must
              be nested under the 'state' key. All values except
              data['state'] must be pytorch Tensors.
    """

    def __init__(self,
                 env,
                 act_fn,
                 device,
                 batch_size=32,
                 rollout_length=None,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False):
        """Init."""
        self.env = ensure_vec_env(env)
        self.nenv = self.env.num_envs
        self.act = act_fn
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_advantages = norm_advantages
        self.rollout_length = rollout_length

        if rollout_length:
            self.storage = RolloutStorage(self.nenv, device=self.device,
                                          num_steps=self.rollout_length)
        else:
            self.storage = RolloutStorage(self.nenv, device=self.device)
        self._initialized = False

    def init_rollout_storage(self):
        """Initialize rollout storage."""
        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)
        self._ob = nest.map_structure(_to_torch, self.env.reset())
        data = self.act(self._ob)
        if 'action' not in data:
            raise ValueError('the key "action" must be in the dict returned '
                             'act_fn')
        if 'value' not in data:
            raise ValueError('the key "value" must be in the dict returned '
                             'act_fn')
        state = None
        if 'state' in data:
            state = data['state']

        if state is None:
            self.init_state = None
            self.recurrent = False
        else:
            self.recurrent = True

            def _init_state(s):
                return torch.zeros(size=s.shape, device=self.device,
                                   dtype=s.dtype)

            self.init_state = nest.map_structure(_init_state, state)
            self._state = self.init_state

        self._initialized = True

    def _reset(self):
        if not self._initialized:
            self.init_rollout_storage()

        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)
        self._ob = nest.map_structure(_to_torch, self.env.reset(force=False))
        if self.rollout_length is None:
            # In this case, always rollout until the end of the episode
            # else, state reseting happens in rollout_step.
            self._state = self.init_state
        self.storage.reset()
        self._step = 0
        self._dones = None

    def rollout_step(self):
        """Compute one environment step."""
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state)
            else:
                outs = self.act(self._ob, state_in=None)
        cpu_action = nest.map_structure(lambda ac: ac.cpu().numpy(),
                                        outs['action'])
        ob, r, done, infos = self.env.step(cpu_action)
        data = {}
        data['obs'] = self._ob
        data['action'] = outs['action']
        data['reward'] = torch.from_numpy(r).float().to(self.device)
        data['done'] = torch.from_numpy(done).to(self.device)
        data['vpred'] = outs['value']
        for key in outs:
            if key not in ['action', 'value', 'state']:
                data[key] = outs[key]

        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)

        self._ob = nest.map_structure(_to_torch, ob)
        if self.recurrent:
            self._state = outs['state']

        self._step += 1
        truncated = self._get_truncated_envs(infos)
        if self._dones is not None:
            prev_step_not_done = torch.logical_not(self._dones)
            truncated = truncated & prev_step_not_done
        at_end_of_rollout = (self.rollout_length
                             and self._step >= self.rollout_length)
        if at_end_of_rollout or torch.any(truncated):
            next_vpred = self._get_next_value()

        if self.rollout_length:
            assert self._step <= self.rollout_length
        if at_end_of_rollout:
            self._state_reset(data['done'])
            to_augment = torch.logical_not(data['done']) | truncated
            data['done'][:] = True
        else:
            to_augment = truncated
        if torch.any(to_augment):
            data['reward'][to_augment] += self.gamma * next_vpred[to_augment]

        self._dones = data['done']
        self.storage.insert(data)

    def _get_truncated_envs(self, infos):
        truncated = []
        for info in infos:
            if 'TimeLimit.truncated' in info:
                truncated.append(info['TimeLimit.truncated'])
            else:
                truncated.append(False)
        return torch.tensor(truncated, dtype=torch.bool, device=self.device)

    def _get_next_value(self):
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state)
            else:
                outs = self.act(self._ob, state_in=None)
        return outs['value']

    def _state_reset(self, dones):
        if self.recurrent:
            def _state_item_reset(x):
                x[0, dones].zero_()
            nest.map_structure(_state_item_reset, self._state)

    def rollout(self):
        """Compute entire rollout and advantage targets."""
        self._reset()
        while not self.storage.rollout_complete:
            self.rollout_step()
        self.storage.compute_targets(self.gamma, self.lambda_,
                                     norm_advantages=self.norm_advantages)
        return self.storage.rollout_length()

    def sampler(self):
        """Create sampler to iterate over rollout data."""
        return self.storage.sampler(self.batch_size, self.recurrent,
                                    self.device)


if __name__ == '__main__':
    import unittest
    from dl.rl.modules import Policy, ActorCriticBase
    from dl.rl.envs import make_env
    from dl.modules import FeedForwardNet, Categorical, DiagGaussian
    from gym.spaces import Tuple
    from dl.rl.util.vec_env import VecEnvWrapper
    from torch.nn.utils.rnn import PackedSequence
    import numpy as np

    class FeedForwardBase(ActorCriticBase):
        """Test feed forward network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32], activate_last=True)
            if hasattr(self.action_space, 'n'):
                self.dist = Categorical(32, self.action_space.n)
            else:
                self.dist = DiagGaussian(32, self.action_space.shape[0])
            self.vf = torch.nn.Linear(32, 1)

        def forward(self, ob):
            """Forward."""
            if isinstance(ob, (list, tuple)):
                ob = ob[0]
            x = self.net(ob.float())
            return self.dist(x), self.vf(x)

    class RNNBase(ActorCriticBase):
        """Test recurrent network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32], activate_last=True)
            if hasattr(self.action_space, 'n'):
                self.dist = Categorical(32, self.action_space.n)
            else:
                self.dist = DiagGaussian(32, self.action_space.shape[0])
            self.lstm = torch.nn.LSTM(32, 32, 1)
            self.vf = torch.nn.Linear(32, 1)

        def forward(self, ob, state_in=None):
            """Forward."""
            if isinstance(ob, PackedSequence):
                x = self.net(ob.data.float())
                x = PackedSequence(x, batch_sizes=ob.batch_sizes,
                                   sorted_indices=ob.sorted_indices,
                                   unsorted_indices=ob.unsorted_indices)
            else:
                x = self.net(ob.float()).unsqueeze(0)
            if state_in is None:
                x, state_out = self.lstm(x)
            else:
                x, state_out = self.lstm(x, state_in['lstm'])
            if isinstance(x, PackedSequence):
                x = x.data
            else:
                x = x.squeeze(0)
            state_out = {'lstm': state_out, '1': torch.zeros_like(state_out[0])}
            return self.dist(x), self.vf(x), state_out

    class RolloutActor(object):
        """actor."""

        def __init__(self, pi):
            """init."""
            self.pi = pi

        def __call__(self, ob, state_in=None):
            """act."""
            outs = self.pi(ob, state_in)
            data = {'value': outs.value,
                    'action': outs.action}
            if outs.state_out:
                data['state'] = outs.state_out
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

    def test(env, base, batch_size, nested, rollout_length):
        pi = Policy(base(env.observation_space,
                         env.action_space))
        if nested:
            env = NestedVecObWrapper(env)
        nenv = env.num_envs
        data_manager = RolloutDataManager(env, RolloutActor(pi), 'cpu',
                                          batch_size=batch_size,
                                          rollout_length=rollout_length)
        for _ in range(3):
            data_manager.rollout()
            count = 0
            for batch in data_manager.sampler():
                assert 'key1' in batch
                count += 1
                assert 'done' in batch
                data_manager.act(batch['obs'])
            if data_manager.recurrent:
                assert count == np.ceil(nenv / data_manager.batch_size)
            else:
                n = data_manager.storage.get_rollout()['reward'].data.shape[0]
                assert count == np.ceil(n / data_manager.batch_size)

    def env_discrete(nenv):
        """Create discrete env."""
        return make_env('CartPole-v1', nenv=nenv)

    def env_continuous(nenv):
        """Create continuous env."""
        return make_env('LunarLanderContinuous-v2', nenv=nenv)

    class TestRolloutDataCollection(unittest.TestCase):
        """Test case."""

        def test_feed_forward(self):
            """Test feed forward network."""
            test(env_discrete(2), FeedForwardBase, 8, False, None)

        def test_recurrent(self):
            """Test recurrent network."""
            test(env_discrete(2), RNNBase, 2, False, None)

        def test_feed_forward_nested_ob(self):
            """Test feed forward network."""
            test(env_discrete(2), FeedForwardBase, 8, False, None)

        def test_recurrent_nested_ob(self):
            """Test recurrent network."""
            test(env_discrete(2), RNNBase, 2, False, None)

        def test_feed_forward_fixed_length(self):
            """Test feed forward network."""
            test(env_discrete(2), FeedForwardBase, 8, False, 64)

        def test_recurrent_fixed_length(self):
            """Test recurrent network."""
            test(env_discrete(2), RNNBase, 2, False, 64)

        def test_feed_forward_nested_ob_fixed_length(self):
            """Test feed forward network."""
            test(env_discrete(2), FeedForwardBase, 8, False, 64)

        def test_recurrent_nested_ob_fixed_length(self):
            """Test recurrent network."""
            test(env_discrete(2), RNNBase, 2, False, 64)

    unittest.main()
