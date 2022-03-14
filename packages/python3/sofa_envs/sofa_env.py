import numpy as np
import gym
import os
import json
import time
import subprocess
import tempfile
from .const import SOFA_PY2_ENTRYPOINT
from dl import nest


class SimulationInterface(object):
    def __init__(self, scene_id, robot_id, asset_dir, render=False):
        self.r_send, self.w_send = os.pipe()
        self.r_recv, self.w_recv = os.pipe()
        os.set_inheritable(self.r_send, True)
        os.set_inheritable(self.w_recv, True)

        # gui = 'qt' if render else 'batch'
        gui = 'qglviewer' if render else 'batch'

        command = (f"runSofa -g {gui} -n -1 --start "
                   f"--input-file {SOFA_PY2_ENTRYPOINT} --argv "
                   f"{scene_id} {robot_id} {asset_dir} "
                   f"{self.r_send} {self.w_recv}")
        print(command)
        self.encoder = subprocess.Popen(command, shell=True, close_fds=False)
        print('waiting for python2 to start...')
        time.sleep(1)
        self.closed = False

    def _numpyify(self, ob):
        # hack to turn lists into numpy arrays
        # only works because the lists in observations only contain numbers
        # temporarily add list to nest.ITEMS
        nest.add_item_class(list)
        np_ob = []
        for o in nest.flatten(ob):
            if isinstance(o, list):
                np_ob.append(np.asarray(o, dtype=np.float32))
            else:
                np_ob.append(o)

        ob = nest.pack_sequence_as(tuple(np_ob), ob)
        nest.ITEMS = nest.ITEMS[:-1] # Remove list from nest.ITEMS
        return ob

    def __del__(self):
        if not self.closed:
            self.close()

    def send(self, msg):
        length = f'{len(msg):05}'
        os.write(self.w_send, length.encode('utf-8'))
        os.write(self.w_send, msg.encode('utf-8'))

    def receive(self):
        length = os.read(self.r_recv, 5).decode('utf-8')
        while length == '':
            time.sleep(0.01)
            length = os.read(self.r_recv, 5).decode('utf-8')
        msg = os.read(self.r_recv, int(length)).decode('utf-8')
        return json.loads(msg)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        self.send(json.dumps({"cmd": 'step', "action": action}))
        return self._numpyify(self.receive())

    def reset(self):
        self.send(json.dumps({"cmd": 'reset'}))
        return self._numpyify(self.receive())

    def close(self):
        self.send(json.dumps({"cmd": 'close'}))
        os.close(self.r_send)
        os.close(self.w_send)
        os.close(self.r_recv)
        os.close(self.w_recv)
        self.closed = True


class SofaEnv(gym.Env):
    def __init__(self,
                 scene_id,
                 design_space,
                 reward_fn,
                 termination_fn,
                 dt,
                 gravity,
                 friction_coef,
                 max_steps,
                 steps_per_action,
                 debug,
                 with_gui,
                 reduction=None,
                 default_design=None,
                 ):
        self.scene_id = scene_id
        self.design_space = design_space
        self._reward_fn = reward_fn
        self._termination_fn = termination_fn
        self.dt = dt
        self.gravity = gravity
        self.friction_coef = friction_coef
        self.reduction = reduction
        self.max_steps = max_steps
        self.steps_per_action = steps_per_action
        self.debug = debug
        self.with_gui = with_gui
        self.action_space = self.design_space.action_space
        self.parameter_space = self.design_space.parameter_space
        self.observation_space = gym.spaces.Dict({
            'observation': self.design_space.observation_space,
            'design': self.design_space.obs_parameter_space
        })

        self.design_params = None
        self.asset_directory = None
        self.scene = None
        self.sim_params = {
            'dt': self.dt,
            'gravity': self.gravity,
            'max_steps': self.max_steps,
            'debug': self.debug,
            'with_gui': self.with_gui,
            'friction_coef': self.friction_coef,
            'reduction': self.reduction
        }
        self.asset_dir = None
        self.default_design = default_design
        if default_design is not None and not self.parameter_space.contains(default_design):
            raise ValueError(f"Bad design {default_design} for parameter space {self.parameter_space}")

    def init_scene(self, design_parameters):
        if self.scene is not None:
            self.scene.close()
            self.asset_dir.cleanup()
        os.makedirs('/tmp/designs', exist_ok=True)
        self.asset_dir = tempfile.TemporaryDirectory(prefix="/tmp/designs/")
        with open(os.path.join(self.asset_dir.name, 'sim_params.json'), 'w') as f:
            json.dump(self.sim_params, f)

        self.design_params = np.asarray(design_parameters)
        self._get_normed_design_params()
        self.design_space.build(design_parameters, self.asset_dir.name,
                                self.scene_id)
        self.scene = SimulationInterface(
            self.scene_id,
            self.design_space.robot_id,
            self.asset_dir.name,
            render=self.with_gui
        )

    def _get_normed_design_params(self):
        self.ob_design_params = self.design_space.get_obs_params(self.design_params)

    def reset(self):
        if self.scene is None:
            if self.default_design is None:
                self.init_scene(self.design_space.sample())
            else:
                self.init_scene(self.default_design)
        self.obs = self.design_space.observation(self.scene.reset())
        # step to get the robot on the ground
        for _ in range(15):
            self.scene.step(np.zeros(self.action_space.shape))
        self.current_step = 0
        self.sim_time = 0
        return {'observation': self.obs, 'design': self.ob_design_params}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = self.design_space.action(action)
        # get the observation and calculate the reward
        rwrd = 0
        for i in range(self.steps_per_action):
            prev_obs = self.obs
            self.obs = self.design_space.observation(self.scene.step(action))
            new_r = self._reward_fn(prev_obs, action, self.obs)
            if not np.isnan(new_r):
                rwrd += new_r
            self.current_step += 1

        self.sim_time = self.current_step * self.dt
        done = (self.max_steps <= self.current_step
                or self._termination_fn(self.obs))
        self.obs = np.nan_to_num(self.obs, 0)
        obs = {'observation': self.obs, 'design': self.ob_design_params}
        return obs, rwrd, done, {}

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self._np_random = np.random.RandomState(seed)

    def close(self):
        if self.scene is not None:
            self.scene.close()
            self.asset_dir.cleanup()
