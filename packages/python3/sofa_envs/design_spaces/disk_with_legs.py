from .core import DesignSpace
import gym
import numpy as np
import os
import json


NLEGS = 8
NREGULATORS = 3


def make_ob_space(n_legs, open_loop=False):
    disk_ob_space = gym.spaces.Box(low=-10000, high=10000, shape=(3,),
                                   dtype=np.float32)
    leg_ob_space = gym.spaces.Dict({
        'obs': gym.spaces.Box(low=-10000, high=10000, shape=(4,),
                              dtype=np.float32),
        'active': gym.spaces.Discrete(1)
    })
    ob_dict = {
        f'leg{i}': leg_ob_space for i in range(n_legs)
    }
    ob_dict['disk'] = disk_ob_space
    ob_dict['center'] = disk_ob_space
    if open_loop:
        ob_dict['actions'] = gym.spaces.Box(low=-10000000, high=10000000,
                                            shape=(4*NREGULATORS,),
                                            dtype=np.float32)
    return gym.spaces.Dict(ob_dict)


class DiskWithLegsDesignSpace(DesignSpace):
    action_space = gym.spaces.Box(low=0.0, high=90, shape=(NREGULATORS,),
                                  dtype=np.float32)
    observation_space = make_ob_space(NLEGS)
    parameter_space = gym.spaces.MultiDiscrete(nvec=NLEGS*[NREGULATORS + 1])
    obs_parameter_space = gym.spaces.MultiDiscrete(nvec=NLEGS*[NREGULATORS + 1])
    angles = [(360. / NLEGS) * (i + 0.5) for i in range(NLEGS)]

    robot_id = 'DiskWithLegs'

    def __init__(self, ym, paper_ym):
        self.ym = ym
        self.paper_ym = paper_ym

    def build(self, parameters, asset_dir, scene_id):
        os.makedirs(asset_dir, exist_ok=True)
        with open(os.path.join(asset_dir, "sim_params.json"), 'r') as f:
            sim_params = json.load(f)

        leg_angles = []
        leg_actuators = []
        for i, p in enumerate(parameters):
            if p > 0:
                leg_angles.append(self.angles[i])
                leg_actuators.append(int(p - 1))

        params = {
            'dt': sim_params['dt'],
            'friction_coef': sim_params['friction_coef'],
            'leg_angles': leg_angles,
            'leg_actuators': leg_actuators,
            'leg_ym': self.ym,
            'paper_ym': self.paper_ym,
            'leg_reduction': sim_params['reduction'],
        }
        with open(os.path.join(asset_dir, 'robot_params.json'), 'w') as f:
            json.dump(params, f)

    def get_obs_params(self, params):
        return params


class DiskWithLegsDiscreteDesignSpace(DiskWithLegsDesignSpace):

    def __init__(self, ym, paper_ym):
        self.ym = ym
        self.paper_ym = paper_ym
        self._designs = self.get_all_designs()
        self.parameter_space = gym.spaces.Discrete(n=len(self._designs))

    def expand_design_params(self, design):
        return self._designs[design]

    def build(self, parameters, asset_dir, scene_id):
        design = self._designs[parameters]
        DiskWithLegsDesignSpace.build(self, design, asset_dir, scene_id)

    def get_all_designs(self):
        designs = []
        n = NREGULATORS + 1
        for d in range(n ** NLEGS):
            design = []
            for i in range(NLEGS):
                design.append((d // (n ** i)) % n)
            if sum([d > 0 for d in design]) >= 3:
                designs.append(design)

        def reorder_actuators(design):
            first_act = None
            second_act = None
            normed_design = []
            for d in design:
                if d == 0:
                    normed_design.append(d)
                elif first_act is None or d == first_act:
                    first_act = d
                    normed_design.append(1)
                elif second_act is None or d == second_act:
                    second_act = d
                    normed_design.append(2)
                else:
                    normed_design.append(3)
            return tuple(normed_design)

        return list(set([reorder_actuators(d) for d in designs]))

    def get_obs_params(self, params):
        return np.array(self._designs[params])


class DiskWithLegsSixLegsDiscreteDesignSpace(DiskWithLegsDiscreteDesignSpace):
    def __init__(self, ym, paper_ym):
        DiskWithLegsDiscreteDesignSpace.__init__(self, ym, paper_ym)
        self._designs = self._filter_designs(self._designs)
        self.parameter_space = gym.spaces.Discrete(n=len(self._designs))

    def _filter_designs(self, designs):
        def _remove_design(design):
            nlegs = 0
            for d in design:
                if d != 0:
                    nlegs += 1
            return nlegs <= 6
        return list(filter(_remove_design, designs))


class OpenLoopDesignSpace(DesignSpace):

    def __init__(self):
        self.actions = []

    def action(self, ac):
        self.actions.append(ac)
        self.actions = self.actions[-4:]
        return ac

    def observation(self, obs):
        obs['actions'] = np.zeros((4*NREGULATORS), dtype=np.float32)
        for i, a in enumerate(reversed(self.actions)):
            obs['actions'][NREGULATORS*i:NREGULATORS*(i+1)] = a
        return obs


class DiskWithLegsOpenLoopDesignSpace(OpenLoopDesignSpace, DiskWithLegsDesignSpace):
    observation_space = make_ob_space(NLEGS, True)
    def __init__(self, *args, **kwargs):
        OpenLoopDesignSpace.__init__(self)
        DiskWithLegsDesignSpace.__init__(self, *args, **kwargs)

    def observation(self, obs):
        obs = DiskWithLegsDesignSpace.observation(self, obs)
        return OpenLoopDesignSpace.observation(self, obs)


class DiskWithLegsDiscreteOpenLoopDesignSpace(OpenLoopDesignSpace, DiskWithLegsDiscreteDesignSpace):
    observation_space = make_ob_space(NLEGS, True)
    def __init__(self, *args, **kwargs):
        OpenLoopDesignSpace.__init__(self)
        DiskWithLegsDiscreteDesignSpace.__init__(self, *args, **kwargs)

    def observation(self, obs):
        obs = DiskWithLegsDiscreteDesignSpace.observation(self, obs)
        return OpenLoopDesignSpace.observation(self, obs)


class DiskWithLegsSixLegsDiscreteOpenLoopDesignSpace(OpenLoopDesignSpace, DiskWithLegsSixLegsDiscreteDesignSpace):
    observation_space = make_ob_space(NLEGS, True)
    def __init__(self, *args, **kwargs):
        OpenLoopDesignSpace.__init__(self)
        DiskWithLegsSixLegsDiscreteDesignSpace.__init__(self, *args, **kwargs)

    def observation(self, obs):
        obs = DiskWithLegsSixLegsDiscreteDesignSpace.observation(self, obs)
        return OpenLoopDesignSpace.observation(self, obs)
