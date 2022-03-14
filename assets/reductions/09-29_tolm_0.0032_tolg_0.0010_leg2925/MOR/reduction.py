"""Reduce a leg for all possible designs by toggling legs on and off
"""
import mor_util

import os
import shutil
import json


def hamiltonian_path_animation(objToAnimate, dt, factor, **params):
    duration = objToAnimate.duration
    id = objToAnimate.params['leg_id']
    A = 3.5
    t = factor * duration
    leg_order = [4, 1, 5, 0, 7, 3, 6, 2]

    def get_hamiltonian_cycle():
        """
        Sequence is a composition of two hamiltonian cycles on the hypercube.
        This cycle was chosen to evenly distribute inflation changes across legs.
        """
        p4 = [0, 1, 3, 7, 15, 11, 9, 8, 12, 13, 5, 4, 6, 14, 10, 2]
        path = []
        for i, p in enumerate(p4):
            node = p << 4
            subpath = p4[:i+1][::-1] + p4[i+1:][::-1]
            path += [x + node for x in subpath]
        return path
    path = [0, 0] + get_hamiltonian_cycle()

    period = float(duration) / len(path)

    def get_pressure(t, id):
        id = leg_order[id]
        ind = int(t / period)
        progress = (t - period * ind) / period
        if ind >= len(path):
            ind = len(path) - 1
            progress = 1
        is_active = path[ind] & (1 << id)
        # if is_active:
        #     return A
        # else:
        #     return 0.0
        is_active_prev = ind > 0 and path[ind-1] & (1 << id)
        if not is_active_prev and not is_active:
            return 0.0
        elif is_active_prev and is_active:
            return A
        elif is_active_prev and not is_active:
            # ramp down
            return max(0.0, A * (1 - 2 * progress))
        else:
            # ramp up
            return min(A, A * 2 * progress)

    p = get_pressure(t, id)
    if id == 0:
        print("PERIOD: {}, PRESSURE: {}, T: {}".format(int(t / period), p, t))
    objToAnimate.item.value = p


ANIMATION_FN = hamiltonian_path_animation


class DiskWithLegsReduction(mor_util.ModelOrderReduction):
    # Parameters for unpaired reduction
    ANGLES = [22.5 + i * 45. for i in range(8)]
    DURATION = 16.0
    SAVE_PERIOD = 2.0  # timesteps
    PARTS = ['leg{:04d}'.format(int(10*a)) for a in ANGLES]

    def create_scene(self, root, phase=None):
        from robots import get_robot_class
        from scenes import get_scene_class
        if phase is None:
            phase = [1] * len(self.ANGLES)
        if self.use_all_reductions:
            leg_reduction = self.reduction_prefix
            reduced_inds = [True] * len(self.ANGLES)
        elif self.load_reduced:
            leg_reduction = self.reduction_prefix
            reduced_inds = [self.part_to_reduce == part for part in self.PARTS]
        else:
            leg_reduction = None
            reduced_inds = [False] * len(self.ANGLES)

        try:
            part_ind = self.PARTS.index(self.part_to_reduce)
        except Exception:
            part_ind = None

        def keep(ind):
            return ind == part_ind or phase[ind]

        angles = [a for i, a in enumerate(self.ANGLES) if keep(i)]
        reduced_inds = [x for i, x in enumerate(reduced_inds) if keep(i)]

        sim_params = {
            'dt': 0.01,
            'friction_coef': 1.2,
            'leg_ym': 1160,
            'paper_ym': 2320,
            'gravity': [0., 0., -9800.],
            'debug': False,
            'with_gui': False
        }
        param_file = os.path.join(self.output_dir, 'sim_params.json')
        if not os.path.exists(param_file):
            with open(os.path.join(self.output_dir, 'sim_params.json'), 'w') as f:
                json.dump(sim_params, f)
        self.robot = get_robot_class('DiskWithLegs')(sim_params['dt'],
                                                     sim_params['friction_coef'],
                                                     angles,
                                                     [0] * len(angles),
                                                     sim_params['leg_ym'],
                                                     sim_params['paper_ym'],
                                                     leg_reduction=leg_reduction,
                                                     reduced_legs=reduced_inds)
        self.scene = get_scene_class('EmptyScene')(root, self.robot,
                                                   self.output_dir)

    def get_scene_data(self, root):
        pressure_contraints = [
            leg.cavity_node.pressure_constraint.getPathName().split('/', 1)[-1]
            for leg in self.robot.legs
        ]
        return {
            'reduction_node': self.robot.node.getPathName(),
            'pressure_constraints': pressure_contraints,
            'leg_nodes': [leg.node.getPathName() for leg in self.robot.legs],
        }

    def make_object_list(self):
        with open(os.path.join(self.output_dir, "sim_params.json"), 'r') as f:
            self.sim_params = json.load(f)
        dt = self.sim_params['dt']
        objs = []
        for i, constraint in enumerate(self.scene_data['pressure_constraints']):
            objs.append(mor_util.ObjToAnimate(
                constraint,
                'hamiltonian_path_animation',
                duration=self.DURATION,
                save_period=self.SAVE_PERIOD,
                dt=dt,
                leg_id=i
            ))
        return objs

    def get_reduction_node(self):
        if self.part_to_reduce in self.PARTS:
            ind = self.PARTS.index(self.part_to_reduce)
            return self.scene_data['leg_nodes'][ind]
        else:
            return self.scene_data['reduction_node']
