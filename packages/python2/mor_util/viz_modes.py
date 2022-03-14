from robots import get_robot_class
from scenes import get_scene_class
from splib import animation
from mor_util.rotate_basis import ModeUtil
import os
import json
import numpy as np


mode_util = ModeUtil('../../assets/reductions/09-29_tolm_0.0032_tolg_0.0010_leg0225/MOR/data/modes.txt')
mode_util.load()
modes = mode_util.modes[3:]


def animate_fn(root, robot, factor):
    step = int(factor / 0.01)
    n = len(modes)
    dofs = robot.node.dofs
    pos = np.array(dofs.position)
    pos -= np.mean(pos, axis=0)
    norm = np.mean(np.linalg.norm(pos, axis=1))
    mode = modes[step]
    mode_norm = np.linalg.norm(mode, axis=1).mean()
    mode = (norm * mode / mode_norm).tolist()
    dofs.position = mode


def createScene(rootNode):
    asset_dir = 'tmp_assets'
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
    with open(asset_dir + '/sim_params.json', 'w') as f:
        data = {
            'dt': 0.01,
            'gravity': [0, 0, 0],
            'with_gui': True,
            'debug': False,
            'friction_coef': 1.2
        }
        json.dump(data, f)
    robot = get_robot_class('Leg')(0.01, 1.2, 1170, 2340)
    scene = get_scene_class('EmptyScene')(rootNode, robot, asset_dir)
    animation.AnimationManager(rootNode)
    animation.animate(animate_fn, {'root': rootNode, 'robot':robot}, duration=1.0)
