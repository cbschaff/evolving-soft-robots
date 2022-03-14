#!/usr/bin/env python2
from __future__ import print_function
import argparse
import os
import json
import time
import numpy as np
from splib import animation
from robots import get_robot_class
from scenes import get_scene_class


class SimulationInterface(object):
    def __init__(self, read, write):
        self.read = read
        self.write = write

    def receive(self):
        length = os.read(self.read, 5).decode('utf-8')
        while length == '':
            time.sleep(0.01)
            length = os.read(self.read, 5).decode('utf-8')
        cmd = os.read(self.read, int(length)).decode('utf-8')
        return json.loads(cmd)

    def send(self, msg):
        length = '%05d' % len(msg)
        os.write(self.write, length.encode('utf-8'))
        os.write(self.write, msg.encode('utf-8'))

    def animate(self, target, scene, factor):
        cmd = self.receive()

        if cmd['cmd'] == 'step':
            scene.act(cmd['action'])
            obs = scene.observe()
            if isinstance(obs, np.ndarray):
                obs = obs.tolist()
            self.send(json.dumps(obs))

        elif cmd['cmd'] == 'reset':
            scene.reset()
            obs = scene.observe()
            if isinstance(obs, np.ndarray):
                obs = obs.tolist()
            self.send(json.dumps(obs))

        elif cmd['cmd'] == 'close':
            exit()

        else:
            raise ValueError("Found bad command: %s" % cmd['cmd'])


def createScene(rootNode):
    parser = argparse.ArgumentParser('sofa_start')
    parser.add_argument("scene_id", type=str, help="id of scene to launch")
    parser.add_argument("robot_id", type=str, help="id of robot to simulate")
    parser.add_argument("asset_dir", type=str, help="asset directory")
    parser.add_argument("read", type=int, help="read pipe")
    parser.add_argument("write", type=int, help="write pipe")

    import sys
    print(sys.argv)
    args = parser.parse_args()
    with open(os.path.join(args.asset_dir, "robot_params.json"), 'r') as f:
        robot_params = json.load(f)

    robot = get_robot_class(args.robot_id)(**robot_params)

    scene = get_scene_class(args.scene_id)(rootNode, robot, args.asset_dir)
    interface = SimulationInterface(args.read, args.write)

    animation.AnimationManager(rootNode)
    animation.animate(interface.animate, {'target': rootNode, 'scene': scene},
                      duration=2.2, mode="loop")

    return rootNode
