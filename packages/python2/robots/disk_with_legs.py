from __future__ import print_function
from .core import Robot
from .legs import Leg, ATTACHMENT_INDS
import numpy as np

TOP_CENTER_DISK = [46, 47, 48]


class DiskWithLegs(Robot):
    volume_mesh = '/code/src/evolving-soft-robots/assets/meshes/disk/body.msh'
    collision_mesh = '/code/src/evolving-soft-robots/assets/meshes/disk/collision.stl'
    poissonRatio = 0.3
    youngModulus = 5000
    mass = 0.03 * 2.5
    n_legs = 8
    all_angles = [(360. / n_legs) * (i + 0.5) for i in range(n_legs)]

    def __init__(self, dt, friction_coef, leg_angles, leg_actuators, leg_ym,
                 paper_ym, leg_reduction=None, reduced_legs=None):
        self.leg_angles = leg_angles
        self.leg_actuators = leg_actuators
        self.leg_ym = leg_ym
        self.paper_ym = paper_ym
        if leg_reduction:
            self.leg_reductions = []
            for i, angle in enumerate(self.leg_angles):
                if reduced_legs is None or reduced_legs[i]:
                    self.leg_reductions.append(leg_reduction + '_leg{:04d}'.format(int(10*angle)))
                else:
                    self.leg_reductions.append(None)
        else:
            self.leg_reductions = [None] * len(self.leg_angles)
        self.dt = dt
        self.friction_coef = friction_coef

    def load(self, root):
        self.node = root.createChild("diskWithLegsNode")

        self.create_disk(self.node)
        # create legs
        self.legs = [Leg(self.dt, self.friction_coef, self.leg_ym, self.paper_ym,
                         angle, reduction=r)
                     for angle, r in zip(self.leg_angles, self.leg_reductions)]
        for leg in self.legs:
            leg.load(self.node)
        self.create_constraints()

    def create_disk(self, root):
        self.disk_node = root.createChild("diskNode")
        self.disk_node.createObject('EulerImplicitSolver', name='odesolver',
                                    firstOrder="false",
                                    rayleighStiffness='0.1',
                                    rayleighMass='0.1')
        self.disk_node.createObject('SparseLDLSolver', name="preconditioner",
                                    template='CompressedRowSparseMatrixd')

        self.disk_node.createObject(
            'MeshGmshLoader',
            name='loader',
            filename=self.volume_mesh,
            scale3d=[1.0, 1.0, 1.0]
        )

        self.disk_node.createObject('TetrahedronSetTopologyContainer',
                                    src="@loader", name="container")
        self.disk_node.createObject('MechanicalObject', template='Vec3d',
                                    name='dofs', showIndices='false',
                                    showIndicesScale='0.001', rx='0')
        self.disk_node.createObject('UniformMass', totalMass=self.mass)

        self.disk_node.createObject('TetrahedronFEMForceField', template='Vec3d',
                                    method='large', name='FEM',
                                    poissonRatio=self.poissonRatio,
                                    youngModulus=self.youngModulus)

        self.disk_node.createObject('GenericConstraintCorrection',
                                    solverName="preconditioner")

        # Visual Model
        self.visual_node = self.disk_node.createChild('visualNode')
        self.visual_node.createObject(
            'MeshSTLLoader',
            name='loader',
            filename=self.collision_mesh,
        )
        self.visual_node.createObject('OglModel', src="@loader",
                                      depthTest=True,
                                      color='0.7 0.7 1 1.0')
        self.visual_node.createObject("BarycentricMapping")

        # Collision Model
        self.collision_node = self.disk_node.createChild('collisionNode')
        self.collision_node.createObject(
            'MeshSTLLoader',
            name='loader',
            filename=self.collision_mesh,
        )
        self.collision_node.createObject('TriangleSetTopologyContainer',
                                         src='@loader', name='container')
        self.collision_node.createObject('MechanicalObject',
                                         name='collisMO', template='Vec3d')
        # everything in the same collision group "0" will not contact eachother -- and self collision is not
        # enabled unless you turn it on -- (a sofa plugin might be needed -- not entirely sure) I think
        self.collision_node.createObject('TriangleCollisionModel', group="0",
                                         contactStiffness="10.0",
                                         contactFriction=self.friction_coef)
        self.collision_node.createObject('LineCollisionModel', group="0",
                                         contactStiffness="10.0",
                                         contactFriction=self.friction_coef)
        self.collision_node.createObject('PointCollisionModel', group="0",
                                         contactStiffness="10.0",
                                         contactFriction=self.friction_coef)
        self.collision_node.createObject('BarycentricMapping')

    def create_constraints(self):
        self.constraint_nodes = []
        for leg in self.legs:
            self.attach_leg(leg)

    def attach_leg(self, leg):
        node = self.disk_node.createChild('leg{}ConstraintNode'.format(leg.angle))
        self.constraint_nodes.append(node)
        x = np.array(leg.node.loader.getData('position').value)
        constraint_positions = x[ATTACHMENT_INDS].flatten()
        str_constraint_pos = " ".join([str(pos) for pos in constraint_positions])
        node.createObject("MechanicalObject", name="points", template="Vec3d",
                          position=str_constraint_pos, showIndices='0')
        node.createObject("BarycentricMapping")

        for i, ind in enumerate(ATTACHMENT_INDS):
            node.createObject(
                "BilateralInteractionConstraint",
                name="leg"+leg.angle+"_"+str(ind),
                template="Vec3d",
                object1='@' + leg.node.dofs.getPathName(),
                object2='@' + node.points.getPathName(),
                first_point=str(ind),
                second_point=i,
            )

    def reset(self):
        for leg in self.legs:
            leg.reset()

    def act(self, action):
        for leg, actuator in zip(self.legs, self.leg_actuators):
            leg.act([action[actuator]])

    def observe(self):
        ob = {}
        disk_ob = self.disk_node.dofs.getData('position').value
        obs = np.array([disk_ob[ind] for ind in TOP_CENTER_DISK])
        ob['disk'] = list(np.mean(obs, axis=0))
        leg_ob = {a: {'obs': [0., 0., 0., 0.], 'active': False}
                  for a in self.all_angles}

        for a, leg in zip(self.leg_angles, self.legs):
            leg_ob[a]['obs'] = leg.observe()
            leg_ob[a]['active'] = True
        for i, a in enumerate(self.all_angles):
            ob['leg{}'.format(i)] = leg_ob[a]

        # make all observations relative to disk
        ob['center'] = list(ob['disk'])
        origin_xy = ob['disk'][:2]
        ob['disk'][0] = 0
        ob['disk'][1] = 0
        for i, _ in enumerate(self.all_angles):
            if ob['leg{}'.format(i)]['active']:
                ob['leg{}'.format(i)]['obs'][0] -= origin_xy[0]
                ob['leg{}'.format(i)]['obs'][1] -= origin_xy[1]
        return ob


class DiskWithLegsOpenLoop(DiskWithLegs):
    def reset(self):
        self.prev_actions = None
        for leg in self.legs:
            leg.reset()

    def observe(self):
        ob = []
        disk_ob = self.disk_node.dofs.getData('position').value
        ob.extend(disk_ob[TOP_CENTER_DISK])
        actions = [leg.action for leg in self.legs]
        if self.prev_actions is None:
            self.prev_actions = actions
        ob.extend(actions)
        ob.extend(self.prev_actions)
        self.prev_actions = actions
        return ob
