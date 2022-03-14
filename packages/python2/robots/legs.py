from __future__ import print_function
from .core import Robot
import os
import numpy as np
from numpy.polynomial import Polynomial
import mor_util
from functools import partial

# Index of the top center of the pneunet. Used for control
TOP_CENTER = 2363
# Indices used for calibrating the bend of the pneunet
CALIBRATION_INDS = [160, 561, 563, 161]
# Indices used for attaching the pneunet to the disk.
# They are the 4 corners of one end of the pneunet
ATTACHMENT_INDS = [160, 163, 164, 247]

# For getting the region of interest for the paper layer
BOTTOM_CORNERS = [152, 154, 158]


SOFA_PY2_ENTRYPOINT = '/code/src/evolving-soft-robots/packages/python2/sofa_start.py'

# Best fit for warping real kPa to simulated kPa
REAL_KPA = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

SIM_KPA = [29.541015625, 65.673828125, 93.505859375, 128.173828125, 178.466796875,
           224.365234375, 277.587890625, 316.162109375, 343.505859375, 382.568359375]
ACTION_WARPING = Polynomial.fit(REAL_KPA, SIM_KPA, 1)


class Leg(Robot):
    mesh_dir = '/code/src/evolving-soft-robots/assets/meshes'
    poissonRatio = 0.2
    mass = 0.042 * 2.5
    paperPoissonRatio = 0.49
    max_kpa_per_second = 150  # kPa

    def __init__(self, dt, friction_coef, body_ym, paper_ym, angle=0.0, reduction=None):
        self.angle = '{:04d}'.format(int(10 * angle))
        self.bank_dir = self.mesh_dir + "/pneunet_" + self.angle
        self.body_ym = body_ym if body_ym else self.youngModulus
        self.paper_ym = paper_ym if paper_ym else self.paperYoungModulus
        if reduction:
            self.mor_dir = mor_util.get_reduction_path(reduction)
        else:
            self.mor_dir = None
        self.dt = dt
        self.friction_coef = friction_coef
        self.max_pressure_diff = self.dt * self.max_kpa_per_second

    def load(self, root):
        self.root = root.createChild('leg'+self.angle+'Node')
        if self.mor_dir is not None:
            with mor_util.ReductionGraphEditor(self.mor_dir, self.root) as node:
                self._load(node)
        else:
            self._load(self.root)

    def _load(self, root):
        # Create leg node containing the physics and volume mesh
        self.node = root.createChild('legNode')
        self.node.createObject('EulerImplicitSolver', name='odesolver',
                               firstOrder="false",
                               rayleighStiffness='0.1',
                               rayleighMass='0.1')
        self.node.createObject('SparseLDLSolver', name="preconditioner",
                               template='CompressedRowSparseMatrixd')
        # self.node.createObject('LULinearSolver', name="LUSolver")
        # self.node.createObject('ShewchukPCGLinearSolver', name="preconditioner",
        #                        iterations="1000", tolerance="1e-9",
        #                        preconditioners="LUSolver", build_precond="1",
        #                        update_step="1000")
        # self.node.createObject('ShewchukPCGLinearSolver', name="preconditioner",
        #                        iterations="1000")

        self.node.createObject(
            'MeshGmshLoader',
            name='loader',
            filename=os.path.join(self.bank_dir, "body.msh"),
            scale3d=[1.0, 1.0, 1.0]
        )

        self.node.createObject('TetrahedronSetTopologyContainer',
                               src="@loader", name="container")
        self.node.createObject('TetrahedronSetGeometryAlgorithms')
        self.node.createObject('MechanicalObject', template='Vec3d',
                               name='dofs', showIndices='false',
                               showIndicesScale='0.001', rx='0')
        self.node.createObject('UniformMass', totalMass=str(self.mass))
        # self.node.createObject('MeshMatrixMass', totalMass=str(self.mass))
        self.node.createObject('TetrahedronFEMForceField', template='Vec3d',
                               method='large', name='FEM',
                               poissonRatio=str(self.poissonRatio),
                               youngModulus=str(self.body_ym))

        pos = self.node.loader.position
        c1 = np.array(pos[BOTTOM_CORNERS[0]])
        c2 = np.array(pos[BOTTOM_CORNERS[1]])
        c3 = np.array(pos[BOTTOM_CORNERS[2]])
        # pad plane corners
        z_offset = np.array([0, 0, -0.1])
        d = (c1-c2) + (c1-c3)
        c1 += 0.1 * d/np.linalg.norm(d) + z_offset
        d = (c2-c1) + (c2-c3)
        c2 += 0.1 * d/np.linalg.norm(d) + z_offset
        d = (c3-c1) + (c3-c2)
        c3 += 0.1 * d/np.linalg.norm(d) + z_offset
        plane = ' '.join([str(p) for p in np.concatenate([c1, c2, c3])])+' 0.2'
        self.node.createObject('PlaneROI', name='membraneROISubTopo',
                               plane=plane, position='@dofs.rest_position',
                               computeTetrahedra="false", drawBoxes='true',
                               drawEdges='true', drawPoints='true')
        # self.node.createObject('BoxROI', name='membraneROISubTopo',
        #                        box='-300 -300 -4.1 300 300 -3.9',
        #                        computeTetrahedra="false", drawBoxes='true')
        self.node.createObject('GenericConstraintCorrection',
                               solverName="preconditioner")

        # Paper layer
        self.paper_node = self.node.createChild('paperNode')
        self.paper_node.createObject(
            'TriangleSetTopologyContainer',
            position='@../membraneROISubTopo.pointsInROI',
            triangles='@../membraneROISubTopo.trianglesInROI',
            name='container')
        self.paper_node.createObject(
            'TriangleFEMForceField', template='Vec3d',
            name='FEM', method='large',
            poissonRatio=str(self.paperPoissonRatio),
            youngModulus=str(self.paper_ym)
        )

        # Cavity and pressure constraint
        self.cavity_node = self.node.createChild('cavityNode')
        self.cavity_node.createObject(
            'MeshSTLLoader', name='loader',
            filename=os.path.join(self.bank_dir, 'cavity.stl')
        )
        self.cavity_node.createObject('Mesh', src='@loader', name='topo')
        self.cavity_node.createObject('MechanicalObject', name='cavity')
        # this pressure constraint is how the cavity is actually controlled
        self.cavity_node.createObject('SurfacePressureConstraint',
                                      name="pressure_constraint",
                                      template='Vec3d', value="0.00",
                                      triangles='@topo.triangles',
                                      drawPressure='0',
                                      drawScale='0.002', valueType="pressure")
        self.cavity_node.createObject('BarycentricMapping', name='mapping',
                                      mapForces='false', mapMasses='false')

        # Collision Model
        self.collision_node = self.node.createChild('collisionNode')
        self.collision_node.createObject(
            'MeshSTLLoader',
            name='loader',
            filename=os.path.join(self.bank_dir, "collision.stl")
        )
        self.collision_node.createObject('TriangleSetTopologyContainer',
                                         src='@loader', name='container')
        self.collision_node.createObject('MechanicalObject',
                                         name='collisMO', template='Vec3d')
        self.collision_node.createObject('TriangleCollisionModel', group="0",
                                         contactStiffness="10.0",
                                         contactFriction=str(self.friction_coef),
                                         selfCollision=False)
        self.collision_node.createObject('LineCollisionModel', group="0",
                                         contactStiffness="10.0",
                                         contactFriction=str(self.friction_coef),
                                         selfCollision=False)
        self.collision_node.createObject('PointCollisionModel', group="0",
                                         contactStiffness="10.0",
                                         contactFriction=str(self.friction_coef),
                                         selfCollision=False)
        self.collision_node.createObject('BarycentricMapping')

        # Visual Model
        self.visual_node = self.node.createChild('visualNode')
        # self.visual_node.createObject(
        #     'MeshSTLLoader',
        #     name='loader',
        #     filename=os.path.join(self.bank_dir, "collision.stl")
        # )
        self.visual_node.createObject(
            'MeshGmshLoader',
            name='loader',
            filename=os.path.join(self.bank_dir, "body.msh"),
            scale3d=[1.0, 1.0, 1.0]
        )
        self.visual_node.createObject('OglModel', src="@loader",
                                      depthTest=True,
                                      writeZTransparent=True,
                                      color='0.7 0.7 0.7 0.8')
        self.visual_node.createObject("BarycentricMapping")

    def reset(self):
        self.action = 0.0

    def _add_to_action(self, action):
        action_diff = np.clip(
            action - self.action, -self.max_pressure_diff, self.max_pressure_diff
        )
        self.action += action_diff

    def act(self, action):
        self._add_to_action(action[0])
        warping = ACTION_WARPING
        action = max(0.0, warping(self.action))
        self.cavity_node.pressure_constraint.value = action * self.dt

    def observe(self):
        # from util.plots import plot_forces
        # forces = np.array(self.node.dofs.getData('force').value)
        # plot_forces(forces)
        ob = self.node.dofs.getData('position').value
        return ob[TOP_CENTER] + [self.action]


class CalibrationLeg(Leg):
    def __init__(self, *args, **kwargs):
        Leg.__init__(self, *args, **kwargs)
        self.bank_dir = self.mesh_dir + "/pneunet_0000"

    # no action warping and use calibration points in observation
    def act(self, action):
        self._add_to_action(action[0])
        self.cavity_node.pressure_constraint.value = self.action * self.dt

    def observe(self):
        ob = self.node.dofs.getData('position').value
        ob = [ob[ind] for ind in CALIBRATION_INDS]
        return np.asarray(ob).flatten().tolist() + [self.action]
