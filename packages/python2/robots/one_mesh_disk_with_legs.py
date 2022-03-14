from __future__ import print_function
from .core import Robot
import numpy as np
import mor_util


class OneMeshDiskWithLegs(Robot):
    volume_mesh = '/code/src/evolving-soft-robots/assets/meshes/whole_robot_meshes/volume.vtk'
    collision_mesh = '/code/src/evolving-soft-robots/assets/meshes/whole_robot_meshes/volume_collision.stl'
    cavity_mesh = '/code/src/evolving-soft-robots/assets/meshes/whole_robot_meshes/cavity_0_0_{}.stl'
    poissonRatio = 0.05
    youngModulus = 25000 # 96626
    paperPoissonRatio = 0.49
    paperYoungModulus = 500000 # 5380343
    mass = 0.035 + 4 * 0.042
    friction_coef = 0.7
    max_kpa_per_second = 100  # kPa

    def __init__(self, dt, asset_dir, leg_actuators, reduction=None):
        self.leg_actuators = leg_actuators
        if reduction:
            self.mor_dir = mor_util.get_reduction_path(reduction)
        else:
            self.mor_dir = None
        self.max_pressure_diff = dt * self.max_kpa_per_second

    def load(self, root):
        self.node = root.createChild("diskWithLegsNode")

        if self.mor_dir is not None:
            with mor_util.ReductionGraphEditor(self.mor_dir, self.node) as node:
                self.create_body(node)
        else:
            self.create_body(self.node)

    def create_body(self, root):
        root.createObject('EulerImplicitSolver', name='odesolver',
                          firstOrder="false",
                          rayleighStiffness='0.1',
                          rayleighMass='0.1')
        root.createObject('SparseLDLSolver', name="preconditioner",
                          template='CompressedRowSparseMatrix3d')

        root.createObject(
            'MeshVTKLoader',
            name='loader',
            filename=self.volume_mesh,
            scale3d=[1.0, 1.0, 1.0]
        )

        root.createObject('TetrahedronSetTopologyContainer',
                          src="@loader", name="container")
        root.createObject('MechanicalObject', template='Vec3d',
                          name='dofs', showIndices='false',
                          showIndicesScale='0.001', rx='0')
        root.createObject('UniformMass', totalMass=self.mass)
        # root.createObject('MeshMatrixMass', totalMass=self.mass)

        root.createObject('TetrahedronFEMForceField', template='Vec3d',
                          method='large', name='FEM', drawAsEdges="1",
                          poissonRatio=self.poissonRatio,
                          youngModulus=self.youngModulus)

        root.createObject('GenericConstraintCorrection',
                          solverName="preconditioner")

        # Visual Model
        self.visual_node = root.createChild('visualNode')
        self.visual_node.createObject(
            'MeshSTLLoader',
            name='loader',
            filename=self.collision_mesh,
        )
        self.visual_node.createObject('OglModel', src="@loader",
                                      color='0.7 0.7 1 0.6')
        self.visual_node.createObject("BarycentricMapping")

        # Collision Model
        self.collision_node = root.createChild('collisionNode')
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

        # Create cavities
        self.cavity_nodes = []
        for i in range(4):
            cavity = root.createChild('cavityNode{}'.format(i))
            self.cavity_nodes.append(cavity)
            cavity.createObject(
                'MeshSTLLoader', name='loader',
                filename=self.cavity_mesh.format(i)
            )
            cavity.createObject('Mesh', src='@loader', name='topo')
            cavity.createObject('MechanicalObject', name='cavity')
            # this pressure constraint is how the cavity is actually controlled
            cavity.createObject('SurfacePressureConstraint',
                                name="pressure_constraint",
                                template='Vec3d', value="0.00",
                                triangles='@topo.triangles',
                                drawPressure='0',
                                drawScale='0.002', valueType="pressure")
            cavity.createObject('BarycentricMapping', name='mapping',
                                mapForces='false', mapMasses='false')

        # Paper layer
        root.createObject('BoxROI', name='membraneROISubTopo',
                          box='-300 300 -7.2 300 -300 -7.1',
                          computeTetrahedra="false", drawBoxes='true')
        self.paper_node = root.createChild('paperNode')
        self.paper_node.createObject(
            'TriangleSetTopologyContainer',
            position='@../membraneROISubTopo.pointsInROI',
            triangles='@../membraneROISubTopo.trianglesInROI',
            name='container')
        self.paper_node.createObject(
            'TriangleFEMForceField', template='Vec3d',
            name='FEM', method='large',
            poissonRatio=str(self.paperPoissonRatio),
            youngModulus=str(self.paperYoungModulus)
        )

    def reset(self):
        self.action = np.zeros(2)

    def _add_to_action(self, action):
        action = np.array(action)
        action_diff = np.clip(
            action - self.action, -self.max_pressure_diff, self.max_pressure_diff
        )
        self.action += action_diff

    def act(self, action):
        self._add_to_action(action)
        for i, cavity in enumerate(self.cavity_nodes):
            a = self.action[self.leg_actuators[i]]
            cavity.pressure_constraint.value = a

    def observe(self):
        # hack for now
        ob = self.node.dofs.getData('position').value
        ob = np.array(ob)
        return self.action
