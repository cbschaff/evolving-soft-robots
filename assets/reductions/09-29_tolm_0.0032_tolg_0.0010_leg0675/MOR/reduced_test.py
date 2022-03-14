# -*- coding: utf-8 -*-
import os
import Sofa
from numpy import add,subtract,multiply
try:
    from splib.numerics import *
except:
    raise ImportError("ModelOrderReduction plugin depend on SPLIB"\
                     +"Please install it : https://github.com/SofaDefrost/STLIB")

path = os.path.dirname(os.path.abspath(__file__))

def TRSinOrigin(positions,modelPosition,translation,rotation,scale=[1.0,1.0,1.0]):
    posOrigin = subtract(positions , modelPosition)
    if any(isinstance(el, list) for el in positions):
        posOriginTRS = transformPositions(posOrigin,translation,eulerRotation=rotation,scale=scale)
    else:
        posOriginTRS = transformPosition(posOrigin,TRS_to_matrix(translation,eulerRotation=rotation,scale=scale))
    return add(posOriginTRS,modelPosition).tolist()
    
def newBox(positions,modelPosition,translation,rotation,offset,scale=[1.0,1.0,1.0]):
    pos = TRSinOrigin(positions,modelPosition,translation,rotation,scale)
    offset =transformPositions([offset],eulerRotation=rotation,scale=scale)[0]
    return add(pos,offset).tolist()

def Reduced_test(
                  attachedTo=None,
                  name="Reduced_test",
                  rotation=[0.0, 0.0, 0.0],
                  translation=[0.0, 0.0, 0.0],
                  scale=[1.0, 1.0, 1.0],
                  surfaceMeshFileName=False,
                  surfaceColor=[1.0, 1.0, 1.0],
                  nbrOfModes=36,
                  hyperReduction=True):
    """
    Object with an elastic deformation law.

        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | argument            | type      | definition                                                                                      |
        +=====================+===========+=================================================================================================+
        | attachedTo          | Sofa.Node | Where the node is created;                                                                      |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | name                | str       | name of the Sofa.Node it will                                                                   |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | rotation            | vec3f     | Apply a 3D rotation to the object in Euler angles.                                              |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | translation         | vec3f     | Apply a 3D translation to the object.                                                           |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | scale               | vec3f     | Apply a 3D scale to the object.                                                                 |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | surfaceMeshFileName | str       | Filepath to a surface mesh (STL, OBJ). If missing there is no visual properties to this object. |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | surfaceColor        | vec3f     | The default color used for the rendering of the object.                                         |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | nbrOfModes          | int       | Number of modes we want our reduced model to work with                                          |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+
        | hyperReduction      | Bool      | Controlled if we have the simple reduction or the hyper-reduction                               |
        +---------------------+-----------+-------------------------------------------------------------------------------------------------+

    """

    modelRoot = attachedTo.createChild(name)

    legNode_MOR = modelRoot.createChild('legNode_MOR')
    legNode_MOR.createObject('EulerImplicitSolver' , firstOrder = 'false', rayleighStiffness = '0.1', name = 'odesolver', rayleighMass = '0.1')
    legNode_MOR.createObject('SparseLDLSolver' , name = 'preconditioner', template = 'CompressedRowSparseMatrixd')
    legNode_MOR.createObject('GenericConstraintCorrection' , solverName = 'preconditioner')
    legNode_MOR.createObject('MechanicalObject' , position = [0]*nbrOfModes, template = 'Vec1d')
    legNode_MOR.createObject('MechanicalMatrixMapperMOR' , object1 = '@./MechanicalObject', object2 = '@./MechanicalObject', listActiveNodesPath = path + r'/data/listActiveNodes.txt', template = 'Vec1d,Vec1d', usePrecomputedMass = True, timeInvariantMapping2 = True, performECSW = hyperReduction, timeInvariantMapping1 = True, precomputedMassPath = path + r'/data/UniformMass_reduced.txt', nodeToParse = @./legNode)


    legNode = legNode_MOR.createChild('legNode')
    legNode.createObject('MeshGmshLoader' , scale3d = multiply(scale,[1.0, 1.0, 1.0]), translation = add(translation,[0.0, 0.0, 0.0]), rotation = add(rotation,[0.0, 0.0, 0.0]), name = 'loader', filename = path + r'/mesh/body.msh')
    legNode.createObject('TetrahedronSetTopologyContainer' , src = '@loader', name = 'container')
    legNode.createObject('TetrahedronSetGeometryAlgorithms')
    legNode.createObject('MechanicalObject' , showIndices = 'false', rx = '0', showIndicesScale = '0.001', name = 'dofs', template = 'Vec3d')
    legNode.createObject('UniformMass' , totalMass = '0.105')
    legNode.createObject('HyperReducedTetrahedronFEMForceField' , RIDPath = path + r'/data/reducedFF_legNode_0_RID.txt', name = 'reducedFF_legNode_0', weightsPath = path + r'/data/reducedFF_legNode_0_weight.txt', youngModulus = '1160', modesPath = path + r'/data/modes.txt', template = 'Vec3d', performECSW = hyperReduction, method = 'large', poissonRatio = '0.2', nbModes = nbrOfModes)
    legNode.createObject('PlaneROI' , drawEdges = 'true', drawBoxes = 'true', name = 'membraneROISubTopo', drawPoints = 'true', computeTetrahedra = 'false', plane = '47.32560145466731 2.2697150672804915 24.266320420710972 150.13086117652077 44.83897602400306 -5.54836310106726 137.84811191995115 74.46951069820223 -5.545972871279842 0.2', position = '@dofs.rest_position')
    legNode.createObject('ModelOrderReductionMapping' , input = '@../MechanicalObject', modesPath = path + r'/data/modes.txt', output = '@./dofs')


    paperNode = legNode.createChild('paperNode')
    paperNode.createObject('TriangleSetTopologyContainer' , position = '@../membraneROISubTopo.pointsInROI', name = 'container', triangles = '@../membraneROISubTopo.trianglesInROI')
    paperNode.createObject('HyperReducedTriangleFEMForceField' , RIDPath = path + r'/data/reducedFF_paperNode_1_RID.txt', name = 'reducedFF_paperNode_1', weightsPath = path + r'/data/reducedFF_paperNode_1_weight.txt', youngModulus = '2320', modesPath = path + r'/data/modes.txt', template = 'Vec3d', performECSW = hyperReduction, method = 'large', poissonRatio = '0.49', nbModes = nbrOfModes)


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('MeshSTLLoader' , scale3d = multiply(scale,[1.0, 1.0, 1.0]), translation = add(translation,[0.0, 0.0, 0.0]), rotation = add(rotation,[0.0, 0.0, 0.0]), name = 'loader', filename = path + r'/mesh/cavity.stl')
    cavityNode.createObject('Mesh' , src = '@loader', name = 'topo')
    cavityNode.createObject('MechanicalObject' , name = 'cavity')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')
    cavityNode.createObject('BarycentricMapping' , mapMasses = 'false', name = 'mapping', mapForces = 'false')


    collisionNode = legNode.createChild('collisionNode')
    collisionNode.createObject('MeshSTLLoader' , scale3d = multiply(scale,[1.0, 1.0, 1.0]), translation = add(translation,[0.0, 0.0, 0.0]), rotation = add(rotation,[0.0, 0.0, 0.0]), name = 'loader', filename = path + r'/mesh/collision.stl')
    collisionNode.createObject('TriangleSetTopologyContainer' , src = '@loader', name = 'container')
    collisionNode.createObject('MechanicalObject' , name = 'collisMO', template = 'Vec3d')
    collisionNode.createObject('TriangleCollisionModel' , contactStiffness = '10.0', group = '0', contactFriction = '1.2', selfCollision = False)
    collisionNode.createObject('LineCollisionModel' , contactStiffness = '10.0', group = '0', contactFriction = '1.2', selfCollision = False)
    collisionNode.createObject('PointCollisionModel' , contactStiffness = '10.0', group = '0', contactFriction = '1.2', selfCollision = False)
    collisionNode.createObject('BarycentricMapping')


    visualNode = legNode.createChild('visualNode')
    visualNode.createObject('MeshGmshLoader' , scale3d = multiply(scale,[1.0, 1.0, 1.0]), translation = add(translation,[0.0, 0.0, 0.0]), rotation = add(rotation,[0.0, 0.0, 0.0]), name = 'loader', filename = path + r'/mesh/body.msh')
    visualNode.createObject('OglModel' , writeZTransparent = True, src = '@loader', color = '0.7 0.7 0.7 0.8', depthTest = True)
    visualNode.createObject('BarycentricMapping')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')


    cavityNode = legNode.createChild('cavityNode')
    cavityNode.createObject('SurfacePressureConstraint' , drawScale = '0.002', name = 'pressure_constraint', valueType = 'pressure', value = '0.00', drawPressure = '0', template = 'Vec3d', triangles = '@topo.triangles')

    return legNode


#   STLIB IMPORT
from stlib.scene import MainHeader
def createScene(rootNode):
    surfaceMeshFileName = False

    MainHeader(rootNode,plugins=["SofaPython","SoftRobots","ModelOrderReduction"],
                        dt=0.01,
                        gravity=[0.0, 0.0, -9800.0])
    rootNode.VisualStyle.displayFlags="showForceFields"
    
    Reduced_test(rootNode,
                        name="Reduced_test",
                        surfaceMeshFileName=surfaceMeshFileName)

    # translate = 300
    # rotationBlue = 60.0
    # rotationWhite = 80
    # rotationRed = 70

    # for i in range(3):

    #     Reduced_test(rootNode,
    #                    name="Reduced_test_blue_"+str(i),
    #                    rotation=[rotationBlue*i, 0.0, 0.0],
    #                    translation=[i*translate, 0.0, 0.0],
    #                    surfaceColor=[0.0, 0.0, 1, 0.5],
    #                    surfaceMeshFileName=surfaceMeshFileName)
    # for i in range(3):

    #     Reduced_test(rootNode,
    #                    name="Reduced_test_white_"+str(i),
    #                    rotation=[0.0, rotationWhite*i, 0.0],
    #                    translation=[i*translate, translate, -translate],
    #                    surfaceColor=[0.5, 0.5, 0.5, 0.5],
    #                    surfaceMeshFileName=surfaceMeshFileName)

    # for i in range(3):

    #     Reduced_test(rootNode,
    #                    name="Reduced_test_red_"+str(i),
    #                    rotation=[0.0, 0.0, i*rotationRed],
    #                    translation=[i*translate, 2*translate, -2*translate],
    #                    surfaceColor=[1, 0.0, 0.0, 0.5],
    #                    surfaceMeshFileName=surfaceMeshFileName)
