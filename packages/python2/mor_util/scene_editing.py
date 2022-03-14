"""Adapted from https://github.com/SofaDefrost/ModelOrderReduction/blob/master/python/mor/utility/sceneCreation.py"""
from Sofa import getCategories
try:
    from splib.scenegraph import get
except:
    raise ImportError("ModelOrderReduction plugin depend on SPLIB" \
                      + "Please install it : https://github.com/SofaDefrost/STLIB")


def removeObject(obj):
    obj.getContext().removeObject(obj)


def getNodeSolver(node):
    solver = []
    for obj in node.getObjects():
        categories = getCategories(obj.getClassName())
        solverCategories = ["ConstraintSolver", "LinearSolver", "OdeSolver"]
        if any(x in solverCategories for x in categories):
            solver.append(obj)
    return solver


def modifyGraphScene(node, nbrOfModes, newParam):
    '''
    **Modify the current scene to be able to reduce it**

    +------------+-----------+---------------------------------------------------------------------+
    | argument   | type      | definition                                                          |
    +============+===========+=====================================================================+
    | node       | Sofa.node | from which node will search & modify the graph                      |
    +------------+-----------+---------------------------------------------------------------------+
    | nbrOfModes | int       || Number of modes choosed in :py:meth:`.phase3` or :py:meth:`.phase4`|
    |            |           || where this function will be called                                 |
    +------------+-----------+---------------------------------------------------------------------+
    | newParam   | dic       || Contains numerous argument to modify/replace some component        |
    |            |           || of the SOFA scene. *more details see* :py:class:`.ReductionParam`  |
    +------------+-----------+---------------------------------------------------------------------+

    For more detailed about the modification & why they are made see here

    '''
    modesPositionStr = '0'
    for i in range(1, nbrOfModes):
        modesPositionStr = modesPositionStr + ' 0'
    argMecha = {'template': 'Vec1d', 'position': modesPositionStr}

    pathTmp, param = newParam
    node_path = node.getPathName()
    if node_path not in pathTmp:
        print("Problem with path : " + pathTmp)
        return
    pathTmp = pathTmp[len(node_path):]
    if pathTmp[0] == '/':
        pathTmp = pathTmp[1:]
    try:
        currentNode = get(node, pathTmp)
        solver = getNodeSolver(currentNode)
        if 'paramMappedMatrixMapping' in param:
            print('Create new child modelMOR and move node in it')
            myParent = currentNode.getParents()[0]
            modelMOR = myParent.createChild(currentNode.name+'_MOR')
            for parents in currentNode.getParents():
                parents.removeChild(currentNode)
            modelMOR.addChild(currentNode)
            for obj in solver:
                currentNode.removeObject(obj)
                currentNode.getParents()[0].addObject(obj)
            modelMOR.createObject('MechanicalObject', **argMecha)
            modelMOR.createObject('MechanicalMatrixMapperMOR',
                                  **param['paramMappedMatrixMapping'])

            if 'paramMORMapping' in param:
                # Find MechanicalObject name to be able to save to link it to
                # the ModelOrderReductionMapping
                param['paramMORMapping']['output'] = '@./'+currentNode.getMechanicalState().name

                currentNode.createObject('ModelOrderReductionMapping',
                                         **param['paramMORMapping'])
                print("Create ModelOrderReductionMapping in node")
            # else do error !!
    except Exception as e:
        print(e)
        print("Problem with path : "+pathTmp)
