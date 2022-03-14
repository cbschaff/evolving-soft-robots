from reduction import DiskWithLegsReduction
import sys
def createScene(rootNode):
    reducer = DiskWithLegsReduction('/code/src/evolving-soft-robots/assets/reductions/09-29_tolm_0.0032_tolg_0.0010_leg0225/MOR')
    reducer.create_scene(rootNode, phase=None)
    reducer.save_scene_data(rootNode)
    sys.exit()
