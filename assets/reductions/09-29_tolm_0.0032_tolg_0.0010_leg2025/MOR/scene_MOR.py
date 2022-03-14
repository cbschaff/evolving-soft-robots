import sys
sys.path.append('/code/src/evolving-soft-robots/assets/reductions/09-29_tolm_0.0032_tolg_0.0010_leg0225/MOR')
from reduction import DiskWithLegsReduction
def createScene(rootNode, phase=None):
    reducer = DiskWithLegsReduction('/code/src/evolving-soft-robots/assets/reductions/09-29_tolm_0.0032_tolg_0.0010_leg0225/MOR')
    reducer.create_scene(rootNode, phase)
