import os
import json
import subprocess
from mor.reduction import ReduceModel
import inspect
import shutil
import pickle
import time


class ReductionGraphEditor():
    """A context manager that automatically edits a scene graph to use MOR components.

    usage:
        with ReductionGraphEditor(mor_dir, rootNode) as node:
            create_scene(node)  # This function creates the scene graph for the node that was reduced.

    mor_dir: the output directory of model order reduction.
    root: The node in the graph that will be the parent of the reduction node.
    """

    def __init__(self, mor_dir, root):
        self.root = root
        with open(os.path.join(mor_dir, 'params.pkl'), 'r') as f:
            self.params = pickle.load(f)
        path = os.path.dirname(self.params.dataDir[:-1])
        self.add_abs_path(self.params.paramWrapper[1], path)
        self.params.paramWrapper[1]['nbrOfModes'] = self.params.nbrOfModes
        self.change_root_node(root.getPathName())

    def add_abs_path(self, params, path):
        for k, v in params.items():
            if isinstance(v, dict):
                self.add_abs_path(v, path)
            if 'Path' in k:
                if v[0] == '/':
                    params[k] = os.path.join(path, v[1:])
                else:
                    params[k] = os.path.join(path, v)

    def change_root_node(self, new_root):
        print(new_root.split('/'))
        print(self.params.paramWrapper[0])
        self.params.paramWrapper = (
            os.path.join(new_root, self.params.paramWrapper[0].split('/')[-1]),
            self.params.paramWrapper[1]
        )
        print(self.params.paramWrapper[0])

    def __enter__(self):
        from mor.wrapper import replaceAndSave
        from stlib.scene.wrapper import Wrapper
        replaceAndSave.tmp = 0  # reset object counter
        return Wrapper(self.root, replaceAndSave.MORreplace,
                       self.params.paramWrapper)

    def __exit__(self, type, value, traceback):
        from .scene_editing import modifyGraphScene
        modifyGraphScene(self.root, self.params.nbrOfModes,
                         self.params.paramWrapper)


class PhaseTracker():
    def __init__(self, tracking_dir):
        self.d = tracking_dir

    def _get_fname(self, phase):
        return os.path.join(self.d, 'phase{}_complete'.format(phase))

    def mark_phase_complete(self, phase):
        fname = self._get_fname(phase)
        if not os.path.exists(fname):
            open(fname, 'a').close()

    def can_execute_phase(self, phase):
        for i in range(1, phase):
            if not os.path.exists(self._get_fname(i)):
                return False
        for i in range(phase, 5):
            if os.path.exists(self._get_fname(i)):
                return False
        return True


class ModelOrderReduction():
    def __init__(self, output_dir, params=None):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.load_params(params)
        self.phase_tracker = PhaseTracker(self.output_dir)
        self.create_scene_files()

    def load_params(self, params):
        param_path = os.path.join(self.output_dir, 'class_params.pkl')
        if os.path.exists(param_path) and params is None:
            with open(param_path, 'r') as f:
                params = pickle.load(f)
        else:
            if params is None:
                params = {}
            with open(param_path, 'w') as f:
                pickle.dump(params, f)
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    def make_object_list(self):
        raise NotImplementedError

    def get_reduction_node(self):
        raise NotImplementedError

    def create_scene(self, root):
        raise NotImplementedError

    def get_scene_data(self, root):
        raise NotImplementedError

    def inspect_scene_graph(self, force=False):
        data_file = os.path.join(self.output_dir, "scene_data.json")
        if not os.path.exists(data_file) or force:
            subprocess.call("runSofa -g batch {}".format(self.inspection_scene),
                            shell=True)
        with open(os.path.join(self.output_dir, "scene_data.json"), 'r') as f:
            self.scene_data = json.load(f)

    def save_scene_data(self, root):
        data = self.get_scene_data(root)
        with open(os.path.join(self.output_dir, "scene_data.json"), 'w') as f:
            json.dump(data, f)

    def create_scene_files(self):
        class_name = self.__class__.__name__
        self.inspection_scene = os.path.join(self.output_dir,
                                             'scene_inspection.py')
        self.mor_scene = os.path.join(self.output_dir, 'scene_MOR.py')
        self.viz_scene = os.path.join(self.output_dir, 'scene_viz.py')
        if os.path.exists(self.inspection_scene):
            return
        reduction_file = inspect.getfile(self.__class__)
        if reduction_file[-1] == 'c':
            reduction_file = reduction_file[:-1]
        shutil.copyfile(reduction_file,
                        os.path.join(self.output_dir, 'reduction.py'))
        with open(self.inspection_scene, 'w') as f:
            f.write("from reduction import {}\n".format(class_name))
            f.write("import sys\n")
            f.write("def createScene(rootNode):\n")
            f.write("    reducer = {}('{}')\n".format(class_name, self.output_dir))
            f.write("    reducer.create_scene(rootNode, phase=None)\n")
            f.write("    reducer.save_scene_data(rootNode)\n")
            f.write("    sys.exit()\n")

        with open(self.mor_scene, 'w') as f:
            f.write("import sys\n")
            f.write("sys.path.append('{}')\n".format(self.output_dir))
            f.write("from reduction import {}\n".format(class_name))
            f.write("def createScene(rootNode, phase=None):\n")
            f.write("    reducer = {}('{}')\n".format(class_name, self.output_dir))
            f.write("    reducer.create_scene(rootNode, phase)\n")

    def write_viz_scene(self, timeExe):
        class_name = self.__class__.__name__
        with open(self.viz_scene, 'w') as f:
            f.write("from splib.animation import AnimationManager, animate\n")
            f.write("from mor.utility import sceneCreation as u\n")
            f.write("from mor.animation import shakingAnimations as sa\n")
            f.write("from reduction import {}\n".format(class_name))
            f.write("import sys\n")
            f.write("def createScene(rootNode):\n")
            f.write("    phase = eval(sys.argv[1])\n")
            f.write("    reducer = {}('{}')\n".format(class_name, self.output_dir))
            f.write("    reducer.create_scene(rootNode, phase)\n")
            f.write("    reducer.inspect_scene_graph(force=True)\n")
            f.write("    AnimationManager(rootNode)\n")
            f.write("    objs = reducer.make_object_list()\n")
            f.write("    for obj in objs:\n")
            f.write("        animFct = eval('sa.{}'.format(obj.animFct.split('.')[-1]))\n")
            f.write("        obj.animFct = animFct\n")
            f.write("    if phase is None:\n")
            f.write("        phase = [1 for _ in objs]\n")
            f.write("    else:\n")
            f.write("        assert len(phase) == len(objs)\n")
            f.write("    rootNode.createObject('VisualStyle', displayFlags='showForceFields')\n")
            f.write("    dt = rootNode.dt\n")
            f.write("    timeExe = {}\n".format(timeExe))
            f.write("    def create_end_flag():\n")
            f.write("        with open('/tmp/stop_recording_flag', 'w') as f:\n")
            f.write("            pass\n")
            f.write("    for i, obj in enumerate(objs):\n")
            f.write("        if not phase[i]:\n")
            f.write("            continue\n")
            f.write("        animate(obj.animFct, \n")
            f.write("                {'objToAnimate':obj, 'dt':dt}, obj.duration, \n")
            f.write("                onDone=lambda *x, **y: create_end_flag())\n")
            f.write("    u.addAnimation(rootNode, phase, timeExe, dt, objs)\n")

    def reduce(self, tolModes=0.001, tolGIE=0.05, ncpu=8, phases=[1, 2, 3, 4]):
        """Perform a model order reduction."""
        self.inspect_scene_graph()

        reduction_model = ReduceModel(
            self.mor_scene,
            nodeToReduce=self.get_reduction_node(),
            listObjToAnimate=self.make_object_list(),
            tolModes=tolModes,
            tolGIE=tolGIE,
            outputDir=self.output_dir,
            nbrCPU=ncpu,
            packageName='test',
            addToLib=False,
            verbose=True,
            addRigidBodyModes=[1, 1, 1]
        )

        phase_functions = [
            reduction_model.phase1,
            reduction_model.phase2,
            reduction_model.phase3,
            reduction_model.phase4
        ]
        for i in range(1, 5):
            if i not in phases:
                continue
            if not self.phase_tracker.can_execute_phase(i):
                raise ValueError(
                    "Can't execute phase {} because it has already been run or previous phases have not been run.".format(i)
                )
            phase_functions[i-1]()
            self.phase_tracker.mark_phase_complete(i)

        if 4 in phases:
            # save params for dynamic scene modification
            with open(os.path.join(self.output_dir, 'params.pkl'), 'wb') as f:
                pickle.dump(reduction_model.reductionParam, f)

    def animate(self, phase, timeExe):
        """Visualize the reduction scene under the desired animations."""
        self.write_viz_scene(timeExe)
        if phase is not None:
            phase = list(phase)
        view_file = 'scene_viz.py.qglviewer.view'
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), view_file),
            os.path.join(self.output_dir, view_file)
        )
        from sofa_recorder import SofaRecorder
        recorder = SofaRecorder()
        cmd = "runSofa {} --argv '{}'".format(self.viz_scene, phase)
        proc = subprocess.Popen(cmd, shell=True)
        time.sleep(1)
        recorder.start_recording()
        recorder.send(' ')
        while not os.path.exists('/tmp/stop_recording_flag'):
            time.sleep(0.01)
        recorder.send(' ')
        time.sleep(20)
        recorder.stop_recording()
        os.remove('/tmp/stop_recording_flag')
        proc.terminate()
        if not os.path.exists(os.path.join(self.output_dir, 'videos')):
            os.makedirs(os.path.join(self.output_dir, 'videos'))
        recorder.merge_recordings(os.path.join(self.output_dir,
                                               'videos/animate.mp4'))
        recorder.delete_recordings()

    def cleanup(self):
        shutil.rmtree(os.path.join(self.output_dir, 'debug'))
        shutil.rmtree(os.path.join(self.output_dir, 'mesh'))
