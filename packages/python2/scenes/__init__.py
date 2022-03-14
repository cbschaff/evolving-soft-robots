from .core import Scene
from .empty_scene import EmptyScene


_scene_classes = {}
for k, v in locals().items():
    try:
        if issubclass(v, Scene):
            _scene_classes[k] = v
    except TypeError:
        pass


def get_scene_class(id):
    if id not in _scene_classes:
        raise ValueError("Unknown scene class: %s" % id)
    else:
        return _scene_classes[id]
