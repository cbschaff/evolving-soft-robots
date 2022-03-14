from .core import DesignSpace
from .disk_with_legs import *

_design_spaces = {}
for k, v in list(locals().items()):
    try:
        if issubclass(v, DesignSpace):
            _design_spaces[k] = v
    except TypeError:
        pass


def get_design_space(id):
    if id not in _design_spaces:
        raise ValueError(f"Unknown design_space: {id}")
    else:
        return _design_spaces[id]
