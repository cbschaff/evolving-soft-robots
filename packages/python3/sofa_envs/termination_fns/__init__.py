from .generic import *


_termination_functions = {
    k: v for k, v in locals().items() if callable(v)
}


def get_termination_fn(id):
    if id not in _termination_functions:
        raise ValueError(f"Unknown termination function: {id}")
    else:
        return _termination_functions[id]
