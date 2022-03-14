from .disk_with_legs import *


_reward_functions = {
    k: v for k, v in locals().items() if callable(v)
}


def get_reward_fn(id):
    if id not in _reward_functions:
        raise ValueError(f"Unknown reward function: {id}")
    else:
        return _reward_functions[id]
