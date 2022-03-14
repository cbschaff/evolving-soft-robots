"""Manage storage for reductions."""
import os
import shutil


_ROOT = '/code/src/evolving-soft-robots/assets/reductions'
_REGISTRY = None


def register_reduction(name):
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ReductionRegistry()
    return _REGISTRY.register(name)


def get_reduction_path(name):
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ReductionRegistry()
    return _REGISTRY.get_path(name)


def delete_reduction(name):
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ReductionRegistry()
    return _REGISTRY.delete(name)


def list_reductions():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ReductionRegistry()
    return _REGISTRY.list()


class ReductionRegistry():
    def __init__(self):
        self.registry_path = os.path.join(_ROOT, 'registry.txt')

    def _name_to_path(self, name):
        return os.path.join(_ROOT, name, 'MOR')

    def register(self, name):
        path = self._name_to_path(name)
        if os.path.exists(path):
            raise ValueError("{} already exists in registry.".format(name))
        return path

    def delete(self, name):
        path = self._name_to_path(name)
        if os.path.exists(path):
            shutil.rmtree(path)

    def get_path(self, name):
        path = self._name_to_path(name)
        if not os.path.exists(path):
            return None
        return path

    def list(self):
        names = os.listdir(_ROOT)
        return filter(lambda name: os.path.exists(self._name_to_path(name)),
                                                  names)
