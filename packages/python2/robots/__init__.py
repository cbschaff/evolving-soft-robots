from .core import Robot
from .legs import Leg, CalibrationLeg
from .disk_with_legs import DiskWithLegs, DiskWithLegsOpenLoop
from .one_mesh_disk_with_legs import OneMeshDiskWithLegs


_robot_classes = {}
for k, v in locals().items():
    try:
        if issubclass(v, Robot):
            _robot_classes[k] = v
    except TypeError:
        pass


def get_robot_class(id):
    if id not in _robot_classes:
        raise ValueError("Unknown robot class: %s" % id)
    else:
        return _robot_classes[id]
