"""Definition of the python2 robot interface."""


class Robot(object):
    def __init__(self, asset_dir):
        self.asset_dir = asset_dir

    def load(self, root):
        """Load robot into sofa simulation."""
        raise NotImplementedError

    def act(self, action):
        """Execute action in sofa simulation."""
        raise NotImplementedError

    def observe(self):
        """Return observation of the robot."""
        raise NotImplementedError

    def reset(self):
        """Reset the robot."""
        pass
