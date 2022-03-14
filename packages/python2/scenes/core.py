"""Definition of the python2 scene interface."""


class Scene(object):
    def __init__(self, root, robot, asset_dir):
        self.root = root
        self.robot = robot
        self.asset_dir = asset_dir
        self.init_scene()

    def init_scene(self):
        """Initialize the scene."""
        # call robot.load(root) at somepoint to initialize the robot
        raise NotImplementedError

    def act(self, action):
        """Execute action in sofa simulation."""
        raise NotImplementedError

    def observe(self, root):
        """Return observation of the scene."""
        raise NotImplementedError

    def reset(self):
        """Reset the scene."""
        self.root.reset()
        self.robot.reset()
