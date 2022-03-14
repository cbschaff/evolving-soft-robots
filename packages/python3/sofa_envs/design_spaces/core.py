"""Interface for design spaces."""


class DesignSpace(object):
    # Gym spaces to define the observation, action, and parameter space
    # for the environment.
    observation_space = None
    action_space = None
    parameter_space = None

    # The id of the python2 class that simulates robots in this design space.
    robot_id = None

    def sample(self):
        """Sample parameters from self.parameter_space"""
        return self.parameter_space.sample()

    def build(self, parameters, asset_dir, scene_id):
        '''Create meshes and all things necessary for simulation.

        Create meshes and other things and places them in asset_dir.
        '''
        raise NotImplementedError

    def observation(self, obs):
        '''Modify observation data from python2'''
        return obs

    def action(self, action):
        '''Modify action data before passing to python2'''
        return action
