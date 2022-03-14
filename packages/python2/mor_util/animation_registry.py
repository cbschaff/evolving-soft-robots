"""A module for adding custom animation functions for Model Order Reduction.

The animation functions should have the following signature:

def my_animation_fn(objToAnimate, dt, factor, **params):
    # import needed packages
    # animate things

Because MOR requires all animation functions to be in a specific file,
registering your animation function will append the source code to that file.
Therefore, all import statements that your function depends upon should be
included within the function definition.
"""
import inspect
import shutil
import os
import random
import string

REGISTRY = None

ANIMATION_PATH = "/builds/python/mor/animation/shakingAnimations.py"


def register_animation(fn):
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = AnimationRegistry()

    REGISTRY.register(fn)


class AnimationRegistry():
    def __init__(self):
        self.registered_functions = [
            'upDateValue', 'rotationPoint', 'defaultShaking', 'shakingSofia',
            'shakingInverse'
        ]
        self.anim_path = ANIMATION_PATH
        self.anim_backup = '/tmp/' + ''.join([random.choice(string.ascii_letters)
                                              for _ in range(10)]) + '.py'
        shutil.copyfile(self.anim_path, self.anim_backup)

    def register(self, fn):
        if fn.__name__ in self.registered_functions:
            raise ValueError(
                "An animation function with name {} has already been registered".format(fn.__name__)
            )

        self.registered_functions.append(fn.__name__)

        with open(ANIMATION_PATH, 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write(inspect.getsource(fn))

    def __del__(self):
        shutil.copyfile(self.anim_backup, self.anim_path)
        os.remove(self.anim_backup)
