"""Define reward functions here.

The interface for a reward function is:
Input:
    previous_observation, action, observation
Output:
    float
"""


def forward_distance(prev_ob, action, ob):
    return (ob['center'][0] - prev_ob['center'][0]) / 10.  # cm


def zero_reward(prev_ob, action, ob):
    return 0.0
