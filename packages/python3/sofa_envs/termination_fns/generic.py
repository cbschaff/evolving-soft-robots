"""Define termination functions here.

The interface for a termination function is:
Input:
    observation
Output:
    Boolean
"""

import numpy as np
from dl import nest


def no_termination(ob):
    return False


def stop_on_nan(ob):
    return np.any([np.any(np.isnan(o)) for o in nest.flatten(ob)])
