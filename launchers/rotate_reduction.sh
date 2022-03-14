#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# launching app

# . /docker-entrypoint.sh

# args: reduction_name prefix --part --mode
# see python2/reductions/animate.py for args
/docker-entrypoint.sh python2 -m mor_util.rotate_basis "${@:2}"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
