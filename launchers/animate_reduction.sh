#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# launching app

# . /docker-entrypoint.sh

# args: reduction_name prefix --part --mode
# see python2/reductions/animate.py for args
/docker-entrypoint.sh runuser -u sofauser -- python2 -m reductions.animate "${@:2}"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
