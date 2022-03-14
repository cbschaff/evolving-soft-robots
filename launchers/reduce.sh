#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# launching app

# args: reduction_name prefix --tolModes --tolGIE --ncpu --nparallel --part
# see python2/reductions/launch.py for args
/docker-entrypoint.sh python2 -m reductions.launch "${@:2}"


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
