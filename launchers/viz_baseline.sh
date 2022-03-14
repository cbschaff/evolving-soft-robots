#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# launching app

# args: config.yaml
# see python3/training/train.py for args
/docker-entrypoint.sh runuser -u sofauser -- python -m viz.test_baseline


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
