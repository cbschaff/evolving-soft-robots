#!/bin/bash

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# launching app

# args: config.yaml
# see python3/training/train.py for args
/docker-entrypoint.sh python -m training.train /configs/coopt.yaml


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
