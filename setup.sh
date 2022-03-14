#!/bin/bash

# NOTE: this setup script will be executed right before the launcher file inside the container,
#       use it to configure your environment.

alias rs="runuser -u sofauser"
if test -f "/root/wandb_info/.netrc"; then
    cp "/root/wandb_info/.netrc" "/root/.netrc"
    cp "/root/wandb_info/.netrc" "/home/sofauser/.netrc"
fi
