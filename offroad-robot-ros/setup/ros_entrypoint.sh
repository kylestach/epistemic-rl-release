#!/bin/bash
set -e

# setup ros environment
source ~/venv/bin/activate
#source "/opt/ros/$ROS_DISTRO/setup.bash" --

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

cd ~/racecar-ws
source devel/setup.bash

exec "$@"
