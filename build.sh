#!/bin/bash

set -e

echo "-- Building Workspace --"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "-- Sourcing Workspace --"
source install/setup.bash

echo "Build Complete!"
