#!/bin/bash
set -e 

echo "-- Cleaning Workspace --"
rm -rf build/ install/ log/

echo "-- Fresh Build --"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "-- Sourcing Workspace --"
source install/setup.bash

echo "Workspace is clean & rebuilt"
