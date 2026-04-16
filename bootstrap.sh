#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

cd /franka_ws

rosdep update
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "ament_python opencv moveit_ros_planning_interface joint_state_publisher_gui"

colcon build --symlink-install

echo "source /franka_ws/install/setup.bash" >> ~/.bashrc