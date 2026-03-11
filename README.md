# Prerequisites
- Ensure you have ROS2 & Gazebo installed
```bash
sudo apt update 
sudo apt install ros-jazzy-rviz2 ros-jazzy-ros-gz ros-jazzy-xacro ros-jazzy-robot-state-publisher python3-colcon-common-extensions
```

# Installation & Setup
1. Clone the repository
```bash 
git clone git@github.com:a-rdavidson/franka_sorting.git
```

2. Use the provided ```clean_build.sh``` script to build the workspace for the first time
```bash 
./clean_build.sh
```

3. To start the Gazebo Environment, spawn the robot arm, and open rviz:
```bash
source install/setup.bash 
ros2 launch franka_sorting franka_sim.launch.py
```

4. Alternatively launch the Dockerfile with the dependencies and environment preconfigured
```bash 
./run_docker.sh
```
