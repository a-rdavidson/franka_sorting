cmake_minimum_required(VERSION 3.8)
project(franka_sorting)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED) # For HSV segmentation
find_package(moveit_ros_planning_interface REQUIRED) # For pick and place
find_package(franka_msgs REQUIRED) # For Franka hardware integration

# Install launch files, URDFs, and configurations
install(DIRECTORY
  launch
  description
  config
  world
  srdf
  rviz
  DESTINATION share/franka_sorting
)

install(PROGRAMS
  franka_sorting/robot_interface.py
  franka_sorting/bt_pick_place.py
  franka_sorting/action_node.py
  franka_sorting/gpd.py
  DESTINATION lib/franka_sorting
)

ament_package()

