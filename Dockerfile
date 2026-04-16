# OSRF Ros2 jazzy desktop image
FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools & dependencies
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update --fix-missing && apt-get install -y \ 
	python3-colcon-common-extensions \ 
  python3-pip \ 
  python3-numpy \
	ros-jazzy-ros-gz \
	ros-jazzy-xacro \ 
	ros-jazzy-robot-state-publisher \
	ros-jazzy-moveit-ros-planning-interface \ 
	ros-jazzy-joint-state-publisher-gui \ 
	ros-jazzy-rviz2 \ 
	ros-jazzy-tf-transformations \ 
	ros-jazzy-controller-manager \ 
	ros-jazzy-ros2-control \ 
	ros-jazzy-ros2-controllers \
	ros-jazzy-gz-ros2-control \
	git \ 
	libopencv-dev \ 
  libpcl-dev \ 
  libeigen3-dev \
  python3-vcstool \ 
  ros-jazzy-py-trees-ros-viewer \
  ros-jazzy-py-trees \
  ros-jazzy-py-trees-ros \
  ros-jazzy-py-trees-ros-interfaces \
  ros-jazzy-cv-bridge \
  ros-jazzy-moveit-planners-ompl \ 
  tmux \
  vim \
	&& rm -rf /var/lib/apt/lists/*

# Create workspace 
WORKDIR /franka_ws
RUN mkdir -p src/deps

RUN git clone https://github.com/atenpas/gpd.git src/deps/gpd && \
    cmake -S src/deps/gpd -B src/deps/gpd/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build src/deps/gpd/build --target install && \
    ldconfig

COPY src/ /franka_ws/src
COPY build.sh clean_build.sh /franka_ws/

RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -y \
    --skip-keys "ament_python opencv moveit_ros_planning_interface joint_state_publisher_gui"

# Build workspace 
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \ 
		  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Automatically source workspace
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \ 
    echo "source /franka_ws/install/setup.bash" >> ~/.bashrc

# Default cmd
CMD ["bash"]
