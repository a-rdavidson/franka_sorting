# OSRF Ros2 jazzy desktop image
FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://snapshot.ubuntu.com/ubuntu|g' /etc/apt/sources.list
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
    ros-jazzy-py-trees-ros-viewer \
    ros-jazzy-py-trees \
    ros-jazzy-py-trees-ros \
    ros-jazzy-py-trees-ros-interfaces \
    ros-jazzy-cv-bridge \
    ros-jazzy-moveit-planners-ompl \
    ros-jazzy-moveit-simple-controller-manager \
    git \
    libopencv-dev \
    libpcl-dev \
    libeigen3-dev \
    python3-vcstool \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Workspace
WORKDIR /franka_ws
RUN mkdir -p src/deps

# Install GPD
RUN git clone https://github.com/CoMMALab/gpd.git src/deps/gpd && \
    cmake -S src/deps/gpd -B src/deps/gpd/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build src/deps/gpd/build --target install && \
    ldconfig

# Copy helper scripts only
COPY build.sh clean_build.sh bootstrap.sh /franka_ws/

# Auto-source ROS
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

CMD ["bash"]
