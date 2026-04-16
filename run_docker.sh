#!/bin/bash

IMAGE_NAME="franka_sim"
CONTAINER_NAME="franka_simulation"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
	echo "Image not found, building now"
	docker build -t $IMAGE_NAME .
	if [ $? -ne 0 ]; then
		echo "Docker build failed"
		exit 1
	fi
fi

xhost +local:docker > /dev/null

echo "starting container"

docker run -it --rm \
	--name $CONTAINER_NAME \
	--network host \
	--privileged \
	--env="DISPLAY=$DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--device /dev/dri:/dev/dri \
	-v $(pwd)/src:/franka_ws/src \
	$IMAGE_NAME \
	bash -c "
		source /opt/ros/jazzy/setup.bash && \
		cd /franka_ws && \
		rosdep update && \
		rosdep install --from-paths src --ignore-src -y \
		  --skip-keys 'ament_python opencv moveit_ros_planning_interface joint_state_publisher_gui' && \
		colcon build --symlink-install && \
		echo "source /franka_ws/install/setup.bash" >> ~/.bashrc && \
		bash
	"

xhost -local:docker > /dev/null
echo "container exited"

