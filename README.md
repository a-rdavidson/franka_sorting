
# Installation & Setup
1. Clone the repository
```bash 
git clone git@github.com:a-rdavidson/franka_sorting.git
```

2. Build the Docker container

We recommend using the provided Dockerfile to setup all dependencies and prequisites, though you can run the software outside of a container if all dependencies are installed and setup properly

```bash
docker build -t franka_sim .
```

3. Start the docker container using the provided script 
```bash 
./run_docker.sh
```

4. Build the software inside of the docker container 
```bash
./build.sh
```

5. To start the Gazebo Environment, spawn the robot arm, and intiialize the perception system:
```bash
source install/setup.bash 
ros2 launch franka_sorting franka_sim.launch.py
```

6. To launch the behavior tree which will start the sorting process, run: 
```bash
source install/setup.bash
ros2 run franka_sorting bt_pick_place.py
```

