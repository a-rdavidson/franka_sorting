import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    pkg_share = get_package_share_directory('franka_sorting')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_share, 'description', 'franka_camera.urdf.xacro')
    
    # Generate robot description from xacro
    robot_description = {'robot_description': Command(['xacro ', xacro_file])}
    
    panda_desc_share = get_package_share_directory('panda_description')

    resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH', 
        value=[os.path.join(panda_desc_share, '..')]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {os.path.join(pkg_share, "world", "scene.sdf")}', 
            'on_exit_shutdown': 'True'
         }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'franka_panda', '-x', '-0.3'],
        output='screen'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model', 
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ], 
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    load_joint_state_broadcaster = Node(
        package="controller_manager", 
        executable="spawner", 
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"], 
        parameters=[{'use_sim_time': True}]
    )


    load_gripper_controller = Node(
        package="controller_manager", 
        executable="spawner", 
        arguments=["panda_gripper_controller", "--controller-manager", "/controller_manager"], 
        parameters=[{'use_sim_time': True}]
    )


    load_arm_controller = Node(
        package="controller_manager", 
        executable="spawner", 
        arguments=["panda_arm_controller", "--controller-manager", "/controller_manager"], 
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        resource_path, 
        gazebo,
        robot_state_publisher,
        spawn_robot,
        rviz,
        bridge, 
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot,
                on_exit=[load_joint_state_broadcaster, load_arm_controller, load_gripper_controller]
            )
        )
    ])
