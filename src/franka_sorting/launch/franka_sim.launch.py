import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('franka_sorting')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_share, 'description', 'franka.urdf.xacro')
    
    # Generate robot description from xacro
    robot_description = {'robot_description': Command(['xacro ', xacro_file])}
    panda_desc_share = get_package_share_directory('panda_description')

    camera_description_content = Command([
        'xacro ', 
        os.path.join(pkg_share, 'description', 'camera_only.urdf.xacro')
    ])

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
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('franka_sorting'), 'launch', 'move_group.launch.py')
        ]),
        launch_arguments=[
            ("ros2_control_plugin", "gz"),
            ("use_sim_time", "true"),
        ],
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
        arguments=[
            '-topic', 'robot_description',
            '-name', 'franka_panda',
            '-J', 'panda_joint1', '0.0',
            '-J', 'panda_joint2', '-0.785',
            '-J', 'panda_joint3', '0.0',
            '-J', 'panda_joint4', '-2.356',  # Deeply within [-2.92, -0.21]
            '-J', 'panda_joint5', '0.0',
            '-J', 'panda_joint6', '1.57',    # Within [0.13, 3.60]
            '-J', 'panda_joint7', '0.785'
        ],
        output='screen'
    )
    
    #robot_tf = Node(
    #    package='tf2_ros',
    #    executable='static_transform_publisher',
    #    name='world_to_robot_base',
    #    # arguments: x y z yaw pitch roll frame_id child_frame_id
    #    arguments=['-0.3', '0', '0', '0', '0', '0', 'world', 'panda_link0']
    #)

    #spawn_camera = Node(
    #    package='ros_gz_sim',
    #    executable='create',
    #    arguments = [
    #        '-string', camera_description_content, 
    #        '-name', 'external_camera'
    #    ],
    #    output='screen'
    #)

    #rviz = Node(
    #    package='rviz2',
    #    executable='rviz2',
    #    name='rviz2',
    #    output='screen',
    #    parameters=[{'use_sim_time': True}]
    #)
    block_detector_node = Node(
        package='franka_perception',
        executable='block_detector',
        name='block_detector',
        output='screen',
        parameters=[{'use_sim_time': True}]
    ) 
    block_listener_node = Node(
        package='franka_perception',
        executable='block_listener',
        name='block_listener',
        output='screen',
        parameters=[{'use_sim_time': True}]
    ) 
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            #'/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            #'/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            #'/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model', 
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ], 
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    #load_joint_state_broadcaster = Node(
    #    package="controller_manager", 
    #    executable="spawner", 
    #    arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"], 
    #    parameters=[{'use_sim_time': True}]
    #)


    load_gripper_controller = Node(
        package="controller_manager", 
        executable="spawner", 
        arguments=["panda_gripper_controller", "--controller-manager", "/controller_manager"], 
        parameters=[{'use_sim_time': True}]
    )


    #load_arm_controller = Node(
    #    package="controller_manager", 
    #    executable="spawner", 
    #    arguments=["panda_arm_controller", "--controller-manager", "/controller_manager"], 
    #    parameters=[{'use_sim_time': True}]
    #)
    #camera_tf = Node(
    #    package='tf2_ros',
    #    executable='static_transform_publisher',
    #    arguments = ['0.6', '0', '1.2', '0', '1.5708', '0', 'world', 'camera_link']
    #)
    #camera_optical_tf = Node(
    #    package='tf2_ros',
    #    executable='static_transform_publisher',
    #    name='camera_optical_broadcaster',
    #    arguments=['0', '0', '0', '-1.5708', '0', '-1.5708', 'camera_link', 'camera_link_optical'],
    #    parameters=[{'use_sim_time': True}]
    #)
    

    return LaunchDescription([
        resource_path, 
        gazebo,
        robot_state_publisher,
        spawn_robot,
        load_gripper_controller,
        # spawn_camera,
        bridge, 
        #camera_tf,
        #camera_optical_tf,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot,
                on_exit=[move_group, block_detector_node, block_listener_node]
            )
        )
    ])
