#!/usr/bin/env python3

"""
    Node to force robot arm to home position before
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys

HOME_POSITION = {
    'panda_joint1': 0.0,
    'panda_joint2': -0.785,
    'panda_joint3': 0.0,
    'panda_joint4': -1.5708,
    'panda_joint5': 0.0,
    'panda_joint6': 1.8675,
    'panda_joint7': 0.785,
}

class HomePoseNode(Node):
    def __init__(self):
        super().__init__('home_pose')
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_home_goal(self):
        self.get_logger().info('Waiting for joint_trajectory_controller action server...')
        if not self._client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('Action server not available after 30s. Aborting.')
            sys.exit(1)

        joint_names = list(HOME_POSITION.keys())
        positions   = list(HOME_POSITION.values())

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(joint_names)
        point.time_from_start = Duration(sec=3, nanosec=0)  # 3s to reach home

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        goal.trajectory.points = [point]

        self.get_logger().info('Sending home pose goal...')
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Home pose goal rejected by controller.')
            sys.exit(1)

        self.get_logger().info('Goal accepted. Moving to home pose...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Home pose reached successfully.')
            sys.exit(0)
        else:
            self.get_logger().error(f'Failed to reach home pose. Error code: {result.error_code}')
            sys.exit(1)


def main():
    rclpy.init()
    node = HomePoseNode()
    node.send_home_goal()


if __name__ == '__main__':
    main()
