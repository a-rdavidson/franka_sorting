import rclpy
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject


class CollisionObjectListener(Node):
    def __init__(self):
        super().__init__('collision_object_listener')

        self.subscription = self.create_subscription(
            CollisionObject,
            '/collision_object',
            self.callback,
            10
        )

        self.get_logger().info('Listening to /collision_object ...')

    def callback(self, msg):
        #self.get_logger().info('Received CollisionObject')

        #self.get_logger().info(f'ID: {msg.id}')
        #self.get_logger().info(f'Frame: {msg.header.frame_id}')
        #self.get_logger().info(f'Operation: {msg.operation}')

        for i, primitive in enumerate(msg.primitives):
            dims = primitive.dimensions

            pose = msg.primitive_poses[i]

            #self.get_logger().info(
            #    f'Primitive {i}: '
            #    f'dims={dims}'
            #)

            #self.get_logger().info(
            #    f'Pose {i}: '
            #    f'pos=({pose.position.x:.3f}, '
            #    f'{pose.position.y:.3f}, '
            #    f'{pose.position.z:.3f}) '
            #    f'quat=({pose.orientation.x:.3f}, '
            #    f'{pose.orientation.y:.3f}, '
            #    f'{pose.orientation.z:.3f}, '
            #    f'{pose.orientation.w:.3f})'
            #)


def main():

    rclpy.init()

    node = CollisionObjectListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
