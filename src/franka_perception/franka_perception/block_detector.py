
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

from image_geometry import PinholeCameraModel

import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from visualization_msgs.msg import MarkerArray, Marker


class BlockDetector(Node):
    def __init__(self):
        super().__init__('block_detector')

        # Core utilities
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_ready = False

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera info subscriber
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        # Synchronized RGB + depth subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )

        self.ts.registerCallback(self.image_callback)

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)
        # store Marker class for local use
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

    def info_callback(self, info_msg):
        self.camera_model.fromCameraInfo(info_msg)
        self.camera_ready = True

    def find_objects_from_camera(self, cv_rgb, cv_depth):
        hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2HSV)

        # Red wraps HSV, so use two ranges
        mask1 = cv2.inRange(
            hsv,
            np.array([0, 120, 70]),
            np.array([10, 255, 255])
        )

        mask2 = cv2.inRange(
            hsv,
            np.array([170, 120, 70]),
            np.array([180, 255, 255])
        )

        red_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []

        for contour in contours:
            #ignore too small redgs
            if cv2.contourArea(contour) < 100:
                continue

            rect = cv2.minAreaRect(contour)
            (u, v), (w, h), angle = rect


            z_camera = cv_depth[int(v), int(u)]

            if np.isnan(z_camera):
                continue

            objects.append((u, v, w, h, angle, z_camera))

        return objects

    def calculate_transformation_to_object(self, u, v, z_camera, angle, stamp):
        ray = self.camera_model.projectPixelTo3dRay((u, v))

        target_pose = PoseStamped()
        target_pose.header.frame_id = "camera_link_optical"
        target_pose.header.stamp = stamp

        target_pose.pose.position.x = ray[0] * z_camera
        target_pose.pose.position.y = ray[1] * z_camera
        target_pose.pose.position.z = z_camera

        # Since the camera is looking straight down, the block's yaw 
                # in the camera frame corresponds to the image rotation.
        yaw_camera = np.radians(angle)

        # Camera looking downward adjustment
        yaw_world_aligned = yaw_camera - np.pi / 2

        q = quaternion_from_euler(0, np.pi, yaw_world_aligned)

        target_pose.pose.orientation.x = q[0]
        target_pose.pose.orientation.y = q[1]
        target_pose.pose.orientation.z = q[2]
        target_pose.pose.orientation.w = q[3]

        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                'camera_link_optical',
                rclpy.time.Time()
            )

            pose_world = do_transform_pose(target_pose, transform)

            self.get_logger().info(
                f"Block detected: "
                f"X={pose_world.pose.position.x:.3f}, "
                f"Y={pose_world.pose.position.y:.3f}, "
                f"Z={pose_world.pose.position.z:.3f}, "
                f"Yaw={yaw_world_aligned:.2f} rad"
            )

            #TODO: THIS IS DUMMY FIXED DIMENSIONS. EDIT THIS LATER
            dimensions = [0.04, 0.04, 0.03]
            return (pose_world.pose, dimensions)

        except Exception as e:
            self.get_logger().error(f"Transform failed: {e}")
            return None

    def image_callback(self, rgb_msg, depth_msg):
        if not self.camera_ready:
            return
        if self.camera_model.projectionMatrix() is None:
            return

        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        objects = self.find_objects_from_camera(cv_rgb, cv_depth)

        if not objects:
            return
        detected_objects = []
        for obj in objects:
            u, v, w, h, angle, z = obj

            result = self.calculate_transformation_to_object(
                u,
                v,
                z,
                angle,
                rgb_msg.header.stamp
            )
            if result:
                detected_objects.append(result)

        if detected_objects:
            self.publish_objects(detected_objects)



    def publish_objects(self, objects):
        """Add the supplied axis-aligned boxes to MoveIt’s planning scene and
        additionally publish a wireframe MarkerArray for RViz and individual
        CollisionObject messages for downstream consumers.

        Parameters
        ----------
        objects : list of dict
            Each dict must contain:
            - 'pose'       : geometry_msgs/Pose (box center, world frame)
            - 'dimensions' : [dx, dy, dz] edge lengths in metres

        Behavior
        --------
        * Publishes a PlanningScene diff on the existing `self.scene_pub`.
        * Publishes a MarkerArray on `/detected_markers` (LINE_LIST wireframe).
        The publisher is created lazily and stored as `self.marker_pub`.
        * Publishes each CollisionObject on `/collision_object` (one message per
        object) via a lazily-created `self.collision_pub`.
        """

        # 3) Build and publish MarkerArray (wireframe boxes)
        marker_array = self._MarkerArray()
        for idx, (pose, dims) in enumerate(objects):
            dx, dy, dz = dims


            # create marker
            m = self._Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'detected_objects'
            m.id = idx
            m.type = m.CUBE
            m.action = m.ADD
            m.pose = pose

            m.scale.x = dx
            m.scale.y = dy
            m.scale.z = dz
            # color (green)
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            # points are in absolute world coords, so pose must be identity
            marker_array.markers.append(m)


        # publish markers
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(marker_array.markers)} wireframe marker(s)')

        # 4) Publish individual CollisionObject messages for downstream nodes
        # (some systems prefer /collision_object single messages rather than full PlanningScene diffs)
        for idx, (pose, dims) in enumerate(objects):
            co_msg = CollisionObject()
            co_msg.id = f'detected_object_{idx}'
            co_msg.header.frame_id = 'world'
            co_msg.header.stamp = self.get_clock().now().to_msg()
            co_msg.operation = CollisionObject.ADD

            box = SolidPrimitive(type=SolidPrimitive.BOX,
                                dimensions=dims)
            co_msg.primitives.append(box)
            co_msg.primitive_poses.append(pose)

            self.collision_pub.publish(co_msg)

        self.get_logger().info(f'Published {len(objects)} CollisionObject message(s) on /collision_object')
def main():
    rclpy.init()
    node = BlockDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
