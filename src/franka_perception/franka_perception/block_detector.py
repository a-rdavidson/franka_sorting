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
from geometry_msgs.msg import Pose, Quaternion
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from visualization_msgs.msg import MarkerArray, Marker


class BlockDetector(Node):
    def __init__(self):
        super().__init__('block_detector')

        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_ready = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

    def info_callback(self, info_msg):
        self.camera_model.fromCameraInfo(info_msg)
        self.camera_ready = True

    def get_depth_from_contour(self, contour, depth_img):
        mask = np.zeros(depth_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        depth_vals = depth_img[mask == 255]
        depth_vals = depth_vals[~np.isnan(depth_vals)]

        if len(depth_vals) == 0:
            return None

        return float(np.median(depth_vals))

    def find_objects(self, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # --- RED MASK ---
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        # --- BLUE MASK ---
        blue_mask = cv2.inRange(hsv, (100, 150, 50), (140, 255, 255))

        masks = [("red", red_mask), ("blue", blue_mask)]
        detections = []

        for color, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)

                # Filter out noise AND large containers
                if area < 200 or area > 5000:
                    continue

                rect = cv2.minAreaRect(c)
                (u, v), (w, h), _ = rect

                z = self.get_depth_from_contour(c, depth)
                if z is None:
                    continue

                detections.append((u, v, z, color))

        return detections

    def pixel_to_world(self, u, v, z, stamp):
        ray = np.array(self.camera_model.projectPixelTo3dRay((u, v)))

        # Normalize so Z = 1
        # ray = ray / ray[2]

        pose_cam = PoseStamped()
        pose_cam.header.frame_id = "camera_link_optical"
        pose_cam.header.stamp = stamp

        pose_cam.pose.position.x = ray[0] * z
        pose_cam.pose.position.y = ray[1] * z
        pose_cam.pose.position.z = z

        # No orientation (avoid wrong assumptions)
        pose_cam.pose.orientation.w = 1.0

        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                'camera_link_optical',
                rclpy.time.Time()  # latest transform (more stable)
            )

            pose_world = do_transform_pose(pose_cam.pose, transform)

            # Adjust to block center (assuming depth hits top surface)
            block_height = 0.06
            pose_world.position.z -= block_height / 2.0

            return pose_world

        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
            return None

    def image_callback(self, rgb_msg, depth_msg):
        if not self.camera_ready:
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        detections = self.find_objects(rgb, depth)

        objects = []
        for u, v, z, color in detections:
            pose = self.pixel_to_world(u, v, z, rgb_msg.header.stamp)
            if pose:
                objects.append((pose, color))

        self.publish(objects)

    def publish(self, objects):
        marker_array = MarkerArray()

        for i, (pose, color) in enumerate(objects):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "objects"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = pose

            m.scale.x = 0.04
            m.scale.y = 0.04
            m.scale.z = 0.06

            if color == "red":
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            else:
                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 1.0

            m.color.a = 1.0

            marker_array.markers.append(m)

            # Collision object
            co = CollisionObject()
            co.id = f"{color}_{i}"
            co.header.frame_id = "world"
            co.operation = CollisionObject.ADD

            box = SolidPrimitive(type=SolidPrimitive.BOX,
                                 dimensions=[0.04, 0.04, 0.06])

            co.primitives.append(box)
            co.primitive_poses.append(pose)

            self.collision_pub.publish(co)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = BlockDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
