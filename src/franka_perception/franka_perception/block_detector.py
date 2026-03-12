import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_geometry_msgs import do_transform_point

class BlockDetector(Node):
    def __init__(self):
        super().__init__('block_detector')
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        
        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Synchronized Subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

    def info_callback(self, info_msg):
        self.camera_model.fromCameraInfo(info_msg)

    def image_callback(self, rgb_msg, depth_msg):
        if self.camera_model.projectionMatrix() is None:
            return

        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # Very simple segmentation, will have to extend for more colors 
        hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                z_camera = cv_depth[v, u]
                
                if not np.isnan(z_camera):
                    # Project to Camera Frame (camera_link_optical)
                    ray = self.camera_model.projectPixelTo3dRay((u, v))
                    point_camera = PointStamped()
                    point_camera.header = rgb_msg.header
                    point_camera.point.x = ray[0] * z_camera
                    point_camera.point.y = ray[1] * z_camera
                    point_camera.point.z = z_camera

                    # Transform to World Frame
                    try:
                        # Ensure we use the correct optical frame ID from the message
                        transform = self.tf_buffer.lookup_transform(
                            'world', 
                            'camera_link',
                            rclpy.time.Time()
                        )
                        point_world = do_transform_point(point_camera, transform)
                        
                        self.get_logger().info(
                            f"BLOCK DETECTED at World Coords: "
                            f"X={point_world.point.x:.3f}, "
                            f"Y={point_world.point.y:.3f}, "
                            f"Z={point_world.point.z:.3f}"
                        )
                    except Exception as e:
                        self.get_logger().warn(f"Could not transform to world: {e}")

def main():
    rclpy.init()
    node = BlockDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
