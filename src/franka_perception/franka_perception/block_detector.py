import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_geometry_msgs import do_transform_point, do_transform_pose
from tf_transformations import quaternion_from_euler

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
            
            # Get the rotating bounding box
            rect = cv2.minAreaRect(largest)
            (u, v), (w, h), angle = rect
            #self.get_logger().info(
            #    f"{u}, {v}, {w}, {h}"
            #)
            
            # Get Depth at center
            z_camera = cv_depth[int(v), int(u)]
            
            if not np.isnan(z_camera):
                # Position Calculation (as before)
                ray = self.camera_model.projectPixelTo3dRay((u, v))
                
                target_pose = PoseStamped()
                target_pose.header.frame_id = "camera_link_optical"
                target_pose.header.stamp = rgb_msg.header.stamp
                
                target_pose.pose.position.x = ray[0] * z_camera
                target_pose.pose.position.y = ray[1] * z_camera
                target_pose.pose.position.z = z_camera

                # Since the camera is looking straight down, the block's yaw 
                # in the camera frame corresponds to the image rotation.
                yaw_camera = np.radians(angle)
                
                # Subtract 90 degrees caused by relative camera orientation
                yaw_world_aligned = yaw_camera - 1.5708

                # Blocks are usually grasped from above
                q = quaternion_from_euler(0, 3.14159, yaw_world_aligned)
                target_pose.pose.orientation.x = q[0]
                target_pose.pose.orientation.y = q[1]
                target_pose.pose.orientation.z = q[2]
                target_pose.pose.orientation.w = q[3]

                try:
                    # Transform the FULL POSE to World Frame
                    transform = self.tf_buffer.lookup_transform(
                        'world', 
                        'camera_link_optical', 
                        rclpy.time.Time()
                    )
                    pose_world = do_transform_pose(target_pose.pose, transform)
                    
                    self.get_logger().info(
                        f"Full Pose: X={pose_world.position.x:.2f} Y={pose_world.position.y:.2f} "
                        f"Yaw={yaw_world_aligned:.2f}rad"
                    )
                except Exception as e:
                    self.get_logger().error(f"Transform failed: {e}")
def main():
    rclpy.init()
    node = BlockDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
