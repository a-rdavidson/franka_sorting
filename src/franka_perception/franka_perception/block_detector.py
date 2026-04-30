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

DIMENSIONS = [0.06, 0.06, 0.08]

class Track:
    def __init__(self, pose, dims, track_id, color):
        self.pose = pose
        self.dims = dims
        self.id = track_id
        self.color = color
        self.missed = 0

class ObjectTracker:
    def __init__(self):
        self.tracks = []
        self.delete_tracks = []
        self.next_id = 0
        self.MAX_DIST = 0.15   # 15 cm matching threshold
        self.MAX_MISSED = 25    # Frames before deleting track; slightly increased for stability

    def _update_for_existing_scene(self, detections, occlusion_check_func=None):
        new_tracks = []
        new_deletes = []
        used_tracks = set()

        # Match detections → existing tracks
        for det_pose, color in detections:
            best_track = None
            best_dist = float('inf')

            for track in self.tracks:
                dx = track.pose.position.x - det_pose.position.x
                dy = track.pose.position.y - det_pose.position.y
                dist = np.sqrt(dx * dx + dy * dy)

                if color == track.color and dist < best_dist and track.id not in used_tracks:
                    best_dist = dist
                    best_track = track

            if best_track and best_dist < self.MAX_DIST:
                best_track.pose = det_pose
                best_track.dims = DIMENSIONS
                best_track.missed = 0
                new_tracks.append(best_track)
                used_tracks.add(best_track.id)
            else:
                new_track = Track(det_pose, DIMENSIONS, self.next_id, color)
                self.next_id += 1
                new_tracks.append(new_track)

        # Handle unmatched tracks with Occlusion Awareness
        for track in self.tracks:
            if track.id not in used_tracks:
                # If the arm is likely blocking the view, freeze the 'missed' counter
                if occlusion_check_func and occlusion_check_func(track.pose):
                    new_tracks.append(track)
                    continue

                track.missed += 1
                if track.missed < self.MAX_MISSED:
                    new_tracks.append(track)
                else:
                    new_deletes.append(track)
                    
        self.delete_tracks = new_deletes
        self.tracks = new_tracks

        results = []
        for t in self.tracks:
            results.append((t.pose, t.dims, CollisionObject.ADD, t.id, t.color))
        for t in self.delete_tracks:
            results.append((t.pose, t.dims, CollisionObject.REMOVE, t.id, t.color))


        return results

    def _update_for_new_scene(self, detections):
        new_tracks = []
        for det_pose, color in detections:
            new_track = Track(det_pose, DIMENSIONS, self.next_id, color)
            self.next_id += 1
            new_tracks.append(new_track)
            
        self.tracks = new_tracks
        return [(t.pose, t.dims, CollisionObject.ADD, t.id, t.color) for t in self.tracks]
            
    def update(self, detections, occlusion_check_func=None):
        if self.tracks is None or len(self.tracks) == 0:
            return self._update_for_new_scene(detections)
        else:
            return self._update_for_existing_scene(detections, occlusion_check_func)

class BlockDetector(Node):
    def __init__(self):
        super().__init__('block_detector')

        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_ready = False
        self.tracker = ObjectTracker()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

    def info_callback(self, info_msg):
        self.camera_model.fromCameraInfo(info_msg)
        self.camera_ready = True

    def is_occluded(self, object_pose):
        """Checks if arm links (hand/link7) are overhead, blocking the camera's view of the object."""
        try:
            # Check proximity for the hand and the final wrist link[cite: 1, 2]
            for link in ['panda_hand', 'panda_link7']:
                if not self.tf_buffer.can_transform('world', link, rclpy.time.Time()):
                    continue
                
                trans = self.tf_buffer.lookup_transform('world', link, rclpy.time.Time()).transform
                dist_xy = np.sqrt((trans.translation.x - object_pose.position.x)**2 + 
                                  (trans.translation.y - object_pose.position.y)**2)
                
                # If arm is within 15cm of object's XY coordinates, assume occlusion
                if dist_xy < 0.15:
                    return True
            return False
        except Exception:
            return False

    def get_depth_from_contour(self, contour, depth_img):
        mask = np.zeros(depth_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        depth_vals = depth_img[mask == 255]
        depth_vals = depth_vals[~np.isnan(depth_vals)]
        depth_vals = depth_vals[depth_vals < 2.5]

        if len(depth_vals) == 0:
            return None
        return float(np.median(depth_vals))

    def find_objects(self, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)
        blue_mask = cv2.inRange(hsv, (100, 150, 50), (140, 255, 255))

        detections = []
        for color, mask in [("red", red_mask), ("blue", blue_mask)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if 200 < cv2.contourArea(c) < 5000:
                    (u, v), _, _ = cv2.minAreaRect(c)
                    z = self.get_depth_from_contour(c, depth)
                    if z is not None:
                        detections.append((u, v, z, color))
        return detections

    def pixel_to_world(self, u, v, z, stamp):
        ray = np.array(self.camera_model.projectPixelTo3dRay((u, v)))
        ray = ray / ray[2]

        pose_cam = PoseStamped()
        pose_cam.header.frame_id, pose_cam.header.stamp = "camera_link_optical", stamp
        pose_cam.pose.position.x, pose_cam.pose.position.y, pose_cam.pose.position.z = ray[0]*z, ray[1]*z, z
        pose_cam.pose.orientation.w = 1.0

        try:
            transform = self.tf_buffer.lookup_transform('world', 'camera_link_optical', rclpy.time.Time())
            pose_world = do_transform_pose(pose_cam.pose, transform)
            pose_world.position.z -= 0.03  # Adjust to block center
            return pose_world
        except Exception:
            return None

    def image_callback(self, rgb_msg, depth_msg):
        if not self.camera_ready: return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        detections = self.find_objects(rgb, depth)

        objects = []
        for u, v, z, color in detections:
            pose = self.pixel_to_world(u, v, z, rgb_msg.header.stamp)
            if pose: objects.append((pose, color))

        # Use the tracker with the occlusion check functin
        tracked = self.tracker.update(objects, self.is_occluded)
        self.publish_objects(tracked)

    def publish_objects(self, objects):
        marker_array = MarkerArray()
        for pose, dims, operation, track_id, color in objects:
            m = Marker()
            m.header.frame_id, m.header.stamp = 'world', self.get_clock().now().to_msg()
            m.ns, m.id, m.type = 'tracked_objects', track_id, Marker.CUBE
            m.action = Marker.ADD if operation == CollisionObject.ADD else Marker.DELETE
            if (m.action == Marker.DELETE):
                self.get_logger().info(f'We are DELETING an object! The ID is: {m.id}')
            m.pose = pose
            m.scale.x, m.scale.y, m.scale.z = DIMENSIONS
            m.color.a = 1.0
            if color == "red": m.color.r = 1.0
            else: m.color.b = 1.0
            marker_array.markers.append(m)

            co_msg = CollisionObject()
            co_msg.id, co_msg.operation = f"{color}_{track_id}", operation
            co_msg.header.frame_id, co_msg.header.stamp = 'world', self.get_clock().now().to_msg()
            co_msg.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=DIMENSIONS))
            co_msg.primitive_poses.append(pose)
            self.collision_pub.publish(co_msg)

        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = BlockDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
