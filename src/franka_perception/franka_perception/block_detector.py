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

DIMENSIONS = [0.04, 0.04, 0.06]
class Track:
    def __init__(self, pose, dims, track_id, color):
        self.pose = pose
        self.dims = dims
        self.id = track_id
        self.color = color
        self.missed = 0

# a really bad real time tracking. if this doesn't suffice, we would need to do proper SORT
class ObjectTracker:
    def __init__(self):
        self.tracks = []
        self.delete_tracks = []
        self.next_id = 0

        self.MAX_DIST = 0.15   # 15 cm matching threshold
        self.MAX_MISSED = 10    # frames before deleting track

    def _update_for_existing_scene(self, detections):
        #always overwrites current self.tracks with new_tracks.
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

            #found a track that matches well with current detection
            if best_track and best_dist < self.MAX_DIST:
                best_track.pose = det_pose
                best_track.dims = DIMENSIONS
                best_track.missed = 0
                new_tracks.append(best_track)
                used_tracks.add(best_track.id)
            #new object, create new track
            else:
                new_track = Track(det_pose, DIMENSIONS, self.next_id, color)
                self.next_id += 1
                new_tracks.append(new_track)

        # Handle unmatched tracks
        for track in self.tracks:
            if track.id not in used_tracks:
                track.missed += 1
                if track.missed < self.MAX_MISSED:
                    new_tracks.append(track)
                else:
                    new_deletes.append(track)
                    
                    
                    
        self.delete_tracks = new_deletes
        self.tracks = new_tracks

        results = []

        # Active tracks → ADD
        for t in self.tracks:
            results.append((t.pose, t.dims, CollisionObject.ADD, t.id, t.color))

        # Deleted tracks → REMOVE
        for t in self.delete_tracks:
            results.append((t.pose, t.dims, CollisionObject.REMOVE, t.id, t.color))

        return results
    # the reason why we have slightly different update function for new scene is so we don't conflate nearby objects as the same
    def _update_for_new_scene(self, detections):
        new_tracks = []
        for det_pose, color in detections:
            new_track = Track(det_pose, DIMENSIONS, self.next_id, color)
            self.next_id += 1
            new_tracks.append(new_track)
            
        self.tracks = new_tracks
        return [(t.pose, t.dims, CollisionObject.ADD, t.id, t.color) for t in self.tracks]
            
            
        
    def update(self, detections):
        if (self.tracks == None or len(self.tracks) == 0):
            return self._update_for_new_scene(detections)
        else:
            return self._update_for_existing_scene(detections)
        



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
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
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

        if objects:
            tracked = self.tracker.update(objects)
            self.publish_objects(tracked)

    def publish(self, objects):
        marker_array = MarkerArray()



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
            - operation     : remove or add
            - id            : assigned id of the object

        Behavior
        --------
        * Publishes a PlanningScene diff on the existing `self.scene_pub`.
        * Publishes a MarkerArray on `/detected_markers` (LINE_LIST wireframe).
        * Publishes each CollisionObject on `/collision_object` (one message per
        object) via a `self.collision_pub`.
        """

        # 3) Build and publish MarkerArray (wireframe boxes)
        marker_array = MarkerArray()
        for pose, dims, operation, track_id, color in objects:
            dx, dy, dz = dims


            # create marker
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'tracked_objects'
            m.id = track_id
            m.type = m.CUBE
            if (operation == CollisionObject.ADD):
                m.action = m.ADD
            elif (operation == CollisionObject.REMOVE):
                m.action = m.DELETE
            else:
                self.get_logger().error(f"Unknown CollisionObject enum sent to PublishObject, default to add")
                m.action = m.ADD
            m.pose = pose

            m.scale.x = DIMENSIONS[0]
            m.scale.y = DIMENSIONS[1]
            m.scale.z = DIMENSIONS[2]

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


        # publish markers
        self.marker_pub.publish(marker_array)
        #self.get_logger().info(f'Published {len(marker_array.markers)} wireframe marker(s)')

        # 4) Publish individual CollisionObject messages for downstream nodes
        # (some systems prefer /collision_object single messages rather than full PlanningScene diffs)
        for pose, dims, operation, track_id, color in objects:
            co_msg = CollisionObject()
            co_msg.id = f"{color}_{track_id}"
            co_msg.header.frame_id = 'world'
            co_msg.header.stamp = self.get_clock().now().to_msg()
            co_msg.operation = operation

            box = SolidPrimitive(type=SolidPrimitive.BOX,
                                 dimensions=DIMENSIONS)

            co_msg.primitives.append(box)
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
