#!/usr/bin/env python3
"""
RobotInterface — ROS 2 node wrapping MoveIt 2, the gripper controller,
and the Gazebo pose bridge.
"""

import copy
from dataclasses import dataclass

import numpy as np
import tf_transformations

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Empty
from builtin_interfaces.msg import Duration as MsgDuration
from moveit_msgs.msg import (
    Constraints, JointConstraint, PositionConstraint, OrientationConstraint,
    MoveItErrorCodes, CollisionObject, PlanningScene,
    AttachedCollisionObject, WorkspaceParameters,
)
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, PointCloud2, PointField
from shape_msgs.msg import SolidPrimitive



# ─── Scene constants ──────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    pose: 'Pose'
    dims: list  # [dx, dy, dz] in metres

RED_CONTAINER = {
    'center_xy':  (0.0, 0.5),   # (x, y) world frame
    'width':   0.3,           # inner x dimension
    'depth':   0.3,           # inner y dimension
    'height':  0.02,           # wall height above table surface
    'table_z': 0.0,           # table surface z
}

BLUE_CONTAINER = {
    'center_xy':  (0.0, -0.5),   # (x, y) world frame
    'width':   0.3,           # inner x dimension
    'depth':   0.3,           # inner y dimension
    'height':  0.02,           # wall height above table surface
    'table_z': 0.0,           # table surface z
}

# ─── RobotInterface ───────────────────────────────────────────────────────────

class RobotInterface(Node):
    """
    Exposes async goal-send helpers for MoveIt 2 and the gripper controller.

    BT nodes hold a reference to this node and call send_*() from
    initialise(), then poll results from update(). The executor processes
    callbacks between ticks so futures resolve without deadlocking.

    Public attributes
    -----------------
    _detected_objects : dict[str, DetectedObject]
        Latest pose + dims for each object. Updated continuously from the
        Gazebo pose bridge. Read via the blackboard, not directly.
    gripper_pos : list[float]
        Current finger positions [joint1, joint2].
    """

    GRIPPER_JOINTS = ['panda_finger_joint1', 'panda_finger_joint2']
    GRIPPER_OPEN   = [0.04, 0.04]
    GRIPPER_CLOSED = [0.0,  0.0]
    GRIPPER_LINKS  = ['panda_link8', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']

    TOP_DOWN_ORIENTATION = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

    INITIAL_JOINTS = {
        'panda_joint1': 0.0,
        'panda_joint2': -0.2,
        'panda_joint3': 0.0,
        'panda_joint4': -1.0,
        'panda_joint5': 0.0,
        'panda_joint6':  1.0,
        'panda_joint7': 0.0,
    }

    def __init__(self):
        super().__init__('bt_pick_place')

        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_trajectory_controller/follow_joint_trajectory'
        )
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self._detected_objects: dict[str, DetectedObject] = {}
        self.attached_objects: set[str] = set()
        self.gripper_pos: list[float] = [0.04, 0.04]
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(
            CollisionObject,
            '/collision_object',
            self._on_collision_object,
            10
        )
        self._marker_id = 0 
        self._marker_pub = self.create_publisher(MarkerArray, '/grasp_markers', 10)

        self.get_logger().info('Waiting for /move_action ...')
        self._move_client.wait_for_server()
        self.get_logger().info('Waiting for gripper controller ...')
        self._gripper_client.wait_for_server()
        self._cloud_pub = self.create_publisher(PointCloud2, '/gpd_sample_cloud', 10)
        self._gz_attach_pubs: dict[str, object] = {}
        '''for obj_id in GZ_OBJECTS:
            self._gz_attach_pubs[obj_id] = (
                self.create_publisher(Empty, f'/{obj_id}/attach', 10),
                self.create_publisher(Empty, f'/{obj_id}/detach', 10),
            )
        '''
        self.log_lines: list[str] = []
        self._publish_table()
        self.get_logger().info('RobotInterface ready')

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, msg: str):
        """Append msg to the on-screen log buffer and the ROS logger."""
        self.get_logger().info(msg)
        self.log_lines.append(msg)
        self.log_lines = self.log_lines[-12:]

    def publish_pose_axes(self, pose: Pose, label: str, scale: float = 0.1):
        """Publish XYZ axis arrows at pose in RViz (x=red, y=green, z=blue)."""
        ma = MarkerArray()
        q = [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        ]
        o = pose.position
        for i, color in enumerate(colors):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = label
            m.id = self._marker_id; self._marker_id += 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale = Vector3(x=0.008, y=0.015, z=0.0)
            m.color = color
            m.lifetime = MsgDuration(sec=5)
            d = R[:, i] * scale
            m.points = [
                Point(x=o.x, y=o.y, z=o.z),
                Point(x=o.x + float(d[0]), y=o.y + float(d[1]), z=o.z + float(d[2])),
            ]
            ma.markers.append(m)
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = label; m.id = self._marker_id; self._marker_id += 1
        m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
        m.pose = copy.deepcopy(pose); m.pose.position.z += scale + 0.02
        m.scale.z = 0.03
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = label; m.lifetime = MsgDuration(sec=5)
        ma.markers.append(m)
        self._marker_pub.publish(ma)

    def send_joints_goal(self, joint_positions: dict, acm_object_id: str = None):
        """Send a joint-space goal to MoveIt. Returns a Future[GoalHandle]."""
        goal = self._build_move_goal()
        c = Constraints()
        for name, value in joint_positions.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)
        if acm_object_id:
            goal.planning_options.planning_scene_diff = self._build_acm_scene(acm_object_id)
        return self._move_client.send_goal_async(goal)

    def send_pose_goal(self, pose: Pose,
                       acm_object_id: str = None,
                       position_tolerance: float = 0.05,
                       orientation_tolerance: float = 0.2):
        """Send a Cartesian pose goal to MoveIt. Returns a Future[GoalHandle]."""
        goal = self._build_move_goal()
        goal.request.goal_constraints.append(
            self._pose_to_constraints(pose, position_tolerance, orientation_tolerance)
        )
        if acm_object_id:
            goal.planning_options.planning_scene_diff = self._build_acm_scene(acm_object_id)
        return self._move_client.send_goal_async(goal)

    def send_gripper_goal(self, open: bool):
        """Send a gripper open/close goal. Returns a Future[GoalHandle]."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = self.GRIPPER_OPEN if open else self.GRIPPER_CLOSED
        pt.time_from_start = Duration(sec=2, nanosec=0)
        goal.trajectory.points.append(pt)
        for name in self.GRIPPER_JOINTS:
            tol = JointTolerance()
            tol.name = name
            tol.position = 0.01
            goal.goal_tolerance.append(tol)
        goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
        return self._gripper_client.send_goal_async(goal)

    def attach_object(self, object_id: str):
        """Attach object to the gripper in the MoveIt planning scene and Gazebo."""
        obj = self._detected_objects.get(object_id)
        self.log(
            f"In attach_object in robot interface, attach {object_id}. "
            f"Exists in detected_objects? {'YES' if obj is not None else 'NO'}. "
            f"Pose: {obj.pose if obj is not None else 'N/A'}"
        )
        if obj is None:
            self.log(f"[ATTACH ERROR] {object_id} not in detected_objects")
            return

        co = CollisionObject()
        co.id = object_id
        co.header.frame_id = 'world'
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = obj.dims

        co.primitives.append(box)
        co.primitive_poses.append(obj.pose)
        aco = AttachedCollisionObject()
        aco.object = co
        aco.link_name = 'panda_hand'
        #aco.object.id = object_id
        #aco.object.header.frame_id = 'world'
        #aco.object.operation = CollisionObject.ADD
        aco.touch_links = self.GRIPPER_LINKS
        scene = PlanningScene(is_diff=True)
        scene.robot_state.attached_collision_objects.append(aco)
        remove = CollisionObject()
        remove.id = object_id
        remove.header.frame_id = 'world'
        remove.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(remove)
        self._scene_pub.publish(scene)
        #if object_id in self._gz_attach_pubs:
            #self._gz_attach_pubs[object_id][0].publish(Empty())
        self.attached_objects.add(object_id)

    def detach_object(self, object_id: str):
        """Detach object from the gripper in the MoveIt planning scene and Gazebo."""
        aco = AttachedCollisionObject()
        aco.link_name = 'panda_hand'
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.robot_state.attached_collision_objects.append(aco)
        scene.robot_state.is_diff = True
        self._scene_pub.publish(scene)
        #if object_id in self._gz_attach_pubs:
            #self._gz_attach_pubs[object_id][1].publish(Empty())
        self.attached_objects.discard(object_id)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name == 'panda_finger_joint1':
                self.gripper_pos[0] = pos
            elif name == 'panda_finger_joint2':
                self.gripper_pos[1] = pos

    def _on_collision_object(self, msg: CollisionObject):
        # Handle removal
        if msg.operation == CollisionObject.REMOVE:
            if msg.id in self._detected_objects:
                del self._detected_objects[msg.id]
            return

        if not msg.primitives:
            return
        self.get_logger().info(f'ID: {msg.id}')
        for i, primitive in enumerate(msg.primitives):
            dims = list(primitive.dimensions)

            pose = msg.primitive_poses[i]
            
            self._detected_objects[msg.id] = DetectedObject(
            pose=pose,
            dims=dims
        )
        
    

    def _publish_table(self):
        scene = PlanningScene(is_diff=True)

        table = CollisionObject()
        table.header.frame_id = 'world'
        table.header.stamp = self.get_clock().now().to_msg()
        table.id = 'table'
        table.operation = CollisionObject.ADD
        
        table_dims = [1.0, 1.5, 0.35] # MUST MATCH SDF
        table_pos = [0.70, 0.0, 0.2]  # MUST MATCH SDF
        
        table.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=table_dims))
        table.primitive_poses.append(
            Pose(position=Point(x=table_pos[0], y=table_pos[1], z=table_pos[2]),
                 orientation=Quaternion(w=1.0)))
        scene.world.collision_objects.append(table)

        # 2. Containers: Moved forward by 0.3m
        container_specs = [
            {'id': 'red_container', 'pos': [0.0, 0.5, 0.01]}, # MUST MATCH SDF
            {'id': 'blue_container', 'pos': [0.0, -0.5, 0.01]} # MUST MATCH SDF
        ]

        wall_thickness = 0.01
        c_width, c_depth, c_height = 0.3, 0.3, 0.02

        for spec in container_specs:
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.header.stamp = self.get_clock().now().to_msg()
            co.id = spec['id']
            co.operation = CollisionObject.ADD
            
            cx, cy, cz = spec['pos']
            walls = [
                ([c_width, wall_thickness, c_height], [cx, cy - c_depth/2, cz]),
                ([c_width, wall_thickness, c_height], [cx, cy + c_depth/2, cz]),
                ([wall_thickness, c_depth, c_height], [cx - c_width/2, cy, cz]),
                ([wall_thickness, c_depth, c_height], [cx + c_width/2, cy, cz]),
            ]

            for d, p in walls:
                co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=d))
                co.primitive_poses.append(
                    Pose(position=Point(x=p[0], y=p[1], z=p[2]),
                         orientation=Quaternion(w=1.0)))
            
            scene.world.collision_objects.append(co)

        self._scene_pub.publish(scene)
    def _build_move_goal(self) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.planner_id = 'RRTConnectkConfigDefault'
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 15.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        ws = WorkspaceParameters()
        ws.header.frame_id = 'world'
        ws.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        ws.max_corner = Vector3(x=2.0, y=2.0, z=2.0)
        goal.request.workspace_parameters = ws
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.request.start_state.is_diff = True
        return goal

    def _pose_to_constraints(self, pose: Pose,
                              position_tolerance: float,
                              orientation_tolerance: float) -> Constraints:
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = 'world'
        pc.link_name = 'panda_hand_tcp'
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [position_tolerance]
        pc.constraint_region.primitives.append(region)
        region_pose = Pose()
        region_pose.position = copy.deepcopy(pose.position)
        region_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pc.constraint_region.primitive_poses.append(region_pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)
        oc = OrientationConstraint()
        oc.header.frame_id = 'world'
        oc.link_name = 'panda_hand_tcp'
        oc.orientation = copy.deepcopy(pose.orientation)
        oc.absolute_x_axis_tolerance = orientation_tolerance
        oc.absolute_y_axis_tolerance = orientation_tolerance
        oc.absolute_z_axis_tolerance = orientation_tolerance
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    def _build_acm_scene(self, object_id: str) -> PlanningScene:
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        acm = scene.allowed_collision_matrix

        acm.default_entry_names = [object_id]
        acm.default_entry_values = [True]
        return scene
