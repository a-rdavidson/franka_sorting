#!/usr/bin/env python3
"""
LAB03 — Behavior Tree pick-and-place.

Run after launch_ctrl:
    ros2 run panda_moveit_config bt_pick_place.py
"""

import copy
import time
import rclpy
import py_trees
import py_trees_ros
import numpy as np  # noqa: F401

from dataclasses import dataclass, field
from typing import Optional

from geometry_msgs.msg import Pose  # noqa: F401
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

from robot_interface import RobotInterface, RED_CONTAINER, BLUE_CONTAINER, DetectedObject
from action_node import ActionNode, GripperNode
from gpd import sample_cuboid_surface, detect_grasps, gpd_to_panda_pose, GraspCandidate  # noqa: F401


# ─── Stats tracker ────────────────────────────────────────────────────────────

@dataclass
class CycleRecord:
    obj_id:        str
    start_time:    float
    end_time:      float   = 0.0
    success:       bool    = False
    failure_phase: str     = ''   # which node name caused the failure


class StatsTracker:
    """
    Collects per-cycle and aggregate statistics for the sorting run.

    Lifecycle
    ---------
    SelectObject SUCCESS          → cycle_start(obj_id)
    CheckObjectIsAttached FAILURE → grasp_miss()
    ProposeGrasps FAILURE         → grasp_plan_fail()
    Any ActionNode subclass FAIL  → planning_fail(node_name)
    MoveToHome retry              → home_retry()
    CheckAllObjectsInContainer    → cycle_end(success, failure_phase)
    main() finally                → print_summary()
    """

    def __init__(self):
        self.session_start: float               = time.monotonic()
        self.cycles:        list[CycleRecord]   = []
        self._cur:          Optional[CycleRecord] = None

        # Aggregate counters (some failures abort before cycle_end is reached)
        self.grasp_misses:       int             = 0   # gripper closed on air
        self.grasp_plan_fails:   int             = 0   # ProposeGrasps exhausted
        self.planning_fails:     dict[str, int]  = {}  # {node_name: count}
        self.home_retries:       int             = 0

    # ── called by BT nodes ────────────────────────────────────────────────

    def cycle_start(self, obj_id: str):
        self._cur = CycleRecord(obj_id=obj_id, start_time=time.monotonic())

    def cycle_end(self, success: bool, failure_phase: str = ''):
        if self._cur is None:
            return
        self._cur.end_time     = time.monotonic()
        self._cur.success      = success
        self._cur.failure_phase = failure_phase
        self.cycles.append(self._cur)
        self._cur = None

    def grasp_miss(self):
        self.grasp_misses += 1

    def grasp_plan_fail(self):
        self.grasp_plan_fails += 1

    def planning_fail(self, node_name: str):
        self.planning_fails[node_name] = self.planning_fails.get(node_name, 0) + 1

    def home_retry(self):
        self.home_retries += 1

    # ── summary ───────────────────────────────────────────────────────────

    def print_summary(self):
        total_s   = time.monotonic() - self.session_start
        n_total   = len(self.cycles)
        n_success = sum(1 for c in self.cycles if c.success)
        n_fail    = n_total - n_success

        durations = [c.end_time - c.start_time for c in self.cycles if c.end_time > 0]
        avg_cycle = (sum(durations) / len(durations)) if durations else 0.0
        min_cycle = min(durations) if durations else 0.0
        max_cycle = max(durations) if durations else 0.0
        std_cycle = float(np.std(durations)) if len(durations) > 1 else 0.0

        throughput = (n_success / (total_s / 60.0)) if total_s > 0 else 0.0
        success_rate = (n_success / n_total * 100.0) if n_total > 0 else 0.0

        W = 54  # inner width of the box

        def row(label: str, value: str) -> str:
            gap = W - len(label) - len(value)
            return f'│  {label}{" " * gap}{value}  │'

        def div() -> str:
            return f'├{"─" * (W + 4)}┤'

        def section(title: str) -> str:
            pad = W + 4 - len(title) - 2
            left  = pad // 2
            right = pad - left
            return f'│{" " * left}{title}{" " * right}│'

        lines = []
        lines.append(f'┌{"─" * (W + 4)}┐')
        lines.append(section('SORTING SESSION STATISTICS'))
        lines.append(div())

        # ── Session
        lines.append(section('Session'))
        lines.append(row('Total runtime',
                          f'{int(total_s // 60)}m {total_s % 60:.1f}s'))
        lines.append(row('Objects attempted', str(n_total)))
        lines.append(row('Objects sorted',    str(n_success)))
        lines.append(row('Sort failures',     str(n_fail)))
        lines.append(row('Success rate',      f'{success_rate:.1f} %'))
        lines.append(row('Throughput',        f'{throughput:.2f} objects / min'))

        # ── Cycle times
        lines.append(div())
        lines.append(section('Cycle time  (SelectObject → container confirmed)'))
        if durations:
            lines.append(row('Mean',   f'{avg_cycle:.1f} s'))
            lines.append(row('Min',    f'{min_cycle:.1f} s'))
            lines.append(row('Max',    f'{max_cycle:.1f} s'))
            lines.append(row('Std dev', f'{std_cycle:.1f} s'))
        else:
            lines.append(row('No completed cycles', '—'))

        # ── Failures by phase
        lines.append(div())
        lines.append(section('Failure breakdown'))
        lines.append(row('Grasp misses  (gripper closed on air)', str(self.grasp_misses)))
        lines.append(row('Grasp plan failures  (GPD exhausted)',  str(self.grasp_plan_fails)))
        lines.append(row('Home retries',                          str(self.home_retries)))
        if self.planning_fails:
            for node, count in sorted(self.planning_fails.items()):
                lines.append(row(f'  Planning fail — {node}', str(count)))
        else:
            lines.append(row('Planning failures', '0'))

        # ── Per-object breakdown
        if self.cycles:
            lines.append(div())
            lines.append(section('Per-object log'))
            for i, c in enumerate(self.cycles, 1):
                dur    = c.end_time - c.start_time if c.end_time else 0.0
                status = 'OK  ' if c.success else 'FAIL'
                phase  = f'  ← {c.failure_phase}' if not c.success and c.failure_phase else ''
                entry  = f'{i:>2}. [{status}] {c.obj_id:<14} {dur:>5.1f}s{phase}'
                # Truncate to fit box width
                entry  = entry[: W]
                gap    = W - len(entry)
                lines.append(f'│  {entry}{" " * gap}  │')

        lines.append(f'└{"─" * (W + 4)}┘')
        print('\n' + '\n'.join(lines) + '\n')


# ─── Provided nodes ───────────────────────────────────────────────────────────

class ReadScene(py_trees.behaviour.Behaviour):
    """
    Entry point for every run: clean up stale planning-scene state, then
    wait for fresh collision-object data for ALL known object.

    initialise(): detach all known objects from the previous run,
                  clear the object cache so stale latched messages are ignored.
    update():     poll detected_objects; once all objects in GZ_OBJECTS are
                  detected and stable, add them to the planning scene and
                  return SUCCESS.

    Blackboard writes: /detected_objects (dict[str, DetectedObject])
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('ReadScene')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='ReadScene')
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.WRITE)

    _STABLE_TICKS = 10  # ~2 s at 200 ms/tick
    
    def _is_sorted(self, obj_id, obj_pose) -> bool: 
        container = RED_CONTAINER if 'red' in obj_id.lower() else BLUE_CONTAINER

        cx, cy = container['center_xy']
        hx, hy = container['width'] / 2.0, container['depth'] / 2.0

        return (abs(obj_pose.position.x - cx) < hx and 
                abs(obj_pose.position.y - cy) < hy)

    def initialise(self) :

        for obj_id in self.robot._detected_objects.keys():
            self.robot.detach_object(obj_id)
        self.robot.log('[INFO] ReadScene: detached stale objects')

        self._stable = {}   # obj_id -> consecutive stable ticks
        self._last_z = {}   # obj_id -> last seen z

    def update(self):
        all_stable = True
        for obj_id in self.robot._detected_objects.keys():
            obj = self.robot._detected_objects.get(obj_id)
            if obj is None:
                all_stable = False
                continue
            z = obj.pose.position.z
            last = self._last_z.get(obj_id)
            if last is None or abs(z - last) > 0.001:
                self._stable[obj_id] = 0
            else:
                self._stable[obj_id] = self._stable.get(obj_id, 0) + 1
            self._last_z[obj_id] = z
            if self._stable.get(obj_id, 0) < self._STABLE_TICKS:
                all_stable = False

        if not all_stable:
            return py_trees.common.Status.RUNNING


        # All objects stable — add to planning scene
        scene = PlanningScene(is_diff=True)
        for obj_id in self.robot._detected_objects.keys():
            obj = self.robot._detected_objects[obj_id]

            if self._is_sorted(obj_id, obj.pose):
                # Skip objects already in container 

                self.robot.log(f"[INFO] ReadScene: skipping {obj_id}, already sorted")
                co = CollisionObject()
                co.id = obj_id 
                co.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(co) 
                continue
            # Otherwise add object as obstacle
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.header.stamp = self.robot.get_clock().now().to_msg()
            co.id = obj_id
            co.operation = CollisionObject.ADD
            co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=obj.dims))
            co.primitive_poses.append(obj.pose)
            scene.world.collision_objects.append(co)

        self.robot._scene_pub.publish(scene)
        self.bb.detected_objects = dict(self.robot._detected_objects)
        return py_trees.common.Status.SUCCESS


class MoveToHome(ActionNode):
    """Reset to a known-good state, then move home.

    On every entry (including restarts after failure):
      1. Open gripper (fire-and-forget)
      2. Gz-detach + planning-scene remove all objects
      3. Move arm to INITIAL_JOINTS

    Always succeeds at resetting state regardless of prior situation.
    Retries the motion up to _MAX_RETRIES times on transient planner failures.
    """

    _MAX_RETRIES = 3

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToHome', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToHome')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)
        self._retries = 0

    def _reset_state(self):
        # Open gripper (best-effort, don't wait for result)
        self.robot.send_gripper_goal(open=True)
        # Detach and remove all objects
        if (self.bb.target_object_id):
            self.robot.detach_object(self.bb.target_object_id)
        for obj_id in self.robot._detected_objects.keys():
            self.robot.detach_object(obj_id)
        # Clear planning scene
        scene = PlanningScene(is_diff=True)
        for obj_id in self.bb.detected_objects.keys():
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.header.stamp = self.robot.get_clock().now().to_msg()
            co.id = obj_id
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)
        self.robot._scene_pub.publish(scene)

    def initialise(self):
        self._retries = 0
        self._reset_state()
        super().initialise()

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE and self._retries < self._MAX_RETRIES:
            self._retries += 1
            self.robot.log(f'[WARN] MoveToHome: retrying ({self._retries}/{self._MAX_RETRIES})')
            self.robot.stats.home_retry()                    
            self._reset_state()
            super().initialise()
            return py_trees.common.Status.RUNNING
        if status == py_trees.common.Status.FAILURE:
            self.robot.stats.planning_fail('MoveToHome')    
        return status

    def _send_goal(self):
        return self.robot.send_joints_goal(RobotInterface.INITIAL_JOINTS)



# ─── Pick sub-nodes ───────────────────────────────────────────────────────────


class OpenGripper(GripperNode):
    def __init__(self, robot: RobotInterface):
        super().__init__('OpenGripper', robot)

    def _send_goal(self):
        return self.robot.send_gripper_goal(open=True)


class CloseGripper(GripperNode):
    def __init__(self, robot: RobotInterface):
        super().__init__('CloseGripper', robot)

    def _send_goal(self):
        return self.robot.send_gripper_goal(open=False)


class MoveToPreGrasp(ActionNode):
    """Move to 15 cm above the top-ranked grasp proposal.

    Blackboard reads: /grasp_proposals (list[Pose])
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToPreGrasp', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToPreGrasp')
        self.bb.register_key('/grasp_proposals', access=py_trees.common.Access.READ)

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.robot.stats.planning_fail('MoveToPreGrasp')  
        return status

    def _send_goal(self):
        pre = copy.deepcopy(self.bb.grasp_proposals[0])
        pre.position.z += 0.08
        self.robot.publish_pose_axes(pre, 'pre_grasp', scale=0.1)
        return self.robot.send_pose_goal(pre)


class MoveToGrasp(ActionNode):
    """Descend to the top-ranked grasp proposal.

    Blackboard reads: /grasp_proposals (list[Pose]), /target_object_id (str)
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToGrasp', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToGrasp')
        self.bb.register_key('/grasp_proposals',  access=py_trees.common.Access.READ)

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.robot.stats.planning_fail('MoveToGrasp')    
        return status

    def _send_goal(self):
        grasp = self.bb.grasp_proposals[0]
        self.robot.publish_pose_axes(grasp, 'grasp')
        return self.robot.send_pose_goal(
            grasp,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
        )


class CheckObjectIsAttached(py_trees.behaviour.Behaviour):
    """
    Verify the gripper is not fully closed — if fingers stopped before
    GRIPPER_CLOSED, something is between them.

    Returns SUCCESS if gap > threshold, FAILURE if fingers closed fully
    (missed the object).
    """

    # If total finger gap is above this the gripper caught something
    _CONTACT_THRESHOLD = sum(RobotInterface.GRIPPER_CLOSED) + 0.005

    def __init__(self, robot: RobotInterface):
        super().__init__('CheckObjectIsAttached')
        self.robot = robot

    def update(self):
        gap = sum(self.robot.gripper_pos)
        if gap > self._CONTACT_THRESHOLD:
            self.robot.log(f'[OK]   CheckObjectIsAttached: gap={gap:.4f}')
            return py_trees.common.Status.SUCCESS
        self.robot.log(f'[FAIL] CheckObjectIsAttached: gap={gap:.4f} (fully closed)')
        self.robot.stats.grasp_miss()                        
        return py_trees.common.Status.FAILURE


class DetachObject(py_trees.behaviour.Behaviour):
    """Detach the target object from the planning scene. Always returns SUCCESS.

    Blackboard reads: /target_object_id (str)
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('DetachObject')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='DetachObject')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def update(self):
        self.robot.detach_object(self.bb.target_object_id)
        self.robot.log(f'[INFO] DetachObject: {self.bb.target_object_id}')
        return py_trees.common.Status.SUCCESS


class AttachObject(py_trees.behaviour.Behaviour):
    """Attach the target object in the planning scene.

    Blackboard reads: /target_object_id (str)
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('AttachObject')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='AttachObject')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def update(self):
        self.robot.log(f"[INFO] AttachObject: {self.bb.target_object_id}") 
        self.robot.attach_object(self.bb.target_object_id)
        return py_trees.common.Status.SUCCESS


class Retreat(ActionNode):
    """Move back to 15 cm above the grasp pose after closing.

    Blackboard reads: /grasp_proposals (list[Pose])
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('Retreat', robot)
        self.bb = py_trees.blackboard.Client(name='Retreat')
        self.bb.register_key('/grasp_proposals', access=py_trees.common.Access.READ)

    def initialise(self):
        co = CollisionObject()
        co.header.frame_id = 'world'
        co.header.stamp = self.robot.get_clock().now().to_msg()
        co.id = 'table'
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        self.robot._scene_pub.publish(scene)
        super().initialise()

    def terminate(self, new_status):
        self.robot._publish_table()
        super().terminate(new_status)

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.robot.stats.planning_fail('Retreat')        
        return status

    def _send_goal(self):
        retreat = copy.deepcopy(self.bb.grasp_proposals[0])
        retreat.position.z += 0.20
        return self.robot.send_pose_goal(
            retreat,
            position_tolerance=0.01,
            orientation_tolerance=0.05,
        )


# ─── Drop Nodes ──────────────────────────────────────────────────────────

class MoveToDrop(ActionNode):
    """Move to the drop pose written by ProposeDropPose.

    Blackboard reads: /drop_pose        (Pose)
                      /target_object_id (str)  — for ACM during approach
    """

    _TIMEOUT_SEC = 60.0

    def __init__(self, robot: RobotInterface):
        super().__init__('MoveToDrop', robot)
        self.bb = py_trees.blackboard.Client(name='MoveToDrop')
        self.bb.register_key('/drop_pose',        access=py_trees.common.Access.READ)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.robot.stats.planning_fail('MoveToDrop')     
        return status

    def _send_goal(self):
        drop = self.bb.drop_pose
        self.robot.publish_pose_axes(drop, 'drop_pose', scale=0.10)
        return self.robot.send_pose_goal(
            drop,
            acm_object_id=self.bb.target_object_id,
            position_tolerance=0.03,
            orientation_tolerance=0.15,
        )


class CheckAllObjectsInContainer(py_trees.behaviour.Behaviour):
    """Poll until the current object lands in the container, then check all.

    RUNNING : current object not yet stable in container
    FAILURE : timeout, or current object landed but others remain unsorted
              (triggers RepeatAlways restart → next object)
    SUCCESS : all objects confirmed in container → shutdown

    Blackboard reads: /target_object_id (str)
                      /container         (dict)

    Note: reads live poses from robot._detected_objects (not the BB snapshot)
    since it runs after the drop and needs current positions.
    """

    _TIMEOUT_SEC = 20.0
    _STABLE_TICKS = 5

    def __init__(self, robot: RobotInterface):
        super().__init__('CheckAllObjectsInContainer')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='CheckAllObjectsInContainer')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self.bb.register_key('/red_container',    access=py_trees.common.Access.READ)
        self.bb.register_key('/blue_container',   access=py_trees.common.Access.READ)
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)

    def initialise(self):
        self._deadline = (self.robot.get_clock().now().nanoseconds
                          + int(self._TIMEOUT_SEC * 1e9))
        self._stable = 0

    def _in_correct_container_(self, pose, obj_id) -> bool:
        if pose is None:
            return False
        container = self.bb.red_container if 'red' in obj_id.lower() else self.bb.blue_container

        cx, cy = container['center_xy']
        hx, hy = container['width'] / 2.0, container['depth'] / 2.0
        return (abs(pose.position.x - cx) < hx and
                abs(pose.position.y - cy) < hy)

    def update(self):
        if self.robot.get_clock().now().nanoseconds > self._deadline:
            self.robot.log('[FAIL] CheckAllObjectsInContainer: timeout')
            self.robot.stats.cycle_end(success=False,           
                                       failure_phase='drop timeout')
            return py_trees.common.Status.FAILURE

        #number_of_objects = len(self.robot._detected_objects)
        #if number_of_objects < len(self.bb.detected_objects):
        #    return py_trees.common.Status.RUNNING

        all_in = all(
            self._in_correct_container_(o.pose, o_id)
            for o_id, o in self.robot._detected_objects.items()
        )
        if all_in:
            self.robot.log('[OK]   Goal Status: SUCCESS — all objects sorted')
            self.robot.stats.cycle_end(success=True)            
            return py_trees.common.Status.SUCCESS

        self.robot.log(f'[OK]   {self.bb.target_object_id} in container, others remain')
        self.robot.stats.cycle_end(success=True)               
        return py_trees.common.Status.FAILURE


class RepeatAlways(py_trees.decorators.Decorator):
    """Repeat forever: restart the child on both SUCCESS and FAILURE."""
    def update(self):
        if self.decorated.status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.RUNNING
        return self.decorated.status


# ─── Lab03 TODOs ─────────────────────────────────────────────────────────────

def build_tree(robot: RobotInterface) -> py_trees.behaviour.Behaviour:  # noqa: ARG001
    
    seq = py_trees.composites.Sequence("PickAndPlace", memory=True)
    seq.add_children([
        # Reset Phase
        MoveToHome(robot),
        ReadScene(robot),
        SelectObject(robot),

        # Grasp Phase
        ProposeGrasps(robot),
        OpenGripper(robot), 
        MoveToPreGrasp(robot), 
        MoveToGrasp(robot), 
        CloseGripper(robot), 
        CheckObjectIsAttached(robot),
        AttachObject(robot), 
        Retreat(robot), 
        
        # Place Phase
        ProposeDropPose(robot), 
        MoveToDrop(robot), 
        OpenGripper(robot),
        DetachObject(robot),
        CheckAllObjectsInContainer(robot)
    ])
    
    root = RepeatAlways(name="Repeat", child=seq)
    return root


def init_blackboard():
    bb = py_trees.blackboard.Client(name='init')
    bb.register_key('/detected_objects',          access=py_trees.common.Access.WRITE)
    bb.register_key('/red_container',             access=py_trees.common.Access.WRITE)
    bb.register_key('/blue_container',            access=py_trees.common.Access.WRITE)
    bb.register_key('/target_container',          access=py_trees.common.Access.WRITE)
    bb.register_key('/target_object_id',          access=py_trees.common.Access.WRITE)
    bb.register_key('/grasp_proposals',           access=py_trees.common.Access.WRITE)
    bb.register_key('/drop_pose',                 access=py_trees.common.Access.WRITE)
    bb.detected_objects = dict()

    bb.red_container = RED_CONTAINER
    bb.blue_container = BLUE_CONTAINER
    bb.target_container = dict() 

    bb.target_object_id = ''
    bb.grasp_proposals = list()
    bb.drop_pose = None

class SelectObject(py_trees.behaviour.Behaviour):
    """
    Select the next unplaced object and write its id to the blackboard.

    An object is considered placed when its xy position is within the
    container footprint (z is ignored).

    Returns SUCCESS with /target_object_id written, or FAILURE if no
    unplaced objects remain. Normal termination is handled by
    CheckAllObjectsInContainer, so FAILURE here just indicates an unexpected
    state, like no blocks being reachable.
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('SelectObject')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='SelectObject')
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.WRITE)
        self.bb.register_key('/target_container', access=py_trees.common.Access.WRITE)

        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)
        self.bb.register_key('/red_container', access=py_trees.common.Access.READ)
        self.bb.register_key('/blue_container', access=py_trees.common.Access.READ)

    def _get_correct_container(self, obj_id): 
        if 'red' in obj_id.lower(): 
            return self.bb.red_container
        elif 'blue' in obj_id.lower(): 
            return self.bb.blue_container
        return self.bb.red_container # fallback to red

    def _in_container_xy(self, pose, container) -> bool:
        if pose is None or not container:
            return False
        cx, cy = container['center_xy']
        hx, hy = container['width'] / 2.0, container['depth'] / 2.0
        return (abs(pose.position.x - cx) < hx and
                abs(pose.position.y - cy) < hy)
        
    def update(self):
        self.robot.log(f'[INFO] SelectObject: Number of items {len(self.bb.detected_objects)}')
        for obj_id, obj in self.bb.detected_objects.items(): 
            correct_container = self._get_correct_container(obj_id)

            if not self._in_container_xy(obj.pose, correct_container): 
                self.bb.target_object_id = obj_id
                self.bb.target_container = correct_container
                self.robot.log(f'[INFO] SelectObject: selected {obj_id} for sorting')
                self.robot.stats.cycle_start(obj_id)           
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class ProposeGrasps(py_trees.behaviour.Behaviour):
    """
    Sample the target object surface, run GPD, filter candidates, and
    write valid grasp poses to /grasp_proposals.

    Retries up to _MAX_RETRIES times if no valid candidates are found,
    then returns FAILURE.
    """

    _MAX_RETRIES = 3

    def __init__(self, robot: RobotInterface):
        super().__init__('ProposeGrasps')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='ProposeGrasps')
        self.bb.register_key('/grasp_proposals', access=py_trees.common.Access.WRITE)
        self.bb.register_key('/detected_objects', access=py_trees.common.Access.READ)
        self.bb.register_key('/target_object_id', access=py_trees.common.Access.READ)
        self._retries = 0

    def initialise(self):
        self._retries = 0

    def update(self):
        if self._retries >= self._MAX_RETRIES: 
            self.robot.log('[FAIL] ProposeGrasps: Max retries reached')
            self.robot.stats.grasp_plan_fail()                 
            self.robot.stats.cycle_end(success=False,          
                                       failure_phase='ProposeGrasps')
            return py_trees.common.Status.FAILURE

        obj_id = self.bb.target_object_id
        obj = self.bb.detected_objects.get(obj_id)
        if obj is None: 
            self.robot.log(f'[FAIL] ProposeGrasps: {obj_id} not found in BB')
            self.robot.stats.cycle_end(success=False,          
                                       failure_phase='ProposeGrasps (obj missing)')
            return py_trees.common.Status.FAILURE

        center = (obj.pose.position.x, obj.pose.position.y, obj.pose.position.z)
        orientation = (obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w)
        cloud = sample_cuboid_surface(center, obj.dims, orientation=orientation)
        candidates = detect_grasps(cloud)

        valid_poses = []
        for cand in candidates: 

            approach_vec = cand.R[:, 0] # Approach as first col 
            z_alignment = approach_vec[2] # z_column

            self.robot.log(f'[DEBUG] Grasp Z: {cand.pos[2]:.3f}, Alignment: {z_alignment:.3f}')
            
            # -1 is perfectly vertical, -0.85 should allow 30 degree tilt
            if z_alignment > -0.85: 
                continue

            table_surface_z = 0.40
            # Don't allow grasps too close to the surface of table
            if cand.pos[2] < (table_surface_z + 0.01):
                continue
            
            dist_sq = cand.pos[0]**2 + cand.pos[1]**2
            # Avoid grasps too close or too far from robot base
            if dist_sq < 0.3**2 or dist_sq > 0.85**2:
                continue

            valid_poses.append(gpd_to_panda_pose(cand.pos, cand.R))
        if valid_poses: 
            self.bb.grasp_proposals = valid_poses
            self.robot.log(f'[OK]   ProposeGrasps: {len(valid_poses)} valid grasps generated.')
            return py_trees.common.Status.SUCCESS

        self._retries += 1
        self.robot.log(f'[WARN] ProposeGrasps: No valid grasps found. Retrying ({self._retries}/{self._MAX_RETRIES})...')
        return py_trees.common.Status.RUNNING


class ProposeDropPose(py_trees.behaviour.Behaviour):
    """
    Compute a drop pose above the container and write it to /drop_pose.

    The pose must be within the container footprint with sufficient
    clearance above the walls.
    """

    def __init__(self, robot: RobotInterface):
        super().__init__('ProposeDropPose')
        self.robot = robot
        self.bb = py_trees.blackboard.Client(name='ProposeDropPose')
        self.bb.register_key('/target_container', access=py_trees.common.Access.READ)
        self.bb.register_key('/drop_pose', access=py_trees.common.Access.WRITE)

    def update(self):
        cx, cy = self.bb.target_container['center_xy']
        
        # Half width - 3cm so object isn't dropped on the rim
        hx = (self.bb.target_container['width'] / 2.0) - 0.05
        hy = (self.bb.target_container['depth'] / 2.0) - 0.05

        drop_x = cx + np.random.uniform(-hx, hx)
        drop_y = cy + np.random.uniform(-hy, hy)
        
        # Add 10cm to height of container walls for drop height
        wall_height = self.bb.target_container.get('height') + self.bb.target_container.get('table_z')
        drop_z = wall_height + 0.10

        drop = Pose()
        drop.position.x = float(drop_x)
        drop.position.y = float(drop_y)
        drop.position.z = float(drop_z)
        
        drop.orientation = RobotInterface.TOP_DOWN_ORIENTATION

        self.bb.drop_pose = drop
        self.robot.log(f'[INFO] ProposeDropPose: Computed drop pose at ({drop_x:.3f}, {drop_y:.3f}, {drop_z:.3f})')
        return py_trees.common.Status.SUCCESS

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    robot = RobotInterface()
    robot.stats = StatsTracker()               # attach tracker to the robot handle

    init_blackboard()
    root = build_tree(robot)

    tree = py_trees_ros.trees.BehaviourTree(root=root, unicode_tree_debug=False)
    tree.setup(node=robot, timeout=15.0)

    def on_tick(t):
        log_panel = '\n'.join(robot.log_lines) if robot.log_lines else '(no log)'
        print('\033[2J\033[H' +
              py_trees.display.unicode_tree(t.root, show_status=True) +
              '\n' +
              py_trees.display.unicode_blackboard() +
              '\n─── log ───────────────────────────────\n' +
              log_panel)
        if t.root.status in (py_trees.common.Status.SUCCESS,
                             py_trees.common.Status.FAILURE):
            rclpy.shutdown()

    tree.post_tick_handlers.append(on_tick)
    robot.create_timer(0.2, tree.tick)

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        tree.shutdown()
        robot.destroy_node()
        robot.stats.print_summary()            # ← always prints, even on Ctrl-C


if __name__ == '__main__':
    main()
