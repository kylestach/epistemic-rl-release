from enum import IntEnum
import collections
from typing import Any, Dict, List, Tuple

import rospy
import tf2_ros
import tf2_geometry_msgs as tf2gm
import geometry_msgs.msg as gm
import std_msgs.msg as stdm
import sensor_msgs.msg as sm
import nav_msgs.msg as nm

import numpy as np
import quaternion as npq
from threading import RLock


# Enum for the state machine
class State(IntEnum):
    LEARNING = 0
    COLLISION = 1
    INVERTED = 2
    STUCK = 3
    TELEOP = 4
    TELEOP_RECORD = 5
    RECOVERY = 6
    RECOVERY_PAUSE = 7


class InferenceStateMachine:
    def __init__(self, config: Dict[str, Any]):
        self.thresh_z_flip = config.get("thresh_z_flip", 0.5)
        self.thresh_xy_crash = config.get("thresh_xy_crash", 2.0)
        self.stuck_duration = config.get("stuck_duration", 3.0)
        self.stuck_radius = config.get("stuck_radius", 0.1)
        self.recovery_time = config.get("recovery_time", 2.0)
        self.pause_time = config.get("pause_time", 0.5)

        self.position_history_lock = RLock()
        self.position_history = collections.deque()
        self.last_was_recorded = False
        self.did_crash = False
        self.did_flip = False
        self.did_get_stuck = False

        self.state = None
        self.enter_state(State.LEARNING)

        self.has_gazebo_uninvert = True
        self.last_position = gm.Point()
        self.flips_from_odom = config.get("flips_from_odom", False)
        self.recovery_direction = 0
        self.last_forward_action = 0

    def setup_ros(self):
        self.transform_buffer = tf2_ros.Buffer(rospy.Duration.from_sec(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.transform_buffer)
        self.imu_sub = rospy.Subscriber(
            rospy.get_param("~imu_topic", "/imu/data"), sm.Imu, self.imu_callback
        )
        self.mode_publisher = rospy.Publisher(
            rospy.get_param("~mode_topic", "/offroad/mode"),
            stdm.String,
            queue_size=1,
        )
        self.odom_sub = rospy.Subscriber(
            rospy.get_param("~odom_topic", "/odom"),
            nm.Odometry,
            self.odom_callback,
        )

        if self.has_gazebo_uninvert:
            import gazebo_msgs.msg as gzm
            import gazebo_msgs.srv as gzs

            self.gz_set_state = rospy.ServiceProxy(
                "/gazebo/set_model_state", gzs.SetModelState
            )

    def imu_callback(self, imu: sm.Imu):
        tx = self.transform_buffer.lookup_transform(
            "base_link", imu.header.frame_id, rospy.Time()
        )

        # Transform the acceleration into the base_link frame
        robot_linear_acceleration = tf2gm.do_transform_vector3(
            gm.Vector3Stamped(vector=imu.linear_acceleration), tx
        ).vector

        if (
            not self.flips_from_odom
            and robot_linear_acceleration.z < -self.thresh_z_flip * 9.81
        ):
            self.did_flip = True

        if (
            imu.linear_acceleration.x**2 + imu.linear_acceleration.y**2
            > (self.thresh_xy_crash * 9.81) ** 2
        ):
            self.did_crash = True

    def odom_callback(self, odom: nm.Odometry):
        stuck_window_begin = odom.header.stamp - rospy.Duration.from_sec(
            self.stuck_duration
        )

        self.last_position = odom.pose.pose.position
        position = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])

        if self.flips_from_odom:
            orientation = odom.pose.pose.orientation
            robot_down = npq.as_rotation_matrix(
                npq.from_float_array(
                    [orientation.w, orientation.x, orientation.y, orientation.z]
                )
            )[:, -1]
            if robot_down[-1] < -self.thresh_z_flip:
                self.did_flip = True

        with self.position_history_lock:
            if len(self.position_history) > 0:
                # Check if we've been within a radius of the same position for the last few seconds
                if self.position_history[0][0] < stuck_window_begin:
                    distances_to_pos = [
                        np.linalg.norm(ph - position)
                        for t, ph in self.position_history
                        if t > stuck_window_begin
                    ]
                    if len(distances_to_pos) > 0 and all(d < self.stuck_radius for d in distances_to_pos):
                        self.did_get_stuck = True

                # Clear outdated positions
                while self.position_history[0][
                    0
                ] < stuck_window_begin - rospy.Duration.from_sec(1.0):
                    self.position_history.popleft()

            # Add the current position to the history
            self.position_history.append((odom.header.stamp, position))

    def enter_state(self, state):
        with self.position_history_lock:
            self.position_history.clear()
        self.state = state
        self.state_enter_time = rospy.Time.now()

    def enter_recovery(self):
        assert (
            self.state == State.COLLISION or self.state == State.STUCK
        ), "Cannot enter recovery from state " + str(self.state)
        self.enter_state(State.RECOVERY)
        self.recovery_steer = np.random.uniform(-1, 1)

    def try_uninvert(self):
        if self.has_gazebo_uninvert:
            import gazebo_msgs.msg as gzm
            import gazebo_msgs.srv as gzs

            self.gz_set_state.call(
                gzs.SetModelStateRequest(
                    gzm.ModelState(
                        model_name="jackal",
                        pose=gm.Pose(
                            position=self.last_position,
                            orientation=gm.Quaternion(0, 0, 0, 1),
                        ),
                        twist=gm.Twist(),
                    )
                ),
            )

    def time_in_state(self):
        return rospy.Time.now() - self.state_enter_time

    def check_stuck(self):
        result, self.did_get_stuck = self.did_get_stuck, False
        return result

    def check_inverted(self):
        result, self.did_flip = self.did_flip, False
        return result

    def check_collided(self):
        result, self.did_crash = self.did_crash, False
        return result

    def should_record(self):
        return self.state == State.LEARNING or self.state == State.TELEOP_RECORD

    def is_autonomous(self):
        return self.state == State.LEARNING

    def is_teleop_record(self):
        return self.state == State.TELEOP_RECORD

    def is_teleop(self):
        return self.state == State.TELEOP

    def is_recovery(self):
        return self.state == State.RECOVERY

    def should_publish_action(self):
        return (
            self.state == State.LEARNING
            or self.state == State.TELEOP_RECORD
            or self.state == State.RECOVERY
        )

    def handle_teleop(self):
        self.enter_state(State.TELEOP)

    def handle_teleop_record(self):
        self.enter_state(State.TELEOP_RECORD)

    def tick_state(self):
        # Process transitions
        if self.state == State.LEARNING:
            # Learning state; check for transitions
            if self.check_inverted():
                self.enter_state(State.INVERTED)
                self.try_uninvert()
            elif self.check_collided():
                self.recovery_direction = -1 if self.last_forward_action > 0 else 1
                self.enter_state(State.COLLISION)
            elif self.check_stuck():
                self.recovery_direction = -1 if self.last_forward_action > 0 else 1
                self.enter_state(State.STUCK)

        elif self.state == State.COLLISION or self.state == State.STUCK:
            # Transient state; immediately proceed to recovery
            if self.time_in_state() > rospy.Duration(0.2):
                self.enter_recovery()

        elif self.state == State.TELEOP or self.state == State.TELEOP_RECORD:
            if self.time_in_state() > rospy.Duration(self.pause_time):
                self.enter_state(State.LEARNING)

        elif self.state == State.RECOVERY:
            if self.time_in_state() > rospy.Duration(self.recovery_time):
                self.enter_state(State.LEARNING)

        elif self.state == State.RECOVERY_PAUSE:
            if self.time_in_state() > rospy.Duration(self.pause_time):
                self.enter_state(State.LEARNING)

        elif self.state == State.INVERTED:
            if not self.check_inverted():
                self.enter_state(State.RECOVERY_PAUSE)
            elif self.time_in_state() > rospy.Duration(1.0):
                self.try_uninvert()
                self.enter_state(State.INVERTED)
        else:
            raise AssertionError("Invalid state " + str(self.state))

        self.did_crash = False
        self.did_flip = False
        self.did_get_stuck = False

        last_recorded = self.last_was_recorded
        self.last_was_recorded = self.should_record()

        self.mode_publisher.publish(stdm.String(self.state.name))

        return last_recorded or self.should_record()
