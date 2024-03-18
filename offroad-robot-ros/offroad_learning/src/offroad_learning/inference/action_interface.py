import rospy

import numpy as np

from ackermann_msgs import msg as am
from geometry_msgs import msg as gm

from typing import Callable, Type, Union


class ActionInterface:
    def __init__(
        self,
        action_type: Type,
        teleop_topic: str,
        teleop_record_topic: str,
        action_topic: str,
    ):
        self.teleop_callback = None
        self.action_type = action_type
        self.teleop_topic = teleop_topic
        self.teleop_record_topic = teleop_record_topic
        self.action_topic = action_topic

    def setup_ros(self):
        self.teleop_subscriber = rospy.Subscriber(
            self.teleop_topic,
            self.action_type,
            self.receive_teleop,
        )
        self.teleop_record_subscriber = rospy.Subscriber(
            self.teleop_record_topic,
            self.action_type,
            self.receive_teleop_record,
        )
        self.action_publisher = rospy.Publisher(
            self.action_topic,
            self.action_type,
            queue_size=1,
        )

    def publish_action(self, action: np.ndarray):
        self.action_publisher.publish(self.array_to_action(action))

    def set_teleop_callback(self, callback: Callable[[np.ndarray, bool], None]):
        self.teleop_callback = callback

    def receive_teleop(self, msg):
        """
        Handle a teleop command from the joystick, which tells us not to run the agent.
        """
        if self.teleop_callback is not None:
            self.teleop_callback(self.action_to_array(msg), record=False)

    def receive_teleop_record(self, msg):
        """
        Handle a teleop command from the joystick, but record the data.
        """
        if self.teleop_callback is not None:
            self.teleop_callback(self.action_to_array(msg), record=True)

    def action_to_array(self, action: rospy.AnyMsg):
        raise NotImplementedError()

    def array_to_action(self, array: rospy.AnyMsg):
        raise NotImplementedError()


class AckermannInterface(ActionInterface):
    def __init__(
        self,
        teleop_topic: str,
        teleop_record_topic: str,
        action_topic: str,
    ):
        super().__init__(
            am.AckermannDriveStamped,
            teleop_topic,
            teleop_record_topic,
            action_topic,
        )

    def action_to_array(self, action: am.AckermannDriveStamped):
        """
        Convert an action to an array.
        """
        return np.array(
            [action.drive.speed, action.drive.steering_angle], dtype=np.float32
        )

    def array_to_action(self, array: np.ndarray):
        """
        Convert an array to an action.
        """
        msg = am.AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = array[0]
        msg.drive.steering_angle = array[1]
        return msg


class TwistInterface(ActionInterface):
    def __init__(
        self,
        teleop_topic: str,
        teleop_record_topic: str,
        action_topic: str,
        stamped: bool = False,
    ):
        super().__init__(
            gm.TwistStamped if stamped else gm.Twist,
            teleop_topic,
            teleop_record_topic,
            action_topic,
        )
        self.stamped = stamped

    def action_to_array(self, action: Union[gm.TwistStamped, gm.Twist]):
        """
        Convert an action to an array.
        """
        if self.stamped:
            action = action.twist
        return np.array(
            [action.linear.x, action.angular.z], dtype=np.float32
        )

    def array_to_action(self, array: np.ndarray):
        """
        Convert an array to an action.
        """
        msg = gm.TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = array[0]
        msg.twist.angular.z = array[1]
        return msg if self.stamped else msg.twist
