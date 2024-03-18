import nav_msgs.msg as nm
import geometry_msgs.msg as gm
import ackermann_msgs.msg as am
import sensor_msgs.msg as sm
import vesc_msgs.msg as vm
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import tf2_geometry_msgs as tf2gm
import tf
import tf.transformations as tft
import rospy


class ImageConverter:
    """
    Convert a sensor_msgs/Image message to a numpy array.
    """

    def __init__(self, width=None, height=None, name="pixels"):
        self.width = width
        self.height = height
        self.name = name

    def __call__(self, msg: sm.Image):
        bridge = CvBridge()

        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.width is not None and self.height is not None:
            img = cv2.resize(
                img, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        return msg.header.stamp.to_nsec(), {self.name: img}


class CompressedImageConverter:
    """
    Convert a sensor_msgs/Image message to a numpy array.
    """

    def __init__(self, width=None, height=None):
        self.width = width
        self.height = height

    def __call__(self, msg: sm.CompressedImage):
        bridge = CvBridge()

        img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.width is not None and self.height is not None:
            img = cv2.resize(
                img, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        return msg.header.stamp.to_nsec(), {"pixels": img}


class ImuConverter:
    """
    Convert a sensor_msgs/Imu message to a numpy array.
    """

    def __init__(self, base_link_id: str = "base_link"):
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration.from_sec(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_link_id = base_link_id

    def __call__(self, msg: sm.Imu):
        tx = self.tf_buffer.lookup_transform(
            self.base_link_id, msg.header.frame_id, rospy.Time()
        )
        robot_linear_acceleration = tf2gm.do_transform_vector3(
            gm.Vector3Stamped(vector=msg.linear_acceleration), tx
        ).vector
        robot_angular_velocity = tf2gm.do_transform_vector3(
            gm.Vector3Stamped(vector=msg.angular_velocity), tx
        ).vector

        return msg.header.stamp.to_nsec(), {
            "accel": np.array(
                [
                    robot_linear_acceleration.x,
                    robot_linear_acceleration.y,
                    robot_linear_acceleration.z,
                ]
            ),
            "gyro": np.array(
                [
                    robot_angular_velocity.x,
                    robot_angular_velocity.y,
                    robot_angular_velocity.z,
                ]
            ),
        }


class AckermannConverter:
    """
    Convert a ackermann_msgs/AckermannDriveStamped message to a numpy array.
    """

    def __init__(self):
        pass

    def __call__(self, msg: am.AckermannDriveStamped):
        return msg.header.stamp.to_nsec(), {
            "action": np.array([msg.drive.speed, msg.drive.steering_angle]),
        }


class OdometryConverter:
    """
    Convert a nav_msgs/Odometry message to a numpy array.
    """

    def __init__(self, map_frame_id: str, odom_format_vel_relative: bool = True):
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration.from_sec(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_frame_id = map_frame_id
        self.odom_format_vel_relative = odom_format_vel_relative

    def __call__(self, msg: nm.Odometry):
        # Convert the pose to the map frame
        try:
            odom_pose: gm.PoseStamped = gm.PoseStamped(
                pose=msg.pose.pose, header=msg.header
            )
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame_id,
                source_frame=odom_pose.header.frame_id,
                time=rospy.Time(0),
            )
            pose = tf2gm.do_transform_pose(odom_pose, transform)
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            return None

        # Get the yaw angle from the quaternion
        yaw = tft.euler_from_quaternion(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ]
        )[-1]

        relative_linear_velocity = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
        )

        if not self.odom_format_vel_relative:
            # Construct rotation matrix from quaternion
            R = tft.quaternion_matrix(
                [
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ]
            )
            relative_linear_velocity = R[:3, :3].T @ relative_linear_velocity

        return msg.header.stamp.to_nsec(), {
            "position": np.array(
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            ),
            "orientation": np.array(
                [
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ]
            ),
            "relative_linear_velocity": relative_linear_velocity,
            "pose_2d": np.array([pose.pose.position.x, pose.pose.position.y, yaw]),
        }


class PointStampedConverter:
    """
    Convert a geometry_msgs/PointStamped message to a numpy array.
    """

    def __init__(self, map_frame_id: str, name: str = "position") -> None:
        self.map_frame_id = map_frame_id
        self.name = name

    def __call__(self, msg: gm.PointStamped):
        assert (
            msg.header.frame_id == self.map_frame_id
        ), "PointStampedConverter: frame_id does not match map_frame_id"

        return msg.header.stamp.to_nsec(), {
            self.name: np.array([msg.point.x, msg.point.y, msg.point.z]),
        }


class VescConverter:
    """
    Convert a vesc_msgs/VescStateStamped message to a numpy array.
    """

    def __init__(self):
        pass

    def __call__(self, msg: vm.VescStateStamped):
        return msg.header.stamp.to_nsec(), {
            "battery_voltage": np.array([msg.state.voltage_input]),
            "wheel_speed": np.array([msg.state.speed / 4200]),
        }
