import rospy
from offroad_learning.aggregator.generic_aggregator import Aggregator
import tqdm
import time
import os

import sensor_msgs.msg as sm
import nav_msgs.msg as nm
import geometry_msgs.msg as gm
import ackermann_msgs.msg as am
import cv_bridge
import numpy as np


def make_publishers():
    image_publisher = rospy.Publisher("/dummy/camera/image_raw", sm.Image, queue_size=1)
    bridge = cv_bridge.CvBridge()
    i = 0

    def publish_image(_):
        nonlocal i
        try:
            header = rospy.Header(stamp=rospy.Time.now())
            image_publisher.publish(
                bridge.cv2_to_imgmsg(
                    np.ones((100, 100, 3), dtype=np.uint8) * (i % 200), header=header
                )
            )
        except rospy.ROSException:
            pass

    rospy.Timer(rospy.Duration.from_sec(1 / 10), publish_image)

    imu_publisher = rospy.Publisher("/dummy/imu/data", sm.Imu, queue_size=1)

    def publish_imu(_):
        nonlocal i
        try:
            imu_publisher.publish(
                sm.Imu(
                    header=rospy.Header(stamp=rospy.Time.now()),
                    orientation=gm.Quaternion(w=1),
                    angular_velocity=gm.Vector3(x=i),
                    linear_acceleration=gm.Vector3(z=1),
                )
            )
        except rospy.ROSException:
            pass
        i += 1

    rospy.Timer(rospy.Duration.from_sec(1 / 100), publish_imu)

    odom_publisher = rospy.Publisher("/dummy/odom", nm.Odometry, queue_size=1)

    def publish_odom(_):
        nonlocal i
        try:
            odom_publisher.publish(
                nm.Odometry(
                    header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"),
                    child_frame_id="base_link",
                    pose=gm.PoseWithCovariance(
                        pose=gm.Pose(
                            position=gm.Point(x=i),
                            orientation=gm.Quaternion(w=1),
                        ),
                    ),
                    twist=gm.TwistWithCovariance(
                        twist=gm.Twist(
                            linear=gm.Vector3(x=1),
                            angular=gm.Vector3(z=1),
                        ),
                    ),
                )
            )
        except rospy.ROSException:
            pass

    rospy.Timer(rospy.Duration.from_sec(1 / 40), publish_odom)

    action_publisher = rospy.Publisher(
        "/dummy/vesc/low_level/ackermann_cmd_mux/output",
        am.AckermannDriveStamped,
        queue_size=1,
    )

    def publish_action(_):
        try:
            action_publisher.publish(
                am.AckermannDriveStamped(
                    header=rospy.Header(stamp=rospy.Time.now()),
                    drive=am.AckermannDrive(
                        steering_angle=1,
                        speed=1,
                    ),
                )
            )
        except rospy.ROSException:
            pass

    rospy.Timer(rospy.Duration.from_sec(1 / 100), publish_action)


def main():
    rospy.init_node("aggregator")

    aggregator = Aggregator(
        config_file=os.path.join(os.path.dirname(__file__), "topic_config.yaml")
    )

    make_publishers()
    aggregator.setup_ros()

    timer_called = False
    failed = False

    def timer_callback(_):
        nonlocal timer_called, failed
        data = aggregator.get_nearest_synced(
            query_key="pixels",
            query_stamp=rospy.Time.now() - rospy.Duration.from_sec(0.5),
            tolerance=rospy.Duration.from_sec(0.05),
        )
        if data is None:
            print("Stale")
            return

        for k, v in data.items():
            if v is None:
                print(f"Empty {k}")
                return

        timer_called = True

        image_i = data["pixels"].flatten()[0]
        gyro_i = data["gyro"].flatten()[0]
        odom_i = data["position"].flatten()[0]

        if abs(image_i - gyro_i % 200) >= 2:
            failed = True
            raise AssertionError("image_i: {}, gyro_i: {}".format(image_i, gyro_i))

        image_deviation = image_i - gyro_i % 200
        if abs(image_deviation) > 0:
            print("Deviation (gyro to image)", image_deviation)
        if abs(image_deviation) >= 4:
            failed = True
            raise AssertionError("image_i: {}, odom_i: {}".format(image_i, odom_i))

    rospy.Timer(rospy.Duration.from_sec(1 / 3), timer_callback)

    for _ in tqdm.trange(300):
        if failed or rospy.is_shutdown():
            break

        try:
            rospy.rostime.wallsleep(1 / 3)
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            failed = True

    rospy.signal_shutdown("Done")

    assert timer_called
    assert not failed


if __name__ == "__main__":
    main()
