import collections
from itertools import cycle
import rospy
import tf2_ros
import numpy as np
import yaml
import os
from PIL import Image
import cv_bridge

import rospkg

rospack = rospkg.RosPack()

from geometry_msgs.msg import (
    Point,
    PointStamped,
    Transform,
    Vector3,
    Quaternion,
)
import sensor_msgs.msg as sm
import std_msgs.msg as stdm


class GoalGraph:
    def __init__(self, fixed_frame_id: str):
        self.fixed_frame_id = fixed_frame_id

        # Load the goal YAML file from the ROS package share directory
        graph_name = rospy.get_param("~goal_graph_name", "bww8")
        goals_dir = os.path.join(
            rospack.get_path("offroad_learning"), "config", "goals", graph_name
        )
        filename = os.path.join(goals_dir, "goals.yaml")

        with open(filename, "r") as f:
            self.goal_yaml = yaml.load(f, Loader=yaml.FullLoader)

        self.goal_graph_name = self.goal_yaml["name"]

        def load_goal(g):
            result = {"position": g["position"], "image": None}
            if "image" in g:
                result["image"] = Image.open(
                    os.path.join(goals_dir, g["image"])
                ).convert("RGB")
            return result

        goal_list = [load_goal(g) for g in self.goal_yaml["goals"]]
        self.goal_iter = cycle(enumerate(goal_list))

        self.goal_threshold = self.goal_yaml.get("threshold", 1.0)

        self.next_goal_idx, self.next_goal = next(self.goal_iter)

    def setup_ros(self):
        """
        Deferred setup of ROS publishers and subscribers
        """
        self.next_goal_pub = rospy.Publisher(
            "/offroad/goal_point", PointStamped, queue_size=1
        )

        if self.next_goal["image"] is not None:
            self.goal_image_pub = rospy.Publisher(
                "/offroad/goal_image", sm.Image, queue_size=1
            )
            raise AssertionError(f"Goal image not supported, got {self.next_goal}")
        else:
            self.goal_image_pub = None

        # Static transform publisher for the UTM offset
        if "utm_zero" in self.goal_yaml.keys():
            # assert not rospy.get_param(
            #     "~use_realsense"
            # ), "Cannot use UTM offset with realsense (needs GPS)"
            self.utm_offset = -np.array(self.goal_yaml["utm_zero"])
            self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
            self.tf_broadcaster.sendTransform(
                tf2_ros.TransformStamped(
                    header=rospy.Header(
                        stamp=rospy.Time.now(),
                        frame_id=self.fixed_frame_id,
                    ),
                    child_frame_id="utm",
                    transform=Transform(
                        translation=Vector3(*self.utm_offset, 0),
                        rotation=Quaternion(0, 0, 0, 1),
                    ),
                )
            )
        else:
            # assert rospy.get_param(
            #     "~use_realsense"
            # ), "Must provide UTM offset when using GPS"
            pass

    def publish(self):
        header = stdm.Header(
            stamp=rospy.Time.now(),
            frame_id=self.fixed_frame_id,
        )
        self.next_goal_pub.publish(
            PointStamped(
                header=header,
                point=Point(
                    x=self.next_goal["position"][0],
                    y=self.next_goal["position"][1],
                    z=0.0,
                ),
            )
        )

        if self.goal_image_pub is not None:
            self.goal_image_pub.publish(
                cv_bridge.CvBridge().cv2_to_imgmsg(
                    np.asarray(self.next_goal["image"]),
                    header=header,
                )
            )

    def tick(self, robot_position):
        assert robot_position.shape == (2,)

        if (
            np.linalg.norm(robot_position - self.next_goal["position"])
            < self.goal_threshold
        ):
            self.next_goal_idx, self.next_goal = next(self.goal_iter)

        self.publish()
