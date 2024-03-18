import rospy
import tf2_ros
import numpy as np

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_srvs.srv import Empty, EmptyResponse
import sensor_msgs.msg as sm
from cv_bridge import CvBridge
import cv2
import os


class RosGoalGraph:
    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.bridge = CvBridge()

        self.fixed_frame_id = rospy.get_param("~fixed_frame")

        self.goal_loop = []
        self.capture_goal = rospy.Service(
            "/offroad/capture_goal", Empty, self.capture_goal_callback
        )
        self.goal_pub = rospy.Publisher(
            "/offroad/recorded_goal", PoseArray, queue_size=1
        )
        self.joy_sub = rospy.Subscriber("/vesc/joy", sm.Joy, self.joy_callback)
        self.image_sub = rospy.Subscriber(
            rospy.get_param("~image_topic", "/camera/image_raw"),
            sm.Image,
            self.image_callback,
        )
        self.button_was_pressed = False
        self.i = 0
        self.goal_dir = rospy.get_param("~goal_dir", "/tmp/goal_tmp")
        os.makedirs(self.goal_dir, exist_ok=True)

    def joy_callback(self, joy):
        if joy.buttons[1] == 1 and not self.button_was_pressed:
            self.capture_goal_callback(None)
            rospy.logwarn(f"Goals: {self.goal_loop}")
        self.button_was_pressed = joy.buttons[1] == 1

    def image_callback(self, image: sm.Image):
        self.image = self.bridge.imgmsg_to_cv2(image, desired_encoding="rgb8")

    def capture_goal_callback(self, _):
        # Wait for the map to become ready
        try:
            tx = self.buffer.lookup_transform(
                self.fixed_frame_id, "base_link", rospy.Time.now(), rospy.Duration(1.0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Cannot capture goal; transform failed due to {e}")
            return

        position = np.array([tx.transform.translation.x, tx.transform.translation.y])
        self.goal_loop.append(position)

        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = rospy.Time.now()
        for goal in self.goal_loop:
            pose = Pose()
            pose.position.x = goal[0]
            pose.position.y = goal[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.goal_pub.publish(pose_array)

        cv2.imwrite(os.path.join(self.goal_dir, f"goal_{self.i}.png"), self.image)

        self.i += 1

        # Dump the yaml file
        with open(os.path.join(self.goal_dir, "goals.yaml"), "w") as f:
            f.write("goals:\n")
            for i, goal in enumerate(self.goal_loop):
                f.write(f"  - position: [{goal[0]}, {goal[1]}]\n")
                f.write(f'  - image: "goal_{i}.png"\n')

        return EmptyResponse


def main():
    rospy.init_node("goal_graph_recorder")
    goal_graph = RosGoalGraph()
    rospy.spin()


if __name__ == "__main__":
    main()
