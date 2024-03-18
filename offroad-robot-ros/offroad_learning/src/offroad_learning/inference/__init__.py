import rospy
import sys

from .inference_agent import make_ros_agent

import warnings

warnings.filterwarnings("ignore")


def main(_):
    rospy.init_node("inference")
    agent = make_ros_agent()

    agent.setup_ros()

    rospy.spin()
