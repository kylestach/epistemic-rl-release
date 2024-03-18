import numpy as np

import rospy
import rospkg
import std_msgs.msg as stdm

import os
import yaml
from typing import Callable

from offroad_learning.inference.task import Task, CompileStatesWrapper, SelectKeyWrapper
from offroad_learning.utils.zmq_bridge import ZmqBridgeClient
from offroad_learning.inference.inference_agent import RosAgent
from offroad_learning.inference.action_interface import ActionInterface

from typing import Any, Dict


class TestInferenceTask(Task):
    """
    A trivial task where the agent tries to match the goal action.
    """

    def __init__(self):
        self.goal_action = np.array([0.0], dtype=np.float32)

    def tick(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        self.goal_action = np.random.normal(loc=self.goal_action * 0.9, scale=0.1)
        return {
            "goal_action": self.goal_action,
        }

    def compute_reward(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
    ):
        reward = np.exp(-((action - observation["goal_action"]) ** 2) * 5)
        return reward

    def is_truncated(self, observation: Dict[str, Any]):
        return False

    def is_terminated(self, observation: Dict[str, Any]):
        return np.abs(observation["prev_action"]) > 1.0


class TestActionInterface(ActionInterface):
    def __init__(
        self,
        teleop_topic: str,
        teleop_record_topic: str,
        action_topic: str,
    ):
        super().__init__(
            stdm.Float32,
            teleop_topic,
            teleop_record_topic,
            action_topic,
        )

    def action_to_array(self, action: stdm.Float32):
        return np.array([action.data], dtype=np.float32)

    def array_to_action(self, array: np.ndarray):
        return stdm.Float32(data=array[0])


def make_task(config: Dict[str, Any]):
    task = TestInferenceTask()
    task = CompileStatesWrapper(task, config["agent"]["state_keys"])
    task = SelectKeyWrapper(task, "states")
    return task


def inference_node():
    rospy.init_node("inference_node")

    # Load config file
    config_filename = os.path.join(
        os.path.dirname(__file__), "test_inference_config.yaml"
    )
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    # Agree with the server on the config
    bridge = ZmqBridgeClient(
        control_port=rospy.get_param("~zmq_control_port", 5555),
        data_port=rospy.get_param("~zmq_data_port", 5556),
        weights_port=rospy.get_param("~zmq_weights_port", 5557),
        stats_port=rospy.get_param("~zmq_stats_port", 5558),
        trainer_ip=rospy.get_param("~trainer_ip", None),
    )
    config = bridge.handshake_config(config)

    task = make_task(config)
    action_interface = TestActionInterface(
        teleop_topic=rospy.get_param("~teleop_topic", "/teleop"),
        teleop_record_topic=rospy.get_param("~teleop_record_topic", "/teleop_record"),
        action_topic=rospy.get_param("~action_topic", "/action"),
    )

    agent = RosAgent(task, bridge, config, action_interface)
    agent.setup_ros()

    rospy.spin()


if __name__ == "__main__":
    inference_node()
