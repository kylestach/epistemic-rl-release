import numpy as np

from typing import Any, Dict, List

from offroad_learning.inference.task import (
    Task,
    Wrapper,
    ImageEmbeddingsWrapper,
    ImageWrapper,
    CompressionWrapper,
    CompileStatesWrapper,
)
from offroad_learning.inference.goal_graph import GoalGraph
from offroad_learning.inference.state_machine import State as StateMachineState


def compute_relative_goal(pose_2d: np.ndarray, goal_point: np.ndarray):
    """
    Compute the relative goal from the current pose to the goal point.
    """
    vector = goal_point[:2] - pose_2d[:2]
    rotation_matrix = np.array(
        [
            [np.cos(pose_2d[2]), -np.sin(pose_2d[2])],
            [np.sin(pose_2d[2]), np.cos(pose_2d[2])],
        ],
        dtype=np.float32,
    )

    vector = rotation_matrix.T @ vector

    distance = np.linalg.norm(vector)
    return np.concatenate([vector / distance + 1e-6, [distance]])


class RelativeGoalWrapper(Wrapper):
    def __init__(self, task: Task):
        super().__init__(task)

    def preprocess_observations(self, observations: Dict[str, Any], **kwargs):
        results = super().preprocess_observations(observations, **kwargs).copy()

        if observations["pose_2d"] is None:
            results["goal_relative"] = None
        else:
            results["goal_relative"] = compute_relative_goal(
                observations["pose_2d"], observations["goal_point"]
            )

        return results


class RacecarTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.goal_graph = GoalGraph(config.get("fixed_frame_id", "map"))

    def setup_ros(self):
        self.goal_graph.setup_ros()

    def tick(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        if observations is None or observations["pose_2d"] is None:
            return {}

        self.goal_graph.tick(observations["pose_2d"][:2])

        results = {
            "goal_point": self.goal_graph.next_goal["position"],
            "goal_idx": self.goal_graph.next_goal_idx,
        }
        if self.goal_graph.next_goal["image"] is not None:
            results["goal_pixels"] = self.goal_graph.next_goal["image"]
        return results

    def compute_reward(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
    ):
        linear_velocity = observation["relative_linear_velocity"][:2]
        goal_vector = observation["goal_relative"][:2]

        fail_penalty = 10.0 if observation["mode"] in [
            StateMachineState.COLLISION,
            StateMachineState.INVERTED,
            StateMachineState.STUCK,
        ] else 0.0

        return np.dot(linear_velocity, goal_vector)

    def is_truncated(self, observation: Dict[str, Any]):
        return StateMachineState(observation["mode"]) in [StateMachineState.TELEOP]

    def is_terminated(self, observation: Dict[str, Any]):
        return StateMachineState(observation["mode"]) in [
            StateMachineState.COLLISION,
            StateMachineState.RECOVERY,
            StateMachineState.STUCK,
            StateMachineState.INVERTED,
        ]


def make_racecar_task(config: Dict[str, Any]) -> Task:
    task = RacecarTask(config)
    task = RelativeGoalWrapper(task)
    task = CompileStatesWrapper(task, config["agent"]["state_keys"])

    if "encoder" in config:
        encoder_config = config["encoder"]
        encoder_type = encoder_config.get("type", None)
        if encoder_type is None:
            pass
        elif encoder_type == "pixels":
            task = ImageWrapper(
                task,
                pixels_regex=".*pixels.*",
                num_stack=encoder_config["kwargs"]["num_stack"],
            )
        else:
            task = ImageEmbeddingsWrapper(task, encoder_config)

    if "extra_encoders" in config:
        for encoder_config in config["extra_encoders"]:
            task = ImageEmbeddingsWrapper(task, encoder_config)

    task = CompressionWrapper(task, ".*pixels.*")

    return task
