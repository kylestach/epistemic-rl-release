import rospy
import rospkg
import ackermann_msgs.msg as am
import geometry_msgs.msg as gm
import std_msgs.msg as stdm

from gym import spaces

import collections
import os
from typing import Any, Dict, Type
import yaml
from functools import partial

import numpy as np
from flax.core import frozen_dict
from PIL import Image

from offroad_learning.aggregator.generic_aggregator import Aggregator, TopicStatus
from offroad_learning.utils.load_network import make_agent
from offroad_learning.utils.spaces import obs_to_space
from offroad_learning.utils.zmq_bridge import ZmqBridgeClient

from .state_machine import InferenceStateMachine, State as StateMachineState
from .statistics import StatsTracker
from .task import Task
from .racecar_task import make_racecar_task
from .action_interface import ActionInterface, AckermannInterface, TwistInterface


def make_ros_agent():
    # Load config file
    rospack = rospkg.RosPack()
    config_dir = os.path.join(rospack.get_path("offroad_learning"), "config")
    config_name = rospy.get_param("~config", None)
    if config_name is not None:
        config_filename = os.path.join(config_dir, config_name)
        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Agree with the server on the config
    bridge = ZmqBridgeClient(
        control_port=rospy.get_param("~zmq_control_port", 5555),
        data_port=rospy.get_param("~zmq_data_port", 5556),
        weights_port=rospy.get_param("~zmq_weights_port", 5557),
        stats_port=rospy.get_param("~zmq_stats_port", 5558),
        trainer_ip=rospy.get_param("~trainer_ip", None),
    )
    config = bridge.handshake_config(config)

    task = make_racecar_task(config)

    topic_names = {
        "teleop_topic": config["action_space"]["teleop_topic"],
        "teleop_record_topic": config["action_space"]["teleop_record_topic"],
        "action_topic": config["action_space"]["action_topic"],
    }
    if config["action_space"]["type"] == "ackermann":
        action_interface = AckermannInterface(**topic_names)
    elif config["action_space"]["type"] == "twist":
        action_interface = TwistInterface(**topic_names)
    else:
        raise ValueError(f"Unknown action space type: {config['action_space']['type']}")

    return RosAgent(task, bridge, config, action_interface)


class RosAgent:
    """
    An agent that accepts and aggregates ROS messages from several different sources, and periodically performs inference and publishes a new command.
    """

    bridge: ZmqBridgeClient
    action_interface: ActionInterface
    state_machine: InferenceStateMachine
    stats_tracker: StatsTracker

    def __init__(
        self,
        task: Task,
        bridge: ZmqBridgeClient,
        config: Dict[str, Any],
        action_interface: ActionInterface,
        state_machine_class: Type = InferenceStateMachine,
        stats_tracker_class: Type = StatsTracker,
    ):
        # Agree with the server on the config
        self.bridge = bridge
        self.task = task
        self.action_interface = action_interface
        self.action_interface.set_teleop_callback(self.receive_teleop)
        self.config = config

        # Metadata for uploads
        self.camera_frame_counter = 0
        self.seq_idx = 0
        self.upload_times = collections.deque(maxlen=30)

        self.current_lap_collisions = 0
        self.lap_collisions = []

        self.fixed_frame_id = config.get("fixed_frame_id", "map")

        self.daction_max = config["action_space"]["daction_max"]
        self.action_low = np.array(config["action_space"]["low"])
        self.action_high = np.array(config["action_space"]["high"])
        self.zero_action = np.zeros_like(self.action_low)
        self.recovery_speed = config.get("recovery_speed", 0.5)
        self.recovery_max_steer = config.get("recovery_max_steer", 0.5)

        # Setup submodules
        self.state_machine = state_machine_class(config=config)
        self.stats_tracker = stats_tracker_class()

        # Set up the data aggregator
        self.aggregator = Aggregator(config=config)
        self.control_callback_trigger = config.get("control_callback_trigger", "pixels")
        self.control_callback_counter = 0

        # Tell the server the config for the agent
        actor_sample_observation = self.task.prepare_observations_for_actor(
            self.aggregator.zeros(),
        )
        self.agent = make_agent(
            seed=0,
            agent_cls=config["agent"]["agent_cls"],
            agent_kwargs=config["agent"]["agent_kwargs"],
            observation_space=obs_to_space(actor_sample_observation),
            action_space=spaces.Box(
                # low=self.action_low, high=self.action_high, dtype=np.float64
                low=-np.ones_like(self.action_low),
                high=np.ones_like(self.action_high),
                dtype=np.float32,
            ),
        )
        self.bridge.send_data_def(
            self.task.prepare_observations_for_server(
                self.aggregator.zeros(),
                compress=False,
            )
        )

    def control_callback_timer_shim(self, _):
        self.control_callback(self.aggregator.get_latest())

    def setup_ros(self):
        # Other ROS interfaces
        self.task.setup_ros()
        self.aggregator.setup_ros()
        self.state_machine.setup_ros()
        self.action_interface.setup_ros()

        import tensor_dict_msgs.msg as tdm

        self.debug_tensors_pub = rospy.Publisher(
            "/debug_tensors",
            tdm.TensorDict,
            queue_size=1,
        )

        # Callback
        if isinstance(self.control_callback_trigger, str):
            self.aggregator.add_callback(
                self.control_callback_trigger, self.control_callback
            )
            self.control_callback_rate = self.config.get("control_callback_rate", 3)
        else:
            self.control_callback_rate = 1
            self.control_callback_timer = rospy.Timer(
                rospy.Duration.from_sec(self.control_callback_trigger["period"]),
                self.control_callback_timer_shim,
            )

    def receive_teleop(self, action: np.ndarray, record: bool):
        """
        Handle a teleop command from the joystick, which tells us not to run the agent.
        """
        if record:
            self.state_machine.handle_teleop_record()
        else:
            self.state_machine.handle_teleop()

        self.last_teleop_action = (action - self.action_bias) / self.action_scale

    def set_params(self, params: Dict[str, Any]):
        """
        Set the parameters of the agent.

        Args:
            params: The parameters to set. Should be .init(...)["params"].
        """
        if self.agent is None:
            rospy.logwarn("Agent not initialized yet; not setting params")
            return

        new_actor = self.agent.actor.replace(params=frozen_dict.freeze(params["actor"]))
        replace_dict=dict(actor=new_actor)
        if "limits" in params:
            assert hasattr(self.agent, "limits")
            replace_dict["limits"] = self.agent.limits.replace(params=replace_dict)
        self.agent = self.agent.replace(**replace_dict)

    def get_stats(self):
        """
        Get the statistics for the agent.
        """
        return self.stats_tracker.get_stats()

    def send_data(self):
        """
        Get all of the data that has not yet been sent to the server, clearing it from the queue.
        """
        new_upload_times = []
        while self.upload_times:
            seq_idx, time = self.upload_times.popleft()

            status = self.aggregator.status(time)

            if status == TopicStatus.LOST:
                # Remove the data from the upload queue
                rospy.logwarn(
                    f"Lost data at seq_idx {seq_idx}, {(rospy.Time.now() - time).to_sec()}s ago: {self.aggregator.status_verbose(time)}"
                )
                continue

            if status == TopicStatus.NOT_YET:
                # Put the data back in the upload queue
                new_upload_times.append((seq_idx, time))
                continue

            data = self.aggregator.get_nearest_synced("action", time)
            data = self.task.prepare_observations_for_server(data)
            data["seq_idx"] = seq_idx

            if data is None:
                rospy.logwarn("Lost some data...")
                continue

            self.bridge.send_data(data)
            if self.debug_tensors_pub is not None:
                data_copy = data.copy()
                data_copy["observations"] = data_copy["observations"].copy()
                if "pixels" in data_copy["observations"]:
                    data_copy["observations"].pop("pixels")
                if "image_embeddings" in data_copy["observations"]:
                    data_copy["observations"].pop("image_embeddings")
                data_copy.pop("seq_idx")
                import tensor_dict_convert

                self.debug_tensors_pub.publish(
                    tensor_dict_convert.to_ros_msg(data_copy)
                )

        self.upload_times = collections.deque(new_upload_times)

    def send_latest_data(self, data):
        """
        Send the latest data to the server, without doing any synchronization
        """
        data = self.task.prepare_observations_for_server(data)
        data["seq_idx"] = self.seq_idx

        for k, v in data.items():
            if v is None:
                print(f"Missing {k}")
                return

        self.bridge.send_data(data)

        if self.debug_tensors_pub is not None:
            data_copy = data.copy()
            data_copy["observations"] = data_copy["observations"].copy()
            if "pixels" in data_copy["observations"]:
                data_copy["observations"].pop("pixels")
            if "image_embeddings" in data_copy["observations"]:
                data_copy["observations"].pop("image_embeddings")
            data_copy.pop("seq_idx")
            import tensor_dict_convert

            self.debug_tensors_pub.publish(tensor_dict_convert.to_ros_msg(data_copy))

    def run_actor(self, observations):
        """
        Run the actor to get an action.
        """
        if observations is None:
            rospy.logwarn("No observations yet; not running agent")
            return np.zeros_like(self.action_low)

        observations_actor = self.task.prepare_observations_for_actor(observations)

        if observations_actor is None:
            rospy.logwarn("No observations yet; not running actor")
            return np.zeros_like(self.action_low)

        if any(v is None for v in observations_actor.values()):
            rospy.logwarn(
                f"Missing observations { {k for k, v in observations_actor.items() if v is None} }; not running actor"
            )
            return np.zeros_like(self.action_low)

        prev_action = observations["prev_action"]
        if prev_action is None:
            prev_action = np.zeros_like(self.action_low)

        output_low = np.clip(
            prev_action - self.daction_max,
            -1,  # self.action_low,
            1,  # self.action_high,
        )
        output_high = np.clip(
            prev_action + self.daction_max,
            -1,  # self.action_low,
            1,  # self.action_high,
        )
        action, self.agent = self.agent.sample_actions(
            observations_actor,
            output_range=(output_low, output_high),
        )

        return action

    def run_recovery(self):
        """
        Compute an action for recovery mode
        """
        throttle = self.state_machine.recovery_direction * self.recovery_speed
        steer = self.state_machine.recovery_steer * self.recovery_max_steer
        return self.rescale_action_from_real(np.array([throttle, steer]))

    @property
    def action_scale(self):
        return (self.action_high - self.action_low) / 2

    @property
    def action_bias(self):
        return (self.action_high + self.action_low) / 2

    def control_callback(self, observations):
        """
        Callback for the data aggregator. Runs the agent and publishes the action.
        """
        # Camera happens at 30hz but we only want to run the agent at 10hz
        self.control_callback_counter += 1
        if self.control_callback_counter % self.control_callback_rate != 0:
            return

        task_observables = self.task.tick(observations)
        observations = {**observations, **task_observables}

        should_record = self.state_machine.tick_state()
        self.stats_tracker.update(
            task_observables["goal_idx"], self.state_machine.state, observations
        )

        if self.state_machine.state == StateMachineState.LEARNING:
            action = self.run_actor(observations)
            self.publish_action(self.rescale_action_to_real(action))
        elif self.state_machine.state == StateMachineState.TELEOP_RECORD:
            action = self.last_teleop_action
            prev_action = (
                observations["prev_action"]
                if "prev_action" in observations
                else np.zeros_like(self.action_low)
            )
            action = np.clip(
                action, prev_action - self.daction_max, prev_action + self.daction_max
            )
            action = np.clip(action, -1, 1)
            self.publish_action(self.rescale_action_to_real(action))
        elif self.state_machine.state == StateMachineState.RECOVERY:
            action = self.run_recovery()
            self.publish_action(self.rescale_action_to_real(action))
        elif self.state_machine.state == StateMachineState.TELEOP:
            action = self.last_teleop_action
        elif self.state_machine.state in [
            StateMachineState.INVERTED,
            StateMachineState.RECOVERY_PAUSE,
            StateMachineState.COLLISION,
            StateMachineState.STUCK,
        ]:
            action = self.rescale_action_from_real(np.zeros(2))
            self.publish_action(np.zeros(2))
        else:
            raise ValueError(f"Unknown state: {self.state_machine.state}")

        # Actions need to be valid for training/inference
        action = np.clip(action, -1, 1)

        now = rospy.Time.now()
        self.aggregator.insert(
            now,
            {
                k: v
                for k, v in {
                    **task_observables,
                    "action": action,
                    "prev_action": action,
                    "mode": int(self.state_machine.state),
                }.items()
                if v is not None
            },
        )

        self.state_machine.last_forward_action = self.rescale_action_to_real(action)[0]

        if should_record:
            self.upload_times.append((self.seq_idx, now))
            self.seq_idx += 1

        # Communicate with the trainer
        self.send_data()
        self.bridge.send_stats(self.stats_tracker.get_stats())
        new_weights = self.bridge.receive_weights()
        if new_weights is not None:
            self.set_params(new_weights)

    def rescale_action_to_real(self, action):
        """
        Rescale the action from the agent to the real action space.
        """
        return self.action_scale * action + self.action_bias

    def rescale_action_from_real(self, action):
        """
        Rescale the action from the real action space to the agent.
        """
        return (action - self.action_bias) / self.action_scale

    def publish_action(self, action):
        """
        Publish the action to the robot.
        """

        action = np.clip(action, self.action_low, self.action_high)
        self.action_interface.publish_action(action)
