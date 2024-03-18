import os
import time

import numpy as np
import jax
import chex

from gym import spaces
import wandb
from absl import app, flags
import io
from PIL import Image
from typing import Any, Dict
import collections
import json

import tqdm

import jax.numpy as jnp
from flax.core import frozen_dict
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManager,
    CheckpointManagerOptions,
)

from offroad_learning.utils.zmq_bridge import ZmqBridgeServer
from offroad_learning.utils.load_network import make_agent
from offroad_learning.utils.spaces import obs_to_space

from jaxrl5.agents import DrQLearner
from jaxrl5.data import MemoryEfficientReplayBuffer, ReplayBuffer

from functools import partial

flags.DEFINE_string("comment", "", "Comment for W&B")
flags.DEFINE_string(
    "expert_replay_buffer", "", "(Optional) Expert replay buffer pickle file."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("save_interval", 5000, "Number of steps between saving.")
flags.DEFINE_integer("log_interval", 100, "Logging interval.")
flags.DEFINE_integer("batch_size", 1024, "Mini batch size.")
flags.DEFINE_integer(
    "start_training", int(1000), "Number of training steps to start training."
)
flags.DEFINE_integer("replay_buffer_size", 50000, "Capacity of the replay buffer.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
flags.DEFINE_integer("utd_ratio", 8, "Updates per data point")
flags.DEFINE_integer("max_steps", 1000000, "Maximum number of steps to run.")

flags.DEFINE_integer(
    "zmq_control_port", 5555, "Network port for sending weights to robot"
)
flags.DEFINE_integer(
    "zmq_data_port", 5556, "Network port for receiving data from robot"
)
flags.DEFINE_integer(
    "zmq_weights_port", 5557, "Network port for receiving data from robot"
)
flags.DEFINE_integer(
    "zmq_stats_port", 5558, "Network port for receiving data from robot"
)

flags.DEFINE_string(
    "expert_rb_embeddings_key",
    None,
    "Key for the embeddings in the expert replay buffer",
)
flags.DEFINE_string(
    "expert_rb_pixels_key", None, "Key for the pixels in the expert replay buffer"
)
flags.DEFINE_boolean(
    "expert_rb_append_pose",
    False,
    "Append the pose to the expert replay buffer observations",
)

def make_action_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(2,))


def data_to_space(data):
    # Convert a data dict or array into a gym space
    if isinstance(data, dict):
        return spaces.Dict({k: data_to_space(v) for k, v in data.items()})
    elif isinstance(data, np.ndarray):
        return spaces.Box(low=-np.inf, high=np.inf, shape=data.shape)
    else:
        raise ValueError(f"Unknown data type f{type(data)}")


def filter_observation(observation, observation_structure):
    if isinstance(observation_structure, (set, list)):
        keys = set(observation_structure).union({"prev_action"})
        return {k: observation[k] for k in keys}
    if isinstance(observation_structure, str):
        if isinstance(observation, dict):
            return observation[observation_structure]
        return observation
    else:
        raise ValueError(
            f"Unknown observation structure type {type(observation_structure)}"
        )


def filter_batch(batch, observation_structure):
    return {
        "observations": filter_observation(
            batch["observations"], observation_structure
        ),
        "actions": batch["actions"],
        "next_observations": filter_observation(
            batch["next_observations"], observation_structure
        ),
        "rewards": batch["rewards"],
        "dones": batch["dones"],
        "masks": batch["masks"],
    }


class Trainer:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.agent: DrQLearner = None

        # Replay buffer is none until we get the first data back from the robot
        self.replay_buffer = None
        self.replay_buffer_iterator = None
        self.expert_replay_buffer = None
        self.expert_replay_buffer_iterator = None

        if args.expert_replay_buffer:
            import pickle
            with open(args.expert_replay_buffer, "rb") as f:
                self.expert_replay_buffer = pickle.load(f)
            self.expert_replay_buffer_iterator = self.expert_replay_buffer.get_iterator(
                sample_args={
                    "batch_size": self.args.batch_size,
                    "sample_futures": None,
                    "relabel": True,
                }
            )

        self.log_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{args.comment}"

        self.has_done_jit = set()

    def wandb_init(self, config):
        wandb.init(
            project=config.get("wandb_project_name", "offroad_ros_synced"),
            group=config.get("wandb_group_name", "drq"),
            config={
                **config,
                **self.args.flag_values_dict(),
            },
        )
        self.workdir = os.path.join("data", wandb.run.name)
        self.dataset_folder = os.path.join(self.workdir, "dataset")
        self.policy_folder = os.path.join(self.workdir, "policies")
        self.checkpoint_manager = CheckpointManager(
            directory=self.policy_folder,
            checkpointers=PyTreeCheckpointer(),
            options=CheckpointManagerOptions(
                save_interval_steps=self.args.save_interval
            ),
        )

    def prepare_batch(self, batch):
        batch = frozen_dict.freeze(batch)
        return batch

    def update(self):
        update_info = {}
        update_info_expert = {}

        if self.replay_buffer is None:
            return {}, {}

        def train_step(batch: Dict[str, Any], update_temperature: bool):
            params = (update_temperature,)
            if params not in self.has_done_jit:
                first_jit = True
                print(f"First time JITing with update_temperature={update_temperature}")
                t0 = time.time()
            else:
                first_jit = False

            def make_output_range(prev_action):
                return (
                    jnp.clip(
                        prev_action - self.daction_max,
                        -1,#self.action_low,
                        1,#self.action_high,
                    ),
                    jnp.clip(
                        prev_action + self.daction_max,
                        -1,#self.action_low,
                        1,#self.action_high,
                    ),
                )

            prev_action = np.clip(
                batch["observations"]["prev_action"], self.action_low, self.action_high
            )
            prev_action_next = np.clip(
                batch["next_observations"]["prev_action"],
                self.action_low,
                self.action_high,
            )

            output_range = make_output_range(prev_action)
            output_range_next = make_output_range(prev_action_next)

            self.agent, update_info = self.agent.update(
                self.prepare_batch(batch),
                self.args.utd_ratio,
                output_range=output_range,
                output_range_next=output_range_next,
                update_temperature=update_temperature,
            )

            if first_jit:
                print(f"Done JITing, took {time.time() - t0} seconds")
                self.has_done_jit.add(params)

            return update_info

        # Train the agent
        if len(self.replay_buffer) > self.args.batch_size:
            batch = next(self.replay_buffer_iterator)
            update_info = train_step(batch, update_temperature=True)
        if (
            self.expert_replay_buffer
            and len(self.expert_replay_buffer) > self.args.batch_size
        ):
            batch_expert = next(self.expert_replay_buffer_iterator)

            update_info_expert = train_step(batch_expert, update_temperature=False)

        return update_info, update_info_expert

    def on_step(self, num_env_steps):
        self.checkpoint_manager.save(step=num_env_steps, items=self.agent)

        if num_env_steps >= self.args.start_training:
            self.agent = self.agent.replace(
                target_entropy=self.agent.target_entropy - self.config.get("entropy_step", 0)
            )

        if self.args.save_buffer and num_env_steps % self.args.save_interval == 0:
            print("Saving dataset to: " + self.dataset_folder)
            filename = self.replay_buffer.save(self.dataset_folder, num_env_steps + 1)
            print("Saved dataset: " + filename)

    def main(self):
        num_env_steps = 0
        num_training_steps = 0

        last_data_time = time.time()
        last_weights_time = time.time()

        bridge = ZmqBridgeServer(
            control_port=self.args.zmq_control_port,
            data_port=self.args.zmq_data_port,
            weights_port=self.args.zmq_weights_port,
            stats_port=self.args.zmq_stats_port,
        )

        # Handshake with the robot
        print("Waiting for data from robot...")
        bridge.wait_for_handshake()
        config = bridge.config
        self.config = config

        # Set up the agent
        sample_data = bridge.wait_for_data_def()
        sample_observation = sample_data["observations"]

        num_stack = config["encoder"]["kwargs"]["num_stack"]
        last_pixels = collections.deque(maxlen=num_stack)
        for _ in range(num_stack):
            last_pixels.append(np.zeros_like(sample_observation["pixels"]))
        sample_observation["pixels"] = np.stack(last_pixels, axis=-1)

        if "pixels" in sample_observation:
            actor_sample_observation = sample_observation.copy()
            actor_sample_observation["pixels"] = np.zeros_like(
                sample_observation["pixels"]
            )
        else:
            actor_sample_observation = sample_observation["states"].copy()

        self.daction_max = np.asarray(config["action_space"]["daction_max"])
        self.action_low = -np.ones_like(np.asarray(config["action_space"]["low"]))
        self.action_high = np.ones_like(np.asarray(config["action_space"]["high"]))

        self.agent = make_agent(
            seed=self.args.seed,
            agent_cls=config["agent"]["agent_cls"],
            agent_kwargs=config["agent"]["agent_kwargs"],
            observation_space=obs_to_space(actor_sample_observation),
            action_space=spaces.Box(
                # low=self.action_low, high=self.action_high, dtype=np.float32
                low=-np.ones(2), high=np.ones(2), dtype=np.float32
            ),
        )

        # if FLAGS.load_policy is not None:
        #     self.checkpoint_manager.restore()

        if "pixels" not in self.agent.observation_keys():
            sample_observation.pop("pixels")

        if "pixels" in self.agent.observation_keys():
            self.replay_buffer = MemoryEfficientReplayBuffer(
                observation_space=obs_to_space(sample_observation),
                action_space=spaces.Box(
                    #low=self.action_low, high=self.action_high, dtype=np.float32
                    low=-np.ones(2), high=np.ones(2), dtype=np.float32
                ),
                capacity=self.args.replay_buffer_size,
                pixel_keys=("pixels",) if "pixels" in self.agent.observation_keys() else (),
            )
        else:
            self.replay_buffer = ReplayBuffer(
                observation_space=obs_to_space(sample_observation),
                action_space=spaces.Box(
                    #low=self.action_low, high=self.action_high, dtype=np.float32
                    low=-np.ones(2), high=np.ones(2), dtype=np.float32
                ),
                capacity=self.args.replay_buffer_size,
            )
        self.replay_buffer._relabel_fn = partial(
            filter_batch, observation_structure=self.agent.observation_keys()
        )
        self.replay_buffer_iterator = self.replay_buffer.get_iterator(
            sample_args={
                "batch_size": self.args.batch_size,
                "sample_futures": None,
                "relabel": True,
            }
        )

        last_data = None

        self.wandb_init(config)
        pbar_to_start_training = tqdm.tqdm(
            dynamic_ncols=True,
            total=self.args.start_training,
            desc=f"Samples before starting training",
        )
        pbar = tqdm.tqdm(dynamic_ncols=True)

        while True:
            bridge.tick()

            if time.time() - last_weights_time > 1.0 and self.agent is not None:
                weights = {
                    "actor": frozen_dict.unfreeze(self.agent.actor.params),
                }
                if hasattr(self.agent, "limits"):
                    weights["limits"] = frozen_dict.unfreeze(self.agent.limits.params)
                bridge.publish_weights(weights)
                last_weights_time = time.time()

            while len(bridge.data) > 0:
                new_data = bridge.data.popleft()

                # Reinflate new_data
                if "pixels" in new_data["observations"]:
                    if "pixels" not in self.agent.observation_keys():
                        new_data["observations"].pop("pixels")
                    else:
                        pixels: np.ndarray = new_data["observations"]["pixels"]
                        pixels_jpeg_bytes = io.BytesIO(pixels.tobytes())
                        pixels = np.asarray(Image.open(pixels_jpeg_bytes))
                        last_pixels.append(pixels)
                        pixels = np.stack(last_pixels, axis=-1)
                        new_data["observations"]["pixels"] = pixels

                if last_data is None:
                    last_data = new_data
                    continue

                if new_data["seq_idx"] == last_data["seq_idx"] + 1:
                    if not np.allclose(new_data["observations"]["prev_action"], last_data["actions"]):
                        print(f'Warning: actions are different, {new_data["observations"]["prev_action"]} vs {last_data["actions"]}')
                    self.replay_buffer.insert(
                        dict(
                            observations=last_data["observations"],
                            actions=last_data["actions"],
                            next_observations=new_data["observations"],
                            rewards=last_data["rewards"],
                            dones=new_data["dones"],
                            masks=new_data["masks"],
                        )
                    )
                    num_env_steps += 1
                    self.on_step(num_env_steps)
                    last_data_time = time.time()

                    if num_env_steps <= self.args.start_training:
                        pbar_to_start_training.update(1)

                    # The trajectory is done, so the next observation will be
                    # the beginning of a new trajectory
                    if new_data["dones"]:
                        last_data = None

                        if hasattr(self.replay_buffer, "_first"):
                            self.replay_buffer._first = True
                    else:
                        last_data = new_data
                else:
                    print(f"Missing data!")
                    # There's at least one missing datapoint, so this will be
                    # the beginning of a new trajectory
                    last_data = new_data

                    if hasattr(self.replay_buffer, "_first"):
                        self.replay_buffer._first = True

            if time.time() - last_data_time > 5.0:
                # If we haven't seen data in a while, don't continue to train
                print("No data received for 5 seconds, not training")
                time.sleep(1)
                continue

            # Train the agent
            if num_env_steps >= self.args.start_training:
                update_info, update_info_expert = self.update()
                if update_info == {} and update_info_expert == {}:
                    continue

                for k, v in update_info.items():
                    if np.any(np.isnan(v)):
                        print(f"Got NaN in {k}!")

                pbar.update(1)
                num_training_steps += 1

                if num_training_steps % self.args.log_interval == 0:
                    wandb_log = {
                        f"training/target_entropy": np.array(self.agent.target_entropy)
                        if self.agent
                        else 0,
                    }
                    for k, v in update_info.items():
                        wandb_log.update({f"training/{k}": np.array(v)})
                    for k, v in update_info_expert.items():
                        wandb_log.update({f"training/expert/{k}": np.array(v)})
                    wandb_log["environment_steps"] = num_env_steps

                    for k, v in bridge.stats["logging"].items():
                        if hasattr(v, "__iter__"):
                            v = list(v)
                            if len(v) == 0:
                                continue
                            wandb_log[f"stats/best_{k}"] = min(v) if "speed" not in k else max(v)
                            wandb_log[f"stats/average_{k}"] = np.mean(v)
                            wandb_log[f"stats/last_{k}"] = v[-1]
                        else:
                            wandb_log[f"stats/{k}"] = v

                    stats_file = os.path.join(self.workdir, "stats.json")
                    with open(stats_file, "w") as f:
                        json.dump(bridge.stats, f)
                    wandb.save(stats_file)

                    wandb.log(wandb_log, step=num_training_steps)
                    for k, v in bridge.stats["summary"].items():
                        wandb.summary[f"{k}"] = v


def main(_):
    args = flags.FLAGS
    trainer = Trainer(args)
    trainer.main()


if __name__ == "__main__":
    app.run(main)
