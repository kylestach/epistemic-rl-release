import os
import time
import jax

import numpy as np

from gym import spaces
from absl import app, flags
import io
from PIL import Image
from typing import Any, Dict
import pickle

import tqdm

from offroad_learning.utils.zmq_bridge import ZmqBridgeServer
from offroad_learning.utils.spaces import obs_to_space
from jaxrl5.data import MemoryEfficientReplayBuffer, ReplayBuffer

flags.DEFINE_integer("num_transitions", 10000, "Save the replay buffer.")
flags.DEFINE_string("dataset_file", None, "Path to dataset file.")

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


class DataCollector:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.replay_buffer = None

    def main(self):
        num_env_steps = 0

        bridge = ZmqBridgeServer(
            control_port=self.args.zmq_control_port,
            data_port=self.args.zmq_data_port,
            weights_port=self.args.zmq_weights_port,
            stats_port=self.args.zmq_stats_port,
        )

        # Handshake with the robot
        print("Waiting for data from robot...")
        bridge.wait_for_handshake()

        sample_data = bridge.wait_for_data_def()
        sample_observation = sample_data["observations"]
        print("Got data from robot! Data shape", jax.tree_map(lambda x: x.shape, sample_data))
        self.replay_buffer = ReplayBuffer(
            observation_space=obs_to_space(sample_observation),
            action_space=obs_to_space(sample_data["actions"]),
            capacity=self.args.num_transitions+100,
        )

        pbar = tqdm.tqdm(total=self.args.num_transitions, dynamic_ncols=True, desc="Data collection")

        last_data = None

        while num_env_steps < self.args.num_transitions:
            bridge.tick()

            while len(bridge.data) > 0:
                new_data = bridge.data.popleft()

                # Reinflate new_data
                if "pixels" in new_data["observations"]:
                    pixels: np.ndarray = new_data["observations"]["pixels"]
                    pixels_jpeg_bytes = io.BytesIO(pixels.tobytes())
                    new_data["observations"]["pixels"] = np.asarray(
                        Image.open(pixels_jpeg_bytes)
                    )

                if last_data is None:
                    last_data = new_data
                    continue

                if new_data["seq_idx"] == last_data["seq_idx"] + 1:
                    self.replay_buffer.insert(
                        dict(
                            observations=last_data["observations"],
                            actions=last_data["actions"],
                            next_observations=new_data["observations"],
                            rewards=new_data["rewards"],
                            dones=new_data["dones"],
                            masks=new_data["masks"],
                        )
                    )
                    num_env_steps += 1
                    pbar.update()

                    # The trajectory is done, so the next observation will be
                    # the beginning of a new trajectory
                    if new_data["dones"] or new_data["masks"] == 0:
                        last_data = None
                    else:
                        last_data = new_data
                else:
                    print(f"Missing data!")
                    # There's at least one missing datapoint, so this will be
                    # the beginning of a new trajectory
                    last_data = new_data

                    if hasattr(self.replay_buffer, "_first"):
                        self.replay_buffer._first = True

    def save(self):
        with open(self.args.dataset_file, "wb") as f:
            print("Saving dataset to", self.args.dataset_file)
            pickle.dump(self.replay_buffer, f)

def main(_):
    args = flags.FLAGS
    trainer = DataCollector(args)
    try:
        trainer.main()
    except KeyboardInterrupt:
        trainer.save()


if __name__ == "__main__":
    app.run(main)
