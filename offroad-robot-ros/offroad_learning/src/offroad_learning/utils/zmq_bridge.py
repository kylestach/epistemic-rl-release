import zmq
import warnings
import time
import collections

import numpy as np
import jax
import flax

import numpy as np
import jax
import flax

from typing import Any, Dict


class ZmqBridgeClient:
    def __init__(
        self,
        control_port: int,
        data_port: int,
        weights_port: int,
        stats_port: int,
        trainer_ip: str = None,
    ):
        if trainer_ip is None:
            raise ValueError("Must specify trainer_ip")

        self.zmq_context = zmq.Context()

        self.control_socket = self.zmq_context.socket(zmq.REQ)
        self.control_socket.connect(f"tcp://{trainer_ip}:{control_port}")

        self.data_socket = self.zmq_context.socket(zmq.PAIR)
        self.data_socket.connect(f"tcp://{trainer_ip}:{data_port}")

        self.weights_socket = self.zmq_context.socket(zmq.PAIR)
        self.weights_socket.connect(f"tcp://{trainer_ip}:{weights_port}")

        self.stats_socket = self.zmq_context.socket(zmq.PAIR)
        self.stats_socket.connect(f"tcp://{trainer_ip}:{stats_port}")

    def handshake_config(
        self,
        config: Dict[str, Any],
    ):
        """
        Agree on the config with the trainer.
        """
        while self.control_socket.poll(timeout=0) != 0:
            self.control_socket.recv()

        self.control_socket.send_json({"type": "config", "config": config})
        response_config = self.control_socket.recv_json()

        assert response_config["type"] == "config"
        assert response_config["status"] == "ok"

        return response_config["config"]

    def send_data_def(self, sample_data: Dict[str, Any]):
        while self.control_socket.poll(timeout=0) != 0:
            self.control_socket.recv()

        self.control_socket.send_json(
            {
                "type": "data_def",
                "obs_shape": jax.tree_map(np.shape, sample_data),
                "obs_dtype": jax.tree_map(lambda x: str(x.dtype), sample_data),
            }
        )
        response_config = self.control_socket.recv_json()

        assert response_config["type"] == "data_def"
        assert response_config["status"] == "ok"

    def send_data(self, data):
        data_serialized = flax.serialization.msgpack_serialize(data)
        self.data_socket.send(data_serialized)

    def send_stats(self, stats):
        self.stats_socket.send_json(stats)

    def receive_weights(self):
        weights = None
        while self.weights_socket.poll(timeout=0) != 0:
            weights_raw = self.weights_socket.recv()
            weights = flax.serialization.msgpack_restore(weights_raw)
        return weights


class ZmqBridgeServer:
    def __init__(
        self,
        control_port: int,
        data_port: int,
        weights_port: int,
        stats_port: int,
    ):
        self.zmq_context = zmq.Context()

        self.control_socket = self.zmq_context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{control_port}")

        self.data_socket = self.zmq_context.socket(zmq.PAIR)
        self.data_socket.bind(f"tcp://*:{data_port}")

        self.weights_socket = self.zmq_context.socket(zmq.PAIR)
        self.weights_socket.bind(f"tcp://*:{weights_port}")

        self.stats_socket = self.zmq_context.socket(zmq.PAIR)
        self.stats_socket.bind(f"tcp://*:{stats_port}")

        self.config = None
        self.weights = None
        self.sample_data = None
        self.stats = None
        self.data = collections.deque()

        self.poller = zmq.Poller()
        self.poller.register(self.control_socket, zmq.POLLIN)
        self.poller.register(self.data_socket, zmq.POLLIN)
        self.poller.register(self.stats_socket, zmq.POLLIN)

    def wait_for_handshake(self):
        while self.config is None:
            self.tick()
            time.sleep(0.1)
        return self.config

    def wait_for_data_def(self):
        while self.sample_data is None:
            self.tick()
            time.sleep(0.1)
        return self.sample_data

    def handle_control(self):
        request = self.control_socket.recv_json()
        if request["type"] == "config":
            self.config = request["config"]
            self.control_socket.send_json(
                {"type": "config", "config": self.config, "status": "ok"}
            )
        elif request["type"] == "data_def":
            self.sample_data = jax.tree_map(
                lambda shape, dtype: np.zeros(shape, dtype=dtype),
                request["obs_shape"],
                request["obs_dtype"],
                is_leaf=lambda x: isinstance(x, (list, str)),
            )
            self.control_socket.send_json({"type": "data_def", "status": "ok"})
        else:
            warnings.warn(f"Unknown request type {request['type']}")
            self.control_socket.send_json(
                {"type": "error", "status": "unknown request"}
            )

    def handle_stats(self):
        self.stats = self.stats_socket.recv_json()

    def handle_data(self):
        data_raw = self.data_socket.recv()
        self.data.append(flax.serialization.msgpack_restore(data_raw))

    def tick(self):
        while True:
            events = self.poller.poll(timeout=0)

            if len(events) == 0:
                break

            for socket, evt_data in events:
                assert evt_data == zmq.POLLIN

                if socket == self.control_socket:
                    self.handle_control()
                elif socket == self.data_socket:
                    self.handle_data()
                elif socket == self.stats_socket:
                    self.handle_stats()
                else:
                    warnings.warn(f"Unknown socket {socket}")

    def publish_weights(self, weights):
        weights_serialized = flax.serialization.msgpack_serialize(weights)
        self.weights_socket.send(weights_serialized)
