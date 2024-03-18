from offroad_learning.utils.zmq_bridge import ZmqBridgeServer
import numpy as np
import tqdm
import jax
import time


def main():
    control_port = 5555
    data_port = 5556

    server = ZmqBridgeServer(control_port, data_port)
    server.wait_for_handshake()

    assert server.config == {"foo": "bar"}, server.config

    data = {"foo": np.array([1, 2, 3]), "bar": np.array([4, 5, 6])}

    weights = {
        "foo": np.array([1, 2, 3]),
        "bar": np.array([4, 5, 6]),
        "baz": np.array([7, 8, 9]),
    }

    last_send_time = time.time()

    last_seq_idx = -1

    while True:
        server.tick()

        while len(server.data):
            new_data = server.data.pop()
            if new_data["seq_idx"] != last_seq_idx + 1:
                print(f"Missing data!")
            last_seq_idx = new_data["seq_idx"]
            print(jax.tree_map(lambda x: np.shape(x), new_data))

        if time.time() - last_send_time > 2.0:
            last_send_time = time.time()
            server.publish_weights(weights)
            print(server.stats)


if __name__ == "__main__":
    main()
