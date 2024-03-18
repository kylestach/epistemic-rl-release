from offroad_learning.utils.zmq_bridge import ZmqBridgeClient
import numpy as np
import tqdm
import jax
import time


def main():
    control_port = 5555
    data_port = 5556

    trainer_ip = "orwell.bair.berkeley.edu"

    client = ZmqBridgeClient(control_port, data_port, trainer_ip)

    result = client.handshake_config({"foo": "bar"})
    assert result == {"foo": "bar"}, result

    data = {"foo": np.array([1, 2, 3]), "bar": np.array([4, 5, 6])}

    for i in tqdm.trange(100, dynamic_ncols=True):
        data["seq_idx"] = i
        client.send_data(data)

        if i % 10 == 0:
            client.send_stats({"foo": 1, "bar": 2})

        weights = client.receive_weights()
        if weights is not None:
            print("Got weights!")
            print(jax.tree_map(lambda x: x.shape, weights))

        time.sleep(0.1)


if __name__ == "__main__":
    main()
