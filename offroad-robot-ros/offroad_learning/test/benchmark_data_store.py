import jax
import numpy as np
from offroad_learning.data.data_store import DataStore


def _make_random_data(data_config):
    def _make_elem(elem_config):
        if isinstance(elem_config, dict):
            return np.random.randint(
                0, 256, size=elem_config["shape"], dtype=np.uint8
            )
        elif isinstance(elem_config, (tuple, list)):
            return np.random.uniform(size=elem_config)
        raise ValueError(f"Unknown config type {type(elem_config)}")

    return {k: _make_elem(v) for k, v in data_config.items()}


def _add_trajectory(
    data_store: DataStore, data_config: dict, length: int, start_idx: int = 0
):
    for i in range(length):
        data = _make_random_data(data_config)
        if "index" in data:
            data["index"] = np.asarray(start_idx + i, dtype="int32")
        data_store.insert(data, end_of_trajectory=(i == length - 1))


def benchmark():
    # Benchmark
    import time

    data_config = {
        "image": {
            "shape": (64, 64, 3),
            "dtype": "uint8",
        },
    }
    sampling_config = {"image": {"strategy": "sequence", "sequence": {"begin": -2}}}

    data_store = DataStore(capacity=1000, data_shapes=data_config, device=jax.devices("cpu")[0])
    data_store.register_sample_config("rl", sampling_config)

    K = 1000

    start = time.time()
    _add_trajectory(data_store, data_config, K)
    end = time.time()
    print(f"Insert time for {K} samples: {end - start} seconds ({(end-start)/K} seconds per sample)")

    start = time.time()
    for _ in range(2):
        jax.block_until_ready(data_store.sample("rl", 1024))
    end = time.time()
    print(f"JIT time: {end - start} seconds")
    start = time.time()
    for _ in range(K):
        data, valid = jax.block_until_ready(data_store.sample("rl", 1024))
        # print(data["image"].device())
        # print(data["image"].shape, data["image"].min(), data["image"].max())
        # print(data["image"].sum())
    end = time.time()
    print(f"JAX: {K} samples took {end-start} seconds ({(end-start)/K} seconds per sample)")

    arr = np.random.randint(0, 256, size=(1000, 64, 64, 3), dtype=np.uint8)
    start = time.time()
    for _ in range(K):
        idx = np.random.randint(3, 1000, size=(1024,))
        idx = idx[:, None] + np.arange(-2, 1)[None, :]
        x = arr[idx % arr.shape[0]]
        # print(x.shape, x.min(), x.max())
        # print(x.sum().astype(np.int32))
    end = time.time()
    print(f"numpy: {K} samples took {end-start} seconds ({(end-start)/K} seconds per sample)")

if __name__ == "__main__":
    benchmark()