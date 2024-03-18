import numpy as np
import jax.numpy as jnp
import jax
import chex

from typing import Dict, Union, Optional, Tuple, List

from offroad_learning.data.data_store_ops import make_jit_insert, make_jit_sample


class DataStore:
    """
    An in-memory data store for storing and sampling data from a (typically trajectory) dataset.
    """

    def __init__(
        self,
        capacity: int,
        data_shapes: Dict[str, Union[tuple, Dict]],
        device: Optional[jax.Device] = None,
        seed: int = 0,
        min_trajectory_length: int = 2,
    ):
        """
        Args:
            capacity: the maximum number of data points that can be stored
            data_shapes: a dictionary mapping data names to shapes or to {"shape": ..., "dtype": ...}
            device: the device to store the data on. If None, defaults to the first CPU device.
            seed: the random seed to use for sampling
            min_trajectory_length: the minimum length of a trajectory to store
        """

        if device is None:
            device = jax.devices("cpu")[0]

        def _initialize_element(config: Union[tuple, dict]):
            if isinstance(config, tuple):
                return jnp.zeros((capacity, *config), dtype=jnp.float32)
            elif isinstance(config, dict):
                assert "shape" in config, f"Bad config: {config} has no shape"
                return jnp.zeros(
                    (capacity, *config["shape"]), dtype=config.get("dtype", jnp.float32)
                )

        # Do it on CPU
        with jax.default_device(device):
            self.dataset = {k: _initialize_element(v) for k, v in data_shapes.items()}
            self._sample_rng = jax.random.PRNGKey(seed=seed)

        self.metadata = {
            "ep_begin": np.zeros((capacity,), dtype=jnp.int32),
            "ep_end": np.full((capacity,), -1, dtype=jnp.int32),
            "trajectory_id": np.full((capacity,), -1, dtype=jnp.int32),
        }

        self.capacity = capacity
        self.size = 0

        self._current_trajectory_begin = 0
        self._current_trajectory_id = 0
        self._sample_begin_idx = 0
        self._sample_end_idx = 0
        self._insert_idx = 0

        self._device = device

        self._insert_impl = make_jit_insert(device)
        self._sample_impls = {}

        self._min_trajectory_length = min_trajectory_length

    def register_sample_config(
        self, name: str, config: dict, sample_range: Tuple[int, int] = (0, 1)
    ):
        assert (
            sample_range[1] - sample_range[0] > 0
        ), f"Sample range {sample_range} must be positive"
        assert (
            sample_range[1] - sample_range[0] <= self._min_trajectory_length
        ), f"Sample range {sample_range} must be <= the minimum trajectory length {self._min_trajectory_length}"
        self._sample_impls[name] = make_jit_sample(config, self._device, sample_range)

    def insert(self, data: Dict[str, jax.Array], end_of_trajectory: bool):
        """
        Insert a single data point into the data store.
        """

        # Grab the metadata of the sample we're overwriting
        real_insert_idx = self._insert_idx % self.capacity
        overwritten_ep_end = self.metadata["ep_end"][real_insert_idx]

        with jax.default_device(self._device):
            self.dataset = self._insert_impl(self.dataset, data, real_insert_idx)

        self.metadata["ep_begin"][real_insert_idx] = self._current_trajectory_begin
        self.metadata["ep_end"][real_insert_idx] = -1
        self.metadata["trajectory_id"][real_insert_idx] = self._current_trajectory_id

        self._insert_idx += 1
        self._sample_begin_idx = max(
            self._sample_begin_idx, self._insert_idx - self.capacity
        )

        # If there's not enough remaining of the overwritten trajectory to be valid, mark it as invalid
        if (
            overwritten_ep_end != -1
            and overwritten_ep_end - self._sample_begin_idx
            < self._min_trajectory_length
        ):
            self._sample_begin_idx = overwritten_ep_end

        # We don't want to sample from this trajectory until there's enough data to be valid
        if (
            self._insert_idx - self._current_trajectory_begin
            >= self._min_trajectory_length
        ):
            self._sample_end_idx = self._insert_idx

        self.size = min(self.size + 1, self.capacity)

        if end_of_trajectory:
            self.end_trajectory()

    def end_trajectory(self):
        """
        End a trajectory without inserting any data.
        """
        if (
            self._insert_idx - self._current_trajectory_begin
            < self._min_trajectory_length
        ):
            # If necessary, roll back the insert index to the beginning of the current trajectory if it's too short
            # We never set ep_end for this trajectory, so no need to rewrite it
            self._insert_idx = self._current_trajectory_begin
        else:
            # This trajectory is long enough. Mark it as valid.
            self.metadata["ep_end"][
                self._current_trajectory_begin : self._insert_idx
            ] = self._insert_idx

            # Update the metadata for the next trajectory
            self._current_trajectory_begin = self._insert_idx
            self._current_trajectory_id += 1

    def sample(
        self,
        sample_config_name: str,
        batch_size: int,
        force_indices: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        """
        Sample a batch of data from the data store.

        Args:
            sample_config_name: the name of the sample config to use
            batch_size: the batch size
            force_indices: if not None, force the sample to use these indices instead of sampling randomly.
                Assumed to be of shape (batch_size,) and smaller than the last valid index in the data store.
        """

        if len(self._sample_impls) == 0:
            raise ValueError("No sample configs registered")

        rng, key = jax.random.split(self._sample_rng)
        sample_impl = self._sample_impls[sample_config_name]
        sampled_data = sample_impl(
            dataset=self.dataset,
            metadata=self.metadata,
            rng=key,
            batch_size=batch_size,
            sample_begin_idx=self._sample_begin_idx,
            sample_end_idx=self._sample_end_idx,
            sampled_idcs=force_indices,
        )
        self._sample_rng = rng
        return sampled_data

    def save(self, path: str):
        dataset_dict = {
            f"data/{k}": np.asarray(v[: self.size]) for k, v in self.dataset.items()
        }
        metadata_dict = {
            f"metadata/{k}": np.asarray(v[: self.size])
            for k, v in self.metadata.items()
        }
        np.savez(
            path,
            **dataset_dict,
            **metadata_dict,
            size=self.size,
            capacity=self.capacity,
            _current_trajectory_begin=self._current_trajectory_begin,
            _insert_idx=self._insert_idx,
        )

    @classmethod
    def load(
        self,
        path: str,
        capacity: Optional[int] = None,
        device: Optional[jax.Device] = None,
    ):
        loaded_data = np.load(path)
        capacity = loaded_data.pop("capacity")
        size = loaded_data.pop("size")
        current_trajectory_begin = loaded_data.pop("_current_trajectory_begin")
        insert_idx = loaded_data.pop("_insert_idx")
        data = {
            "/".split(k)[1]: jax.v
            for k, v in loaded_data.items()
            if k.startswith("data/")
        }
        metadata = {
            "/".split(k)[1]: v
            for k, v in loaded_data.items()
            if k.startswith("metadata/")
        }

        data_shape = {
            k: {"shape": v.shape[1:], "dtype": str(v.dtype)} for k, v in data.items()
        }
        data_store = DataStore(capacity, data_shape, device=device)

        data_store.dataset = jax.tree_map(
            lambda dataset, data: dataset.at[:size].set(data), data_store.dataset, data
        )
        for k, v in metadata.items():
            data_store.metadata[k][:size] = v

        data_store.size = size
        data_store._current_trajectory_begin = current_trajectory_begin
        data_store._insert_idx = insert_idx

        return data_store

    def __len__(self):
        return len(self.trajectories)
