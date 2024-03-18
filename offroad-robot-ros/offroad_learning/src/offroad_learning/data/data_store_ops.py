import numpy as np
import jax.numpy as jnp
import jax
import chex

from typing import Dict, Tuple, Optional

from jax.experimental import checkify


def expand_to_shape(x: jax.Array, shape: Tuple[int, ...]) -> jax.Array:
    """
    Expand an array with correct prefix dimensions to a given shape.
    """
    assert x.ndim <= len(shape)
    assert shape[: x.ndim] == x.shape, f"Bad shape {shape} for {x}"

    while x.ndim < len(shape):
        x = x[..., None].repeat(shape[x.ndim], axis=-1)

    return x


def _sample_element(
    name: str,
    dataset: Dict[str, jax.Array],
    config: dict,
    sampled_idx: jax.Array,
    ep_begin: jax.Array,
    ep_end: jax.Array,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """
    Sample from the data according to the config.

    Returns:
        data: the sampled data of shape (batch_size, ...) if a single element is selected,
              or (batch_size, time, ...) if a sequence is selected (e.g. history)
        mask: a boolean mask of the sampled data with shape (batch_size, ...) or (batch_size, time, ...)
    """
    device = jax.devices("cpu")[0]
    batch_size = sampled_idx.shape[0]
    source_name = config.get("source", name)
    if source_name not in dataset:
        raise ValueError(
            f"Bad sampling config: {config} has source {source_name} which is not in the dataset"
        )
    dataset_size = dataset[source_name].shape[0]

    def access(idx: jax.Array):
        mask_ep_begin = expand_to_shape(ep_begin, idx.shape)
        mask_ep_end = expand_to_shape(ep_end, idx.shape)

        data = dataset[source_name][
            jnp.clip(idx, mask_ep_begin, mask_ep_end) % dataset_size
        ]

        return data, (idx >= mask_ep_begin) & (idx < mask_ep_end)

    if "strategy" not in config:
        raise ValueError(f"Bad sampling config: {config}")

    strategy = config["strategy"]

    # If we're sampling the latest element, just return it
    if strategy == "latest":
        return access(sampled_idx)

    # If we're sampling a history, return the last N elements including the current
    if strategy == "sequence":
        history_strategy = config["sequence"]

        squeeze = history_strategy.get("squeeze", False)
        sequence_begin = history_strategy.get("begin", 0)
        sequence_end = history_strategy.get("end", 1)
        sequence_len = sequence_end - sequence_begin

        assert sequence_len > 0, f"History length must be positive, got {sequence_len}"
        with jax.default_device(device):
            indices = (
                jnp.arange(sequence_begin, sequence_end)[None, :] + sampled_idx[:, None]
            )
        chex.assert_shape(indices, (batch_size, sequence_len))

        if squeeze:
            assert (
                sequence_len == 1
            ), f"Can only squeeze sequence if length is 1, but got {sequence_len}"
            indices = jnp.squeeze(indices, axis=-1)

        return access(indices)

    # If we're sampling a future element, sample a single element from the future according to the distribution
    if strategy == "future":
        future_strategy = config["future"]

        squeeze = future_strategy.get("squeeze", False)

        distribution = future_strategy["distribution"]
        if distribution == "uniform":
            # Set max_future to sample from [t, t + max_future)
            max_future_length = future_strategy.get("max_future", None)

            # If max_future is None, set it to the end of the episode. Otherwise, clip to the end of the episode.
            if max_future_length is None:
                max_future = ep_end
            else:
                max_future = jnp.minimum(ep_end, sampled_idx + max_future_length)

            future_indices = jax.random.randint(
                key,
                shape=sampled_idx.shape,
                minval=sampled_idx,
                maxval=max_future,
            )
        elif distribution == "exponential":
            lambda_ = future_strategy["lambda"]
            offset = jax.random.exponential(key, shape=sampled_idx.shape) * lambda_
            future_indices = offset.astype(int) + sampled_idx
            future_indices = jnp.minimum(future_indices, ep_end - 1)
        else:
            raise ValueError(f"Unknown distribution {distribution}")

        chex.assert_shape(future_indices, sampled_idx.shape)
        return access(future_indices)

    raise NotImplementedError(f"Unknown strategy {config['strategy']}")


def make_jit_sample(sample_config: dict, device: jax.Device, sample_range: Tuple[int, int]):
    """
    Make a JIT-compiled sample function for a dataset, according to the config.
    """

    def _sample_impl(
        dataset: jax.Array,
        metadata: Dict[str, jax.Array],
        rng: jax.random.KeyArray,
        batch_size: int,
        sample_begin_idx: int,
        sample_end_idx: int,
        sampled_idcs: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        indices_key, *sampling_keys = jax.random.split(
            rng, len(sample_config.keys()) + 1
        )
        sampling_keys = {k: v for k, v in zip(sample_config.keys(), sampling_keys)}

        if sampled_idcs is None:
            sampled_idcs = jax.random.randint(
                indices_key,
                shape=(batch_size,),
                minval=sample_begin_idx,
                maxval=sample_end_idx,
                dtype=jnp.int32,
            )

        ep_begins = jnp.maximum(metadata["ep_begin"][sampled_idcs], sample_begin_idx)
        ep_ends = jnp.minimum(metadata["ep_end"][sampled_idcs], sample_end_idx)
        sampled_idcs = jnp.clip(
            sampled_idcs, ep_begins - sample_range[0], ep_ends - sample_range[1]
        )

        result = {
            k: _sample_element(
                name=k,
                dataset=dataset,
                config=v,
                sampled_idx=sampled_idcs,
                ep_begin=ep_begins,
                ep_end=ep_ends,
                key=sampling_keys[k],
            )
            for k, v in sample_config.items()
        }

        sampled_data = {k: v[0] for k, v in result.items()}
        samples_valid = {k: v[1] for k, v in result.items()}

        return sampled_data, samples_valid

    return jax.jit(_sample_impl, static_argnames=("batch_size",), device=device)


def make_jit_insert(device: jax.Device):
    """
    Make a JIT-compiled insert function for a dataset.
    """

    def _insert_impl(dataset: jax.Array, data: jax.Array, insert_idx: int):
        dataset = dataset.at[insert_idx].set(data)
        return dataset

    def _insert_tree_impl(
        dataset: Dict[str, jax.Array], data: Dict[str, jax.Array], insert_idx: int
    ):
        # Check should never run after JIT
        # chex.assert_trees_all_equal_shapes_and_dtypes(
        #     data, jax.tree_map(lambda x: x[0], dataset)
        # )

        return jax.tree_map(lambda x, y: _insert_impl(x, y, insert_idx), dataset, data)

    return jax.jit(_insert_tree_impl, donate_argnums=(0,), device=device)
