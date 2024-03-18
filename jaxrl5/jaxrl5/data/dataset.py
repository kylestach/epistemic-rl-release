from functools import partial
from random import sample
from typing import Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from gym.utils import seeding

from jaxrl5.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray, length: Optional[int] = None
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        if length is None:
            return dataset_dict[indx]
        else:
            # Get a sample from dataset_dict[k] from indx:indx+length, where indx is an array of indices
            indices = indx[None] + np.arange(length)[:, None]
            return dataset_dict[indices]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx, length)
    else:
        raise TypeError("Unsupported type.")
    return batch


def _compute_episode_lengths_and_indices(dones: np.ndarray) -> np.ndarray:
    episode_lengths = np.zeros(len(dones), dtype=np.int32)
    episode_timesteps = np.zeros(len(dones), dtype=np.int32)
    last_ep_start = 0
    for i in range(len(dones)):
        if dones[i] or i == len(dones) - 1:
            ep_len = i + 1 - last_ep_start
            episode_lengths[last_ep_start : i + 1] = ep_len
            episode_timesteps[last_ep_start : i + 1] = np.arange(ep_len)
            last_ep_start = i + 1
    return episode_lengths, episode_timesteps


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None, episode_lengths_and_timesteps: Optional[Union[Dict[str, np.ndarray], str]] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        if isinstance(episode_lengths_and_timesteps, dict):
            self.episode_lengths = episode_lengths_and_timesteps['episode_lengths']
            self.timesteps = episode_lengths_and_timesteps['episode_timesteps']
        elif episode_lengths_and_timesteps == 'auto':
            self.episode_lengths, self.episode_timesteps = _compute_episode_lengths_and_indices(dataset_dict['dones'])
        elif episode_lengths_and_timesteps == 'empty':
            self.episode_lengths = np.zeros(self.dataset_len, dtype=np.int32)
            self.episode_timesteps = np.zeros(self.dataset_len, dtype=np.int32)
        else:
            self.episode_lengths = None
            self.episode_timesteps = None

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        length: int = None,
    ) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx, length)
            elif length is None:
                batch[k] = self.dataset_dict[k][indx]
            else:
                # Get a sample from dataset_dict[k] from indx:indx+length, where indx is an array of indices
                indices = indx[None] + np.arange(length)[:, None]
                batch[k] = self.dataset_dict[k][indices]

        return frozen_dict.freeze(batch)

    def sample_jax(self, batch_size: int, keys: Optional[Iterable[str]] = None):
        if not hasattr(self, "rng"):
            self.rng = jax.random.PRNGKey(self._seed or 42)

            if keys is None:
                keys = self.dataset_dict.keys()

            jax_dataset_dict = {k: self.dataset_dict[k] for k in keys}
            jax_dataset_dict = jax.device_put(jax_dataset_dict)

            @jax.jit
            def _sample_jax(rng):
                key, rng = jax.random.split(rng)
                indx = jax.random.randint(
                    key, (batch_size,), minval=0, maxval=len(self)
                )
                return rng, jax.tree_map(
                    lambda d: jnp.take(d, indx, axis=0), jax_dataset_dict
                )

            self._sample_jax = _sample_jax

        self.rng, sample = self._sample_jax(self.rng)
        return sample

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        assert 0 < ratio and ratio < 1
        train_index = np.index_exp[: int(self.dataset_len * ratio)]
        test_index = np.index_exp[int(self.dataset_len * ratio) :]

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(
        self, take_top: Optional[float] = None, threshold: Optional[float] = None
    ):
        assert (take_top is None and threshold is not None) or (
            take_top is not None and threshold is None
        )

        (
            episode_starts,
            episode_ends,
            episode_returns,
        ) = self._trajectory_boundaries_and_returns()

        if take_top is not None:
            threshold = np.percentile(episode_returns, 100 - take_top)

        bool_indx = np.full((len(self),), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)

        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(
            episode_returns
        )
        self.dataset_dict["rewards"] *= scaling