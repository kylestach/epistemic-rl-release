import collections
from typing import Optional, Union, Iterable, Callable, Sequence

import gym
import gym.spaces
import jax
import numpy as np

from jaxrl5.data.dataset import Dataset, DatasetDict, _sample
from flax.core import frozen_dict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()

def _handle_boundary_condition(boundary_condition, offset_from_ep_begin, ep_len):
    if boundary_condition == 'wrap':
        return offset_from_ep_begin % ep_len
    elif boundary_condition == 'truncate':
        return np.clip(offset_from_ep_begin, 0, ep_len - 1)
    elif boundary_condition == None:
        return offset_from_ep_begin
    else:
        raise ValueError(f'Unknown boundary condition: {boundary_condition}')


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        relabel_fn: Optional[Callable[[DatasetDict], DatasetDict]] = None,
        extra_fields: Sequence[str] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        if extra_fields is not None:
            for key in extra_fields:
                dataset_dict[key] = np.empty((capacity,), dtype=np.float32)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

        self._relabel_fn = relabel_fn

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def calculate_future_observation_indices(self, base_indices: np.ndarray, sample_futures_config: dict):
        sampling_type = sample_futures_config.get('type', None)
        boundary_condition = sample_futures_config.get('boundary', None)

        observations = self.dataset_dict['observations']

        if 'index' in observations and 'ep_len' in observations:
            index_in_ep = _sample(self.dataset_dict['observations']['index'], base_indices)
            ep_len = _sample(self.dataset_dict['observations']['ep_len'], base_indices)
            ep_begin = base_indices - index_in_ep
            ep_end = ep_begin + ep_len
        else:
            index_in_ep = None
            ep_len = None
            ep_begin = 0
            ep_end = None

        if sampling_type == 'uniform':
            assert ep_begin is not None
            return np.random.randint(ep_begin, ep_end, base_indices.shape)
        elif sampling_type == 'exponential':
            inverse_gamma = sample_futures_config.get('mean', 100.0)
            future_offsets = np.ceil(np.random.exponential(inverse_gamma, base_indices.shape)).astype(np.int32)
            offsets_from_ep_begin = _handle_boundary_condition(boundary_condition, future_offsets + base_indices - ep_begin, ep_len)
            return (ep_begin + offsets_from_ep_begin) % self._size
        elif sampling_type == 'constant':
            future_offsets = sample_futures_config.get('shift', 10)
            offsets_from_ep_begin = _handle_boundary_condition(boundary_condition, future_offsets + base_indices - ep_begin, ep_len)
            return (ep_begin + offsets_from_ep_begin) % self._size
        else:
            raise ValueError(f'Unknown sampling type: {sampling_type}')

    def sample_future_observation(self, indices: np.ndarray, sample_futures: str):
        future_indices = self.calculate_future_observation_indices(indices, sample_futures)
        return _sample(self.dataset_dict['observations'], future_indices)

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               sample_futures = None,
               relabel: bool = False,
               length: Optional[int] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        samples = super().sample(batch_size, keys, indx, length=length)

        if sample_futures:
            samples = frozen_dict.unfreeze(samples)
            samples['future_observations'] = self.sample_future_observation(indx, sample_futures)
            samples = frozen_dict.freeze(samples)

        if relabel and self._relabel_fn is not None:
            samples = frozen_dict.unfreeze(samples)
            samples = self._relabel_fn(samples)
            samples = frozen_dict.freeze(samples)

        return samples
    
    def sample_trajectories(self, batch_size: int, keys: Optional[Iterable[str]] = None, begin_indx: Optional[np.ndarray] = None, max_trajectory_length: int = 16):
        if begin_indx is None:
            if hasattr(self.np_random, 'integers'):
                begin_indx = self.np_random.integers(len(self), size=batch_size)
            else:
                begin_indx = self.np_random.randint(len(self), size=batch_size)

        samples = self.sample(batch_size, keys, begin_indx, length=max_trajectory_length)

        samples = frozen_dict.unfreeze(samples)

        remaining_lengths = self.dataset_dict['observations']['ep_len'][begin_indx] - self.dataset_dict['observations']['index'][begin_indx]

        # Generate a mask for each sample with index < remaining_lengths
        samples['mask'] = np.arange(max_trajectory_length)[:, None] < remaining_lengths[None, :].squeeze(-1)

        samples = frozen_dict.freeze(samples)

        return samples