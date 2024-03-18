from gym import spaces
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from typing import Any, Dict


def obs_to_space(obs: Dict[str, Any]) -> spaces.Space:
    """
    Convert an observation to a gym space.
    """

    if isinstance(obs, (dict, frozen_dict.FrozenDict)):
        return spaces.Dict({k: obs_to_space(v) for k, v in obs.items()})
    elif isinstance(obs, (np.ndarray, jnp.ndarray)):
        return spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape)

    raise ValueError(f"Unknown data type {type(obs)}")
