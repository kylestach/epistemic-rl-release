import jax
import chex

from typing import Dict, List


def rl_sample_config(obs_keys: List[str]):
    pass


def rl_collate(
    data: Dict[str, jax.Array],
    observations_keys: List[str],
):
    batch_size = len(data["reward"])
    chex.assert_shape(data["reward"], (batch_size,))
    chex.assert_shape(data["action"], (batch_size, 2, None))
    chex.assert_shape(data["mask"], (batch_size,))
    for k in observations_keys:
        chex.assert_shape(data[k], (batch_size, 2, None))

    result = {
        "observation": {k: data[k][:, 0] for k in observations_keys},
        "actions": data["action"][:, 0],
        "next_observations": {k: data[k][:, 1] for k in observations_keys},
        "rewards": data["reward"],
        "masks": data["mask"],
    }
    return result
