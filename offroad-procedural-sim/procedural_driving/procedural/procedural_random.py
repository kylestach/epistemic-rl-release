import numpy as np
from typing import Tuple

_global_seed = 0
_random_state = np.random.RandomState(_global_seed)

def set_global_seed(seed):
    global _global_seed, _random_state
    _global_seed = seed
    _random_state = np.random.RandomState(_global_seed)

def procedural_random(chunk_pos: Tuple[int, int], stream: str, global_seed=None):
    global _global_seed
    if global_seed is None:
        global_seed = _global_seed

    return np.random.RandomState(hash((chunk_pos, stream, global_seed)))