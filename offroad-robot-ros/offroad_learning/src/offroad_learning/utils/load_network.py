from flax.training import checkpoints
from flax.core import frozen_dict
import importlib
from ml_collections.config_dict.config_dict import ConfigDict
from gym.spaces import Space
from typing import Optional, Any, Dict

from jaxrl5.agents.agent import Agent


def make_agent(
    seed: int,
    agent_cls: str,
    agent_kwargs: Dict[str, Any],
    observation_space: Space,
    action_space: Space,
    checkpoint_dir: Optional[str] = None,
) -> Agent:
    """
    Make an agent from a config file, and optionally load a checkpoint.
    """
    agents = importlib.import_module("jaxrl5.agents")

    agent_cls = agents.__dict__[agent_cls]

    agent = agent_cls.create(seed, observation_space, action_space, **agent_kwargs)

    if checkpoint_dir is not None:
        return checkpoints.restore_checkpoint(checkpoint_dir, target=agent)
    else:
        return agent
