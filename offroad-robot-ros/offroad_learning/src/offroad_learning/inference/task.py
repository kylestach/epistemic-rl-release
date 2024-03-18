import numpy as np
from PIL import Image

from typing import Any, Dict, Optional, List
import io
import re
import collections
import rospkg
import os


class Task:
    def __init__(self):
        pass

    def setup_ros(self):
        """
        Deferred setup of ROS publishers and subscribers
        """
        pass

    def tick(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tick the task, and return any task-specific observables.

        This can also be used to update the task's internal state.
        """
        return {}

    def compute_reward(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
    ):
        """
        Compute the reward for the given transition.

        Should be independent of the task's internal state.
        """
        raise NotImplementedError()

    def is_terminated(self, observation: Dict[str, Any]):
        """
        Return whether the episode is terminated.

        Should be independent of the task's internal state.
        """
        raise NotImplementedError()

    def is_truncated(self, observation: Dict[str, Any]):
        """
        Return whether the episode is truncated.

        Should be independent of the task's internal state.
        """
        raise NotImplementedError()

    def preprocess_observations(self, observations: Dict[str, Any]):
        """
        Called before preparing observations for the server or actor.
        """
        return observations

    def _preprocess_observations_for_server(
        self, observations: Dict[str, Any], **kwargs
    ):
        """
        Called before preparing observations for the server.

        Override this to do any additional processing
        """
        truncated = self.is_truncated(observations)
        terminated = self.is_terminated(observations)
        actions = observations.pop("action")
        rewards = self.compute_reward(observations, actions)

        return {
            "observations": observations,
            "actions": actions,
            "rewards": np.asarray(rewards),
            "dones": np.asarray(truncated or terminated),
            "masks": np.asarray(not terminated),
        }

    def _preprocess_observations_for_actor(self, observations: Dict[str, Any]):
        """
        Called before preparing observations for the actor.

        Override this to do any additional processing
        """
        return observations

    def prepare_observations_for_server(self, observations: Dict[str, Any], **kwargs):
        """
        Prepare an observation for the server.

        Do not override this method. Override _preprocess_observations_for_server instead.
        """
        return self._preprocess_observations_for_server(
            self.preprocess_observations(observations),
            **kwargs,
        )

    def prepare_observations_for_actor(self, observations: Dict[str, Any]):
        """
        Prepare an observation for the actor.

        Do not override this method. Override _preprocess_observations_for_actor instead.
        """
        return self._preprocess_observations_for_actor(
            self.preprocess_observations(observations)
        )


class Wrapper(Task):
    def __init__(self, task: Task):
        super().__init__()
        self._task = task

    def setup_ros(self):
        self._task.setup_ros()

    def compute_reward(
        self,
        observations: Dict[str, Any],
        actions: np.ndarray,
    ):
        return self._task.compute_reward(
            observations,
            actions,
        )

    def tick(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self._task.tick(observations)

    def is_terminated(self, observations: Dict[str, Any]):
        return self._task.is_terminated(observations)

    def is_truncated(self, observations: Dict[str, Any]):
        return self._task.is_truncated(observations)

    def preprocess_observations(self, observations: Dict[str, Any]):
        return self._task.preprocess_observations(observations)

    def _preprocess_observations_for_server(
        self, observations: Dict[str, Any], **kwargs
    ):
        return self._task._preprocess_observations_for_server(observations, **kwargs)

    def _preprocess_observations_for_actor(self, observations: Dict[str, Any]):
        return self._task._preprocess_observations_for_actor(observations)


class ImageWrapper(Wrapper):
    def __init__(self, task: Task, pixels_regex: str, num_stack: int):
        super().__init__(task)
        self.latest_pixels = {}
        self.num_stack = num_stack
        self.pixels_regex = pixels_regex

    def _preprocess_observations_for_actor(self, observations: Dict[str, Any]):
        observations = self._task.prepare_observations_for_actor(observations)

        for k in observations.keys():
            if not re.match(self.pixels_regex, k):
                continue

            if k not in self.latest_pixels:
                self.latest_pixels[k] = collections.deque(maxlen=self.num_stack)
                for _ in range(self.num_stack):
                    self.latest_pixels[k].append(np.zeros_like(observations[k]))

            self.latest_pixels[k].append(observations[k])

            if len(self.latest_pixels[k]) < self.num_stack:
                observations[k] = None
            else:
                observations[k] = np.stack(self.latest_pixels[k], axis=-1)

        return observations


class ImageEmbeddingsWrapper(Wrapper):
    def __init__(self, task: Task, encoder_config: Dict[str, Any]):
        super().__init__(task)
        self.encoder_config = encoder_config
        self.pixels_key = self.encoder_config.get("pixels_key", "pixels")
        self.goal_pixels_key = self.encoder_config.get("goal_pixels_key", None)
        self.image_embeddings_key = self.encoder_config.get("image_embeddings_key", "image_embeddings")

        rospack = rospkg.RosPack()
        encoder_dir = os.path.join(rospack.get_path("offroad_learning"), "encoders")
        encoder_type = encoder_config["type"]
        if encoder_type == "jax_convnet":
            from offroad_learning.inference.jax_encoders import JaxEncoder

            self.encoder = JaxEncoder(
                **encoder_config["kwargs"], encoder_dir=encoder_dir
            )
        elif encoder_type == "dino":
            from offroad_learning.inference.torch_encoders import DinoEncoder

            self.encoder = DinoEncoder(**encoder_config["kwargs"])
        elif encoder_type == "gnm":
            from offroad_learning.inference.torch_encoders import FullGnmEncoder

            self.encoder = FullGnmEncoder(
                **encoder_config["kwargs"], encoder_dir=encoder_dir
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def preprocess_observations(self, observations: Dict[str, Any]):
        observations = super().preprocess_observations(observations).copy()

        if observations[self.pixels_key] is None:
            observations[self.image_embeddings_key] = None
        elif (
            self.goal_pixels_key is not None
            and observations[self.goal_pixels_key] is None
        ):
            observations[self.image_embeddings_key] = None
        else:
            observations[self.image_embeddings_key] = self.encoder.forward(
                Image.fromarray(observations["pixels"]),
                Image.fromarray(observations[self.goal_pixels_key])
                if self.goal_pixels_key is not None
                else None,
            )

        return observations


class CompressionWrapper(Wrapper):
    def __init__(self, task: Task, pixel_regex: str):
        super().__init__(task)
        self.pixel_regex = pixel_regex

    def _preprocess_observations_for_server(
        self, observations: Dict[str, Any], compress=True, **kwargs
    ):
        # Compress originals so we don't send stacked
        compressed_pixels = {}
        if compress:
            for k in observations.keys():
                if re.match(self.pixel_regex, k):
                    pixel_jpeg_bytes = io.BytesIO()
                    Image.fromarray(observations[k]).save(
                        pixel_jpeg_bytes, format="JPEG", quality=80
                    )
                    compressed_pixels[k] = np.frombuffer(
                        pixel_jpeg_bytes.getvalue(), dtype=np.uint8
                    )

        observations = super()._preprocess_observations_for_server(
            observations, **kwargs
        )
        observations["observations"].update(compressed_pixels)

        return observations


class CompileStatesWrapper(Wrapper):
    def __init__(self, task: Task, states_keys: List[str], key: str = "states"):
        super().__init__(task)
        self.states_keys = states_keys
        self.key = key

    def preprocess_observations(self, observations: Dict[str, Any]):
        observations = super().preprocess_observations(observations).copy()

        if any(observations[k] is None for k in self.states_keys):
            observations[self.key] = None
        else:
            observations[self.key] = np.concatenate(
                [observations[k] for k in self.states_keys], axis=-1
            )

        return observations


class SelectKeyWrapper(Wrapper):
    def __init__(self, task: Task, key: str):
        super().__init__(task)
        self.key = key

    def _preprocess_observations_for_actor(self, observations: Dict[str, Any]):
        observations = super()._preprocess_observations_for_actor(observations)

        if observations[self.key] is None:
            return None

        return observations[self.key]
