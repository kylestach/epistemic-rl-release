import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.core import frozen_dict
from jaxrl5.networks.encoders import D4PGEncoder
from functools import partial
from typing import Type
import numpy as np
from PIL import Image
import optax
import collections
import os


class StackedImageNetEncoder(nn.Module):
    encoder: Type[nn.Module] = partial(
        D4PGEncoder, (32, 32, 32, 32), (3, 3, 3, 3), (2, 2, 2, 2), padding="VALID"
    )

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (*x.shape[:-1], 3, 3))
        StackedEncoder = nn.vmap(
            self.encoder,
            in_axes=-1,
            out_axes=-1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
        x = StackedEncoder()(x)
        return x.sum(-1)


@jax.jit
def _run_encoder(encoder, pixels_stacked):
    return encoder.apply_fn({"params": encoder.params}, pixels_stacked)


def load_encoder(pretrained_model: str):
    encoder_cls = partial(D4PGEncoder, (32, 32, 32, 32), (3, 3, 3, 3), (2, 2, 2, 2))
    encoder = None
    params = checkpoints.restore_checkpoint(pretrained_model, target=None)
    if "critic" in params:
        encoder_params = params["critic"]["params"]["encoder_0"]
        encoder = encoder_cls()
    elif "encoder" in params:
        encoder_params = params["encoder"]["params"]["encoder_0"]
        encoder = encoder_cls()
    elif "params" in params:
        if "D4PGEncoder_0" in params["params"]:
            encoder = StackedImageNetEncoder(encoder=encoder_cls)

            encoder_params = frozen_dict.unfreeze(
                encoder.init(jax.random.PRNGKey(0), jnp.zeros((1, 128, 128, 9)))[
                    "params"
                ]
            )
            encoder_params["VMapD4PGEncoder"] = params["params"]["D4PGEncoder_0"]
        elif "Encoder_0" in params["params"]:
            encoder_params = params["params"]["Encoder_0"]
            encoder = encoder_cls()
        else:
            raise ValueError(
                f"Could not find encoder in pretrained model.params: {jax.tree_map(lambda x: jnp.shape(x), params['params'])}"
            )
    else:
        raise ValueError("Could not find encoder in pretrained model")

    train_state = TrainState.create(
        apply_fn=encoder.apply,
        params=frozen_dict.freeze(encoder_params),
        tx=optax.GradientTransformation(lambda _: None, lambda _: None),
    )
    return train_state


class JaxEncoder:
    def __init__(self, encoder_file: str, num_stack: int, encoder_dir: str):
        self.encoder = load_encoder(os.path.join(encoder_dir, encoder_file))
        self.latest_pixels = collections.deque(maxlen=num_stack)
        self.img_size = (128, 128)
        for _ in range(num_stack):
            self.latest_pixels.append(np.zeros((*self.img_size, 3)))

    def forward(self, image: Image.Image, goal_image: Image.Image):
        self.latest_pixels.append(np.asarray(image.resize(self.img_size)))
        pixels_stacked = np.stack(self.latest_pixels, axis=-1)
        pixels_stacked = (
            pixels_stacked.reshape((1, *pixels_stacked.shape[:-2], -1)) / 255.0
        )
        assert pixels_stacked.shape == (
            1,
            128,
            128,
            9,
        ), f"pixels_stacked.shape: {pixels_stacked.shape}"

        return np.asarray(_run_encoder(self.encoder, pixels_stacked)[0])
