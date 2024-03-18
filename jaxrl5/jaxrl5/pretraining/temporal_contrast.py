from functools import partial
from typing import Sequence, Type
import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import logsumexp

from flax import struct
from flax.training.train_state import TrainState
import flax.linen as nn

from jaxrl5.agents.drq.augmentations import batched_random_crop
from jaxrl5.networks.encoders.d4pg_encoder import D4PGEncoder
from jaxrl5.networks.mlp import MLP

@jax.jit
def _loss_contrastive(a_batch, b_batch):
    # a, b: (batch_size, dim)

    def vmap_inner(a, b_batch):
        return jax.vmap(
            jnp.dot,
            in_axes=(None, 0),
            out_axes=0,
        )(a, b_batch)

    logits = jax.vmap(
        vmap_inner,
        in_axes=(0, None),
        out_axes=0,
    )(a_batch, b_batch)
    assert logits.shape == (a_batch.shape[0], b_batch.shape[0])

    return -jnp.mean(jnp.diag(logits) - logsumexp(logits, axis=1)), logits


class EncoderCompressor(nn.Module):
    encoder_cls: Type[nn.Module]
    compressor_cls: Type[nn.Module]
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = self.encoder_cls(name="encoder_0")(x)
        x = self.compressor_cls(name="compressor_0")(x)
        x = nn.Dense(features=self.latent_dim)(x)

        return x


class TemporalContrastLearner(struct.PyTreeNode):
    rng: jax.random.PRNGKey
    encoder: TrainState
    target_encoder: TrainState
    predictor: TrainState
    tau: float

    @classmethod
    def create(
        cls,
        seed: int,
        encoder_lr: float = 3e-4,
        predictor_lr: float = 3e-4,
        tau: float = 0.005,
        encoder: str = "d4pg",
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        compressor_hidden_dims: Sequence[int] = (256, 256),
        predictor_hidden_dims: Sequence[int] = (256, 256),
        latent_dim: int = 50,
        image_size: int = 128,
        num_channels: int = 3,
        num_stack: int = 3,
        ):

        image_shape = (image_size, image_size, num_channels * num_stack)

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        
        compressor_cls = partial(MLP, hidden_dims=compressor_hidden_dims, activate_final=True)

        encoder_cls = partial(EncoderCompressor, encoder_cls=encoder_cls, compressor_cls=compressor_cls, latent_dim=latent_dim)
        encoder_def = encoder_cls()
        encoder_params = encoder_def.init(jax.random.PRNGKey(seed), jnp.ones(image_shape))['params']
        encoder = TrainState.create(apply_fn=encoder_def.apply, params=encoder_params, tx=optax.adam(encoder_lr))

        target_encoder = TrainState.create(apply_fn=encoder_def.apply, params=encoder_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        predictor_cls = partial(MLP, hidden_dims=(*predictor_hidden_dims, latent_dim), activate_final=False)
        predictor_def = predictor_cls()
        predictor_params = predictor_def.init(jax.random.PRNGKey(seed), jnp.ones((latent_dim,)))['params']
        predictor = TrainState.create(apply_fn=predictor_def.apply, params=predictor_params, tx=optax.adam(predictor_lr))

        return cls(
            rng=jax.random.PRNGKey(seed),
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            tau=tau,
        )

    @jax.jit
    def update(self, batch):
        rng, key_random_crop, key_random_brightness = jax.random.split(self.rng, 3)

        obs = batch["observations"]["pixels"]
        future_obs = batched_random_crop(key_random_crop, batch["future_observations"], "pixels")["pixels"]
        obs = obs.reshape((*obs.shape[:-2], -1))
        future_obs = future_obs.reshape((*future_obs.shape[:-2], -1))
        future_obs = future_obs * jax.random.uniform(key_random_brightness, (obs.shape[0], 1, 1, 1), minval=0.5, maxval=1.5)

        def _loss_fn(params):
            encoder_params, predictor_params = params

            a = self.encoder.apply_fn({'params': encoder_params}, obs)
            a_fut = self.predictor.apply_fn({'params': predictor_params}, a)
            b = self.target_encoder.apply_fn({'params': self.target_encoder.params}, future_obs)
            loss, logits = _loss_contrastive(a_fut, b)

            return loss, {'contrastive_loss': loss, 'logits': logits}

        (encoder_grads, predictor_grads), info = jax.grad(_loss_fn, has_aux=True)((self.encoder.params, self.predictor.params))
        encoder = self.encoder.apply_gradients(grads=encoder_grads)
        predictor = self.predictor.apply_gradients(grads=predictor_grads)
        target_encoder = self.target_encoder.replace(params=optax.incremental_update(
            encoder.params, self.target_encoder.params, self.tau
        ))

        return self.replace(rng=rng, encoder=encoder, target_encoder=target_encoder, predictor=predictor), info
