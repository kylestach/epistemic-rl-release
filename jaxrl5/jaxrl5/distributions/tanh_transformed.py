from typing import Any, Optional

import tensorflow_probability

import jax
import jax.numpy as jnp
import distrax

# Inspired by
# https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# but modified to only compute a mode.


class TanhTransformedDistribution(distrax.Transformed):
    def __init__(self, distribution: distrax.Distribution, low=-1.0, high=1.0):
        self.high = high
        self.low = low
        self.shift = (high + low) / 2.0
        self.scale = (high - low) / 2.0
        bijector = distrax.Block(distrax.Chain([
            distrax.ScalarAffine(scale=1.0, shift=self.shift),
            distrax.ScalarAffine(scale=self.scale, shift=0),
            distrax.Tanh(),
            distrax.ScalarAffine(scale=1.0 / self.scale, shift=0),
            distrax.ScalarAffine(scale=1.0, shift=-self.shift),
        ]), ndims=1)

        super().__init__(
            distribution=distribution, bijector=bijector
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())
