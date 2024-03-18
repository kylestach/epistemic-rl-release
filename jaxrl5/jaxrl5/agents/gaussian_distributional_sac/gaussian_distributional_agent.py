"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Callable
import warnings

import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
import chex
import distrax

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import Normal, TanhTransformedDistribution
from jaxrl5.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    subsample_ensemble,
    PixelMultiplexer,
)

import rlax
from flax import linen as nn


class GaussianStateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jax.Array, actions: jax.Array, *args, **kwargs
    ) -> Tuple[jax.Array, jax.Array]:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        mean = nn.Dense(1, name="OutputQDense")(outputs)
        std = nn.Dense(1, name="OutputStdDense")(outputs)
        mean = jnp.squeeze(mean, axis=-1)
        std = jax.nn.softplus(jnp.squeeze(std, axis=-1))

        chex.assert_shape([mean, std], observations.shape[:-1])

        return mean, std


def gaussian_distributional_target(
    next_q_mean: jax.Array,
    next_q_std: jax.Array,
    rewards: jax.Array,
    discount: float,
):
    chex.assert_equal_shape([next_q_mean, next_q_std, rewards])
    return rewards + discount * next_q_mean, discount * next_q_std


def update_gaussian_distributional_critic(
    critic: TrainState,
    target_critic: TrainState,
    actor: TrainState,
    batch: DatasetDict,
    rng: jax.random.KeyArray,
    num_qs: int,
    tau: float,
    discount: float,
) -> Tuple[TrainState, TrainState, Dict]:
    batch_size = batch["rewards"].shape[0]

    # Sample actions
    rng, key = jax.random.split(rng)
    next_action_distribution = actor.apply_fn(
        {"params": actor.params},
        batch["next_observations"],
    )
    next_actions = next_action_distribution.sample(seed=key)

    # Compute next Q-values
    target_critic_params = target_critic.params

    rng, key = jax.random.split(rng)
    next_target_q_mean, next_target_q_std = target_critic.apply_fn(
        {"params": target_critic_params},
        batch["next_observations"],
        next_actions,
        rngs={"dropout": key},
    )

    rewards = batch["rewards"][None].repeat(num_qs, axis=0)
    chex.assert_shape([next_target_q_mean, next_target_q_std, rewards], (num_qs, batch_size))

    target_q_mean, target_q_std = gaussian_distributional_target(
        next_target_q_mean,
        next_target_q_std,
        rewards,
        discount,
    )

    chex.assert_shape([target_q_mean, target_q_std], (num_qs, batch_size))

    # Compute target Q-values
    def loss(critic_params, rng: jax.random.KeyArray):
        rng, key = jax.random.split(rng)
        qs_mean, qs_std = critic.apply_fn(
            {"params": critic_params},
            batch["observations"],
            batch["actions"],
            rngs={"dropout": key},
        )

        chex.assert_equal_shape([target_q_mean, target_q_std, qs_mean, qs_std])
        loss = distrax.Normal(target_q_mean, target_q_std).kl_divergence(distrax.Normal(loc=qs_mean, scale=qs_std)).mean()

        info = {
            "critic_loss": jnp.mean(loss),
            "critic_mean": jnp.mean(qs_mean),
            "critic_std": jnp.mean(qs_std),
        }

        return loss, info

    rng, key = jax.random.split(rng)
    grads, info = jax.grad(loss, has_aux=True)(critic.params, key)
    critic = critic.apply_gradients(grads=grads)
    target_critic = target_critic.replace(
        params=optax.incremental_update(
            critic.params, target_critic.params, step_size=tau
        )
    )

    return critic, target_critic, info


def gaussian_cvar(mean, std, beta) -> jax.Array:
    # beta is phi(Phi^{-1}(alpha))/(1-alpha)
    # hardcode beta for 0.9
    beta = 1.755
    return mean - std * beta


def update_gaussian_distributional_actor(
    actor: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    rng: jax.random.KeyArray,
    temperature: float,
):
    batch_size = batch["rewards"].shape[0]

    def loss(params, rng):
        action_distribution = actor.apply_fn({"params": params}, batch["observations"])

        rng, key = jax.random.split(rng)
        actions, log_probs = action_distribution.sample_and_log_prob(seed=rng)

        rng, key = jax.random.split(rng)
        q_mean, q_std = critic.apply_fn(
            {"params": critic.params},
            batch["observations"],
            actions,
            rngs={"dropout": key},
        )

        q_cvar = gaussian_cvar(q_mean.mean(axis=0), q_std.mean(axis=0), None)

        assert q_cvar.shape == (batch_size,), f"{q_cvar.shape} != {(batch_size,)}"
        assert log_probs.shape == (batch_size,), f"{log_probs.shape} != {(batch_size,)}"

        actor_loss = (log_probs * temperature - q_cvar).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "cvar": q_cvar.mean(),
            "entropy": -log_probs.mean(),
            "mean_action": jnp.mean(actions[..., 1]),
        }

    grads, info = jax.grad(loss, has_aux=True)(actor.params, rng)
    actor = actor.apply_gradients(grads=grads)

    return actor, info


class GaussianDistributionalSACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)
    initialize_params: Callable[
        [jax.random.KeyArray], Dict[str, TrainState]
    ] = struct.field(pytree_node=False)


    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        critic_weight_decay: Optional[float] = None,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if isinstance(observations, dict):
            assert "states" in observations
            observations = observations["states"]

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)

        actor_base_cls = partial(
            MLP, hidden_dims=actor_hidden_dims, activate_final=True
        )
        actor_cls = partial(Normal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = actor_cls()

        critic_base_cls = partial(
            MLP,
            hidden_dims=critic_hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(
            GaussianStateActionValue,
            base_cls=critic_base_cls,
        )
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = critic_cls()

        temp_def = Temperature(init_temperature)

        # Initialize parameters
        def make_train_states(rng: jax.random.KeyArray) -> Dict[str, TrainState]:
            rngs = jax.random.split(rng, 5)
            actor_params = actor_def.init(rngs[0], observations)["params"]
            critic_params = critic_def.init(rngs[1], observations, actions)["params"]
            temp_params = temp_def.init(rngs[2])["params"]

            critic_optimizer = (
                optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay)
                if critic_weight_decay is not None
                else optax.adam(learning_rate=critic_lr)
            )

            return {
                "actor": TrainState.create(
                    apply_fn=actor_def.apply,
                    params=actor_params,
                    tx=optax.adam(learning_rate=actor_lr),
                ),
                "critic": TrainState.create(
                    apply_fn=critic_def.apply, params=critic_params, tx=critic_optimizer
                ),
                "target_critic": TrainState.create(
                    apply_fn=critic_def.apply,
                    params=critic_params,
                    tx=optax.GradientTransformation(lambda _: None, lambda _: None),
                ),
                "temp": TrainState.create(
                    apply_fn=temp_def.apply,
                    params=temp_params,
                    tx=optax.adam(learning_rate=temp_lr),
                ),
            }

        return cls(
            rng=rng,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            initialize_params=make_train_states,
            **make_train_states(rng),
        )

    def observation_keys(self):
        return "states"

    def reset(self, **kwargs) -> "GaussianDistributionalSACLearner":
        if len(kwargs) > 0:
            warnings.warn(
                "GaussianDistributionalSACLearner.reset() was called with arguments, ignoring them."
            )

        rng, key = jax.random.split(self.rng)
        train_states = self.initialize_params(key)
        del train_states["temp"]
        return self.replace(
            rng=rng,
            **train_states,
        )

    def update_critic(
        self,
        batch: DatasetDict,
    ) -> Tuple["GaussianDistributionalSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        critic, target_critic, info = update_gaussian_distributional_critic(
            self.critic,
            self.target_critic,
            self.actor,
            batch,
            rng=key,
            num_qs=self.num_qs,
            tau=self.tau,
            discount=self.discount,
        )
        return self.replace(rng=rng, critic=critic, target_critic=target_critic), info

    def update_actor(
        self,
        batch: DatasetDict,
    ) -> Tuple["GaussianDistributionalSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        temperature = self.temp.apply_fn({"params": self.temp.params})
        actor, info = update_gaussian_distributional_actor(
            self.actor,
            self.critic,
            batch,
            rng=key,
            temperature=temperature,
        )
        return self.replace(rng=rng, actor=actor), info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    @partial(jax.jit, static_argnames=("utd_ratio", "update_temperature"))
    def update(
        self,
        batch: DatasetDict,
        utd_ratio: int,
        update_temperature: bool = True,
        output_range: Optional[Tuple[float, float]] = None,
        output_range_next: Optional[Tuple[float, float]] = None,
    ):
        if isinstance(batch["observations"], (dict, FrozenDict)):
            batch = {
                **batch,
                "observations": batch["observations"]["states"],
                "next_observations": batch["next_observations"]["states"],
            }

        def slice(i, x):
            assert x.shape[0] % utd_ratio == 0
            batch_size = x.shape[0] // utd_ratio
            return x[batch_size * i : batch_size * (i + 1)]

        new_agent = self
        for i in range(utd_ratio):
            mini_batch = jax.tree_util.tree_map(partial(slice, i), batch)
            new_agent, critic_info = new_agent.update_critic(
                mini_batch,
            )

        new_agent, actor_info = new_agent.update_actor(
            mini_batch,
        )
        if update_temperature:
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        else:
            temp_info = {}

        return new_agent, {
            **actor_info,
            **critic_info,
            **temp_info,
        }
