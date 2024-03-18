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
import chex

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble, PixelMultiplexer

import rlax
from flax import linen as nn


class QuantileStateActionValue(nn.Module):
    base_cls: nn.Module
    num_quantiles: int

    @nn.compact
    def __call__(
        self, observations: jax.Array, actions: jax.Array, *args, **kwargs
    ) -> Tuple[jax.Array, jax.Array]:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        quantiles = nn.Dense(self.num_quantiles, name='OutputQDense')(outputs)
        quantiles = jnp.sort(quantiles, axis=-1)

        return quantiles

def quantile_target(target_q_quantiles: jax.Array, rewards: jax.Array, discount: float, num_quantiles: int):
    batch_dims = target_q_quantiles.shape[:-1]
    assert target_q_quantiles.shape == (*batch_dims, num_quantiles), f"{target_q_quantiles.shape} != {(*batch_dims, num_quantiles)}"
    assert rewards.shape == batch_dims, f"{rewards.shape} != {batch_dims}"

    target_q_quantiles = rewards[..., None] + discount * target_q_quantiles
    assert target_q_quantiles.shape == (*batch_dims, num_quantiles), f"{target_q_quantiles.shape} != {(*batch_dims, num_quantiles)}"

    return target_q_quantiles

def update_quantile_critic(
        critic: TrainState,
        target_critic: TrainState,
        actor: TrainState,
        batch: DatasetDict,
        rng: jax.random.KeyArray,
        num_qs: int,
        tau: float,
        discount: float,
        num_min_qs: Optional[int] = None,
        output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
        target_reduction: str = "min",
        num_quantiles: int = 11,
    ) -> Tuple[TrainState, TrainState, Dict]:
    batch_size = batch["rewards"].shape[0]

    # Sample actions
    rng, key = jax.random.split(rng)
    next_action_distribution = actor.apply_fn(
        {"params": actor.params}, batch["next_observations"], output_range=output_range
    )
    next_actions = next_action_distribution.sample(seed=key)

    # Compute next Q-values
    if num_min_qs is None:
        target_critic_params = target_critic.params
    else:
        rng, key = jax.random.split(rng)
        target_critic_params = subsample_ensemble(
            key, target_critic.params, num_min_qs, num_qs
        )
    rng, key = jax.random.split(rng)
    next_target_q_quantiles = target_critic.apply_fn(
        {"params": target_critic_params}, batch["next_observations"], next_actions, rngs={"dropout": key},
    )

    chex.assert_shape(next_target_q_quantiles, (num_min_qs or num_qs, batch_size, num_quantiles))

    # Take the min over the Q-values' CDFs
    next_target_q_probs = jax.nn.softmax(next_target_q_quantiles, axis=-1)
    assert next_target_q_probs.shape == (num_min_qs or num_qs, batch_size, num_quantiles), f"{next_target_q_probs.shape} != {(num_min_qs or num_qs, batch_size, num_quantiles)}"
    target_q_quantiles = quantile_target(
        target_q_quantiles=next_target_q_quantiles,
        rewards=batch["rewards"][None].repeat(num_min_qs or num_qs, axis=0),
        discount=discount,
        num_quantiles=num_quantiles,
    )

    assert target_q_quantiles.shape == (num_min_qs or num_qs, batch_size, num_quantiles), f"{target_q_quantiles.shape} != {(num_min_qs or num_qs, batch_size, num_quantiles)}"

    # Reduction of target distribution
    if target_reduction == "independent":
        pass
    else:
        raise ValueError(f"Unknown target reduction: {target_reduction}")

    chex.assert_shape(target_q_quantiles, (None, batch_size, num_quantiles))

    rng, key = jax.random.split(rng)
    # Compute target Q-values
    def loss(critic_params):
        qs_quantiles = critic.apply_fn(
            {"params": critic_params}, batch["observations"], batch["actions"], rngs={"dropout": key},
        )

        chex.assert_shape(qs_quantiles, (num_qs, batch_size, num_quantiles))

        loss = jnp.mean(
            rlax.huber_loss(
                qs_quantiles - target_q_quantiles, delta=1.0,
            )
        )

        mean_qs_quantiles = jnp.mean(qs_quantiles, axis=0).mean(axis=0)
        chex.assert_shape(mean_qs_quantiles, (num_quantiles,))

        return loss, {
            "critic_loss": jnp.mean(loss),
            "critic_value_hist": (jnp.full_like(mean_qs_quantiles, 1 / num_quantiles), mean_qs_quantiles),
        }

    grads, info = jax.grad(loss, has_aux=True)(critic.params)
    critic = critic.apply_gradients(grads=grads)
    target_critic = target_critic.replace(params=optax.incremental_update(critic.params, target_critic.params, step_size=tau))

    return critic, target_critic, info

def cvar(probs, quantiles, risk) -> jax.Array:
    assert probs.shape == quantiles.shape
    cdf = jnp.cumsum(probs, axis=-1)
    cdf_clipped = jnp.clip(cdf, a_min=0, a_max=1-risk)
    pdf_clipped = jnp.diff(jnp.concatenate([jnp.zeros_like(cdf_clipped[..., :1]), cdf_clipped], axis=-1), axis=-1) / (1 - risk)
    return jnp.sum(pdf_clipped * quantiles, axis=-1)

def update_quantile_actor(
        actor: TrainState,
        critic: TrainState,
        batch: DatasetDict,
        rng: jax.random.KeyArray,
        temperature: float,
        cvar_risk: float,
        output_range: Optional[Tuple[jax.Array, jax.Array]],
    ):
    batch_size = batch["rewards"].shape[0]

    def loss(params, rng):
        action_distribution = actor.apply_fn({"params": params}, batch["observations"], output_range=output_range)
        rng, key = jax.random.split(rng)
        actions, log_probs = action_distribution.sample_and_log_prob(seed=rng)

        rng, key = jax.random.split(rng)
        critic_quantiles = critic.apply_fn(
            {"params": critic.params}, batch["observations"], actions, rngs={"dropout": key},
        )
        num_quantiles = critic_quantiles.shape[-1]

        q_cvar = cvar(jnp.full_like(critic_quantiles.mean(axis=0), 1.0 / num_quantiles), critic_quantiles.mean(axis=0), risk=cvar_risk)

        assert q_cvar.shape == (batch_size,), f"{q_cvar.shape} != {(batch_size,)}"
        assert log_probs.shape == (batch_size,), f"{log_probs.shape} != {(batch_size,)}"

        actor_loss = (
            log_probs * temperature - q_cvar
        ).mean()

        return actor_loss, {"actor_loss": actor_loss, "cvar": q_cvar.mean(), "entropy": -log_probs.mean()}

    grads, info = jax.grad(loss, has_aux=True)(actor.params, rng)
    actor = actor.apply_gradients(grads=grads)

    return actor, info

class QuantileSACLearner(Agent):
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
    independent_ensemble: bool = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    initialize_params: Callable[[jax.random.KeyArray], Dict[str, TrainState]] = struct.field(pytree_node=False)

    num_quantiles: int = struct.field(pytree_node=False)
    cvar_risk: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_quantiles: int = 51,
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
        cvar_risk: float = 0.0,
        independent_ensemble: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)

        actor_base_cls = partial(MLP, hidden_dims=actor_hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = actor_cls()

        critic_base_cls = partial(
            MLP,
            hidden_dims=critic_hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(QuantileStateActionValue, base_cls=critic_base_cls, num_quantiles=num_quantiles)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = critic_cls()

        temp_def = Temperature(init_temperature)

        # Initialize parameters
        def make_train_states(rng: jax.random.KeyArray) -> Dict[str, TrainState]:
            rngs = jax.random.split(rng, 3)
            actor_params = actor_def.init(rngs[0], observations)["params"]
            critic_params = critic_def.init(rngs[1], observations, actions)["params"]
            temp_params = temp_def.init(rngs[2])["params"]
            return {
                "actor": TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr)),
                "critic": TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay)),
                "target_critic": TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)),
                "temp": TrainState.create(apply_fn=temp_def.apply, params=temp_params, tx=optax.adam(learning_rate=temp_lr)),
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
            num_quantiles=num_quantiles,
            cvar_risk=cvar_risk,
            independent_ensemble=independent_ensemble,
            **make_train_states(rng)
        )

    def reset(self, **kwargs) -> "QuantileSACLearner":
        if len(kwargs) > 0:
            warnings.warn("QuantileSACLearner.reset() was called with arguments, ignoring them.")

        rng, key = jax.random.split(self.rng)
        train_states = self.initialize_params(key)
        del train_states["temp"]
        return self.replace(
            rng=rng,
            **train_states,
        )

    def update_critic(self, batch: DatasetDict, output_range) -> Tuple["QuantileSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        critic, target_critic, info = update_quantile_critic(
            self.critic,
            self.target_critic,
            self.actor,
            batch,
            rng=key,
            tau=self.tau,
            discount=self.discount,
            num_qs=self.num_qs,
            num_min_qs=None if self.independent_ensemble else self.num_min_qs,
            output_range=output_range,
            target_reduction="independent" if self.independent_ensemble else "mix",
        )
        return self.replace(rng=rng, critic=critic, target_critic=target_critic), info

    def update_actor(self, batch: DatasetDict, output_range) -> Tuple["QuantileSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        temperature = self.temp.apply_fn({"params": self.temp.params})
        actor, info = update_quantile_actor(self.actor, self.critic, batch, key, temperature, self.cvar_risk, output_range=output_range)
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
    def update(self, batch: DatasetDict, utd_ratio: int, update_temperature: bool = True, output_range: Optional[Tuple[jax.Array]] = None):
        def slice(i, x):
            assert x.shape[0] % utd_ratio == 0
            batch_size = x.shape[0] // utd_ratio
            return x[batch_size * i : batch_size * (i + 1)]

        new_agent = self
        for i in range(utd_ratio):
            mini_batch = jax.tree_util.tree_map(partial(slice, i), batch)
            mini_batch_output_range = jax.tree_util.tree_map(partial(slice, i), output_range)
            new_agent, critic_info = new_agent.update_critic(mini_batch, output_range=mini_batch_output_range)

        new_agent, actor_info = new_agent.update_actor(
            mini_batch,
            output_range=jax.tree_util.tree_map(
                partial(slice, utd_ratio - 1), output_range
            ),
        )
        if update_temperature:
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        else:
            temp_info = {}

        return new_agent, {**actor_info, **critic_info, **temp_info}
