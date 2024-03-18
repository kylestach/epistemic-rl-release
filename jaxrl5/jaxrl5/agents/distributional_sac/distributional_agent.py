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


class DistributionalStateActionValue(nn.Module):
    base_cls: nn.Module
    num_atoms: int
    min_value: float
    max_value: float

    @nn.compact
    def __call__(
        self, observations: jax.Array, actions: jax.Array, *args, **kwargs
    ) -> Tuple[jax.Array, jax.Array]:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        logits = nn.Dense(self.num_atoms, name="OutputQDense")(outputs)
        atoms = jnp.linspace(self.min_value, self.max_value, self.num_atoms)

        if len(logits.shape) == 2:
            atoms = atoms[None, :].repeat(logits.shape[0], axis=0)
        assert logits.shape == atoms.shape, f"{logits.shape} != {atoms.shape}"

        return logits, atoms


class TanhSquasher(nn.Module):
    high_idx: Optional[int] = struct.field(pytree_node=False, default=None)
    init_value: Optional[float] = struct.field(pytree_node=False, default=None)

    @nn.compact
    def __call__(
        self,
        x: distrax.Distribution,
        output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
    ):
        low = jnp.full(x.event_shape, -1.0)
        high = jnp.full(x.event_shape, 1.0)

        if self.high_idx is not None:
            high_scalar = self.param("high", lambda _: jnp.full((), self.init_value))
            high = high.at[..., self.high_idx].set(high_scalar)

        if output_range is not None:
            low_2, high_2 = output_range
            low = jnp.maximum(low, low_2)
            high = jnp.minimum(high, high_2)

        return TanhTransformedDistribution(
            x,
            low=low,
            high=high,
        )


@partial(jax.jit, static_argnames=("actor_apply_fn", "limits_apply_fn"))
def _sample_actions(
    rng: jax.random.KeyArray,
    actor_apply_fn,
    limits_apply_fn,
    actor_params,
    limits_params,
    observations: jax.Array,
    output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
    **kwargs,
) -> Tuple[jax.Array, jax.random.KeyArray]:
    key, rng = jax.random.split(rng)
    dist = actor_apply_fn({"params": actor_params}, observations, **kwargs)
    dist: distrax.Distribution = limits_apply_fn({"params": limits_params}, dist, output_range=output_range)
    return dist.sample(seed=key), rng


def distributional_target(
    q_atoms: jax.Array,
    target_q_probs: jax.Array,
    target_q_atoms: jax.Array,
    rewards: jax.Array,
    discount: float,
    num_atoms: int,
):
    batch_dims = target_q_probs.shape[:-1]
    assert q_atoms.shape == (
        *batch_dims,
        num_atoms,
    ), f"{q_atoms.shape} != {(*batch_dims, num_atoms)}"
    assert target_q_probs.shape == (
        *batch_dims,
        num_atoms,
    ), f"{target_q_probs.shape} != {(*batch_dims, num_atoms)}"
    assert target_q_atoms.shape == (
        *batch_dims,
        num_atoms,
    ), f"{target_q_atoms.shape} != {(*batch_dims, num_atoms,)}"
    assert rewards.shape == batch_dims, f"{rewards.shape} != {batch_dims}"

    target_value_atoms = rewards[..., None] + discount * target_q_atoms
    assert target_value_atoms.shape == (
        *batch_dims,
        num_atoms,
    ), f"{target_value_atoms.shape} != {(*batch_dims, num_atoms)}"

    def inner(*args):
        return rlax.categorical_l2_project(*args)

    def mid(*args):
        return jax.vmap(inner)(*args)

    result = jax.vmap(mid)(target_value_atoms, target_q_probs, q_atoms)

    assert result.shape == (
        *batch_dims,
        num_atoms,
    ), f"{result.shape} != {(*batch_dims, num_atoms)}"
    return result, target_value_atoms
    # return rlax.categorical_l2_project(target_value_atoms, target_q_probs, q_atoms), q_atoms


def update_distributional_critic(
    critic: TrainState,
    target_critic: TrainState,
    actor: TrainState,
    limits: TrainState,
    batch: DatasetDict,
    rng: jax.random.KeyArray,
    num_qs: int,
    tau: float,
    discount: float,
    num_min_qs: Optional[int] = None,
    target_reduction: str = "min",
    entropy_bonus: float = 0.0,
    output_range_next: Optional[Tuple[jax.Array, jax.Array]] = None,
) -> Tuple[TrainState, TrainState, Dict]:
    batch_size = batch["rewards"].shape[0]

    # Sample actions
    rng, key = jax.random.split(rng)
    next_action_distribution = actor.apply_fn(
        {"params": actor.params},
        batch["next_observations"],
    )
    next_action_distribution = limits.apply_fn(
        {"params": limits.params},
        next_action_distribution,
        output_range=output_range_next,
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
    next_target_q_logits, next_target_q_atoms = target_critic.apply_fn(
        {"params": target_critic_params},
        batch["next_observations"],
        next_actions,
        rngs={"dropout": key},
    )
    # print(jax.tree_map(lambda x: x.shape, target_critic_params))
    # print(next_target_q_logits.shape, next_target_q_atoms.shape)
    next_target_q_atoms = next_target_q_atoms[: num_min_qs or num_qs]
    num_atoms = next_target_q_atoms.shape[-1]

    chex.assert_shape(
        next_target_q_logits, (num_min_qs or num_qs, batch_size, num_atoms)
    )
    chex.assert_equal_shape([next_target_q_logits, next_target_q_atoms])

    # Take the min over the Q-values' CDFs
    next_target_q_probs = jax.nn.softmax(next_target_q_logits, axis=-1)
    assert next_target_q_probs.shape == (
        num_min_qs or num_qs,
        batch_size,
        num_atoms,
    ), f"{next_target_q_probs.shape} != {(num_min_qs or num_qs, batch_size, num_atoms)}"
    target_q_probs, target_q_atoms = distributional_target(
        next_target_q_atoms,
        next_target_q_probs,
        next_target_q_atoms,
        batch["rewards"][None].repeat(num_min_qs or num_qs, axis=0),
        discount,
        num_atoms,
    )

    assert target_q_probs.shape == (
        num_min_qs or num_qs,
        batch_size,
        num_atoms,
    ), f"{target_q_probs.shape} != {(num_min_qs or num_qs, batch_size, num_atoms)}"
    assert target_q_atoms.shape == (
        num_min_qs or num_qs,
        batch_size,
        num_atoms,
    ), f"{target_q_atoms.shape} != {(num_min_qs or num_qs, batch_size, num_atoms)}"

    # Reduction of target distribution
    if target_reduction == "min":
        target_cdf = jnp.cumsum(target_q_probs, axis=-1)
        target_min_cdf = jnp.min(target_cdf, axis=0)
        target_q_probs = jnp.diff(
            jnp.concatenate(
                [jnp.zeros_like(target_min_cdf[..., :1]), target_min_cdf], axis=-1
            ),
            axis=-1,
        )
        target_q_probs = target_q_probs[None]
    elif target_reduction == "mix":
        target_q_probs = jnp.mean(target_q_probs, axis=0)
        target_q_probs = target_q_probs[None]
    elif target_reduction == "independent":
        pass
    else:
        raise ValueError(f"Unknown target reduction: {target_reduction}")

    chex.assert_shape(target_q_probs, (None, batch_size, num_atoms))
    chex.assert_shape(target_q_atoms, (None, batch_size, num_atoms))

    num_atoms = target_q_probs.shape[-1]

    # Compute target Q-values
    def loss(critic_params, rng: jax.random.KeyArray):
        rng, key = jax.random.split(rng)
        qs_logits, qs_atoms = critic.apply_fn(
            {"params": critic_params},
            batch["observations"],
            batch["actions"],
            rngs={"dropout": key},
        )
        qs_probs = jax.nn.softmax(qs_logits, axis=-1)

        chex.assert_shape(qs_logits, (num_qs, batch_size, num_atoms))
        chex.assert_shape(qs_atoms, (num_qs, batch_size, num_atoms))

        critic_entropy = -jnp.sum(jnp.log(qs_probs) * qs_probs, axis=-1).mean()
        loss = rlax.categorical_cross_entropy(
            labels=target_q_probs,
            logits=qs_logits,
        ).mean()

        info = {
            "critic_loss": jnp.mean(loss),
            "critic_value_hist": (
                jnp.mean(jnp.mean(qs_probs, axis=0), axis=0),
                jnp.mean(jnp.mean(qs_atoms, axis=0), axis=0),
            ),
            "distribution_entropy": critic_entropy,
        }

        if entropy_bonus is not None:
            rng, key1, key2, key3 = jax.random.split(rng, 4)
            sampled_actions = actor.apply_fn(
                {"params": actor.params}, batch["observations"], rngs={"dropout": key1}
            ).sample(seed=key2)
            # DO NOT SQUASH HERE. This lets us use an "OOD" version of the policy.
            sampled_qs_logits, _ = critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                sampled_actions,
                rngs={"dropout": key1},
            )
            sampled_qs_probs = jax.nn.softmax(sampled_qs_logits, axis=-1)
            sampled_qs_entropy = -jnp.sum(
                jnp.log(sampled_qs_probs) * sampled_qs_probs, axis=-1
            ).mean()
            loss += entropy_bonus * (critic_entropy - sampled_qs_entropy)
            info["ood_entropy"] = sampled_qs_entropy
            info["entropy_diff"] = critic_entropy - sampled_qs_entropy

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


def cvar(probs, atoms, risk) -> jax.Array:
    assert probs.shape == atoms.shape
    cdf = jnp.cumsum(probs, axis=-1)
    cdf_clipped = jnp.clip(cdf, a_min=0, a_max=1 - risk)
    pdf_clipped = jnp.diff(
        jnp.concatenate([jnp.zeros_like(cdf_clipped[..., :1]), cdf_clipped], axis=-1),
        axis=-1,
    ) / (1 - risk)
    return jnp.sum(pdf_clipped * atoms, axis=-1)


def update_distributional_actor(
    actor: TrainState,
    critic: TrainState,
    limits: TrainState,
    batch: DatasetDict,
    rng: jax.random.KeyArray,
    temperature: float,
    cvar_risk: float,
    output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
):
    batch_size = batch["rewards"].shape[0]

    def loss(params, rng):
        action_distribution = actor.apply_fn({"params": params}, batch["observations"])
        action_distribution = limits.apply_fn(
            {"params": limits.params},
            action_distribution,
            output_range=output_range,
        )

        rng, key = jax.random.split(rng)
        actions, log_probs = action_distribution.sample_and_log_prob(seed=rng)

        rng, key = jax.random.split(rng)
        critic_logits, critic_atoms = critic.apply_fn(
            {"params": critic.params},
            batch["observations"],
            actions,
            rngs={"dropout": key},
        )

        critic_probs = jax.nn.softmax(critic_logits, axis=-1)
        q_cvar = cvar(
            critic_probs.mean(axis=0), critic_atoms.mean(axis=0), risk=cvar_risk
        )

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


def update_distributional_limits(
    actor: TrainState,
    critic: TrainState,
    limits: TrainState,
    batch: DatasetDict,
    rng: jax.random.KeyArray,
    temperature: float,
    cvar_risk: float,
    learned_action_space_idx: int,
    output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
):
    batch_size = batch["rewards"].shape[0]

    def loss(params, rng):
        action_distribution = actor.apply_fn(
            {"params": actor.params}, batch["observations"]
        )
        action_distribution = limits.apply_fn({"params": params}, action_distribution, output_range=output_range)

        rng, key = jax.random.split(rng)
        actions, log_probs = action_distribution.sample_and_log_prob(seed=rng)

        rng, key = jax.random.split(rng)
        critic_logits, critic_atoms = critic.apply_fn(
            {"params": critic.params},
            batch["observations"],
            actions,
            rngs={"dropout": key},
        )

        critic_probs = jax.nn.softmax(critic_logits, axis=-1)
        q_cvar = cvar(
            critic_probs.mean(axis=0), critic_atoms.mean(axis=0), risk=cvar_risk
        )

        assert q_cvar.shape == (batch_size,), f"{q_cvar.shape} != {(batch_size,)}"
        assert log_probs.shape == (batch_size,), f"{log_probs.shape} != {(batch_size,)}"

        actor_loss = (log_probs * temperature - q_cvar).mean()

        return actor_loss, {
            "limit_actor_loss": actor_loss,
            "limit_cvar": q_cvar.mean(),
            "limit_max": action_distribution.high[..., learned_action_space_idx].max(),
        }

    grads, info = jax.grad(loss, has_aux=True)(limits.params, rng)
    limits = limits.apply_gradients(grads=grads)

    return limits, info


class DistributionalSACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    limits: TrainState
    temp: TrainState
    q_entropy_lagrange: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    independent_ensemble: bool = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    initialize_params: Callable[
        [jax.random.KeyArray], Dict[str, TrainState]
    ] = struct.field(pytree_node=False)

    num_atoms: int = struct.field(pytree_node=False)
    q_min: float = struct.field(pytree_node=False)
    q_max: float = struct.field(pytree_node=False)
    cvar_risk: float = struct.field(pytree_node=False)
    cvar_limits: float = struct.field(pytree_node=False)
    q_entropy_target_diff: bool = struct.field(pytree_node=False)
    do_update_limits: bool = struct.field(pytree_node=False)
    learned_action_space_idx: Optional[int] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        q_min: float,
        q_max: float,
        num_atoms: int = 51,
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
        limits_weight_decay: Optional[float] = None,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        cvar_risk: float = 0.0,
        independent_ensemble: bool = False,
        learned_action_space_idx: Optional[int] = None,
        learned_action_space_initial_value: float = 1.0,
        q_entropy_target_diff: float = 0.5,
        q_entropy_lagrange_init: float = 1e-3,
        q_entropy_lagrange_lr: float = 1e-4,
        cvar_limits: float = 0.9,
        limits_lr: float = 1e-5,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        if cvar_limits == "cvar_risk":
            cvar_limits = cvar_risk

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
            DistributionalStateActionValue,
            base_cls=critic_base_cls,
            num_atoms=num_atoms,
            min_value=q_min,
            max_value=q_max,
        )
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = critic_cls()

        temp_def = Temperature(init_temperature)
        tanh_def = TanhSquasher(
            high_idx=learned_action_space_idx,
            init_value=learned_action_space_initial_value,
        )

        q_entropy_lagrange_def = Temperature(q_entropy_lagrange_init)

        sample_actor_distribution = distrax.MultivariateNormalDiag(
            jnp.zeros(
                action_dim,
            ),
            jnp.ones(
                action_dim,
            ),
        )

        do_update_limits = learned_action_space_idx is not None

        # Initialize parameters
        def make_train_states(rng: jax.random.KeyArray) -> Dict[str, TrainState]:
            rngs = jax.random.split(rng, 5)
            actor_params = actor_def.init(rngs[0], observations)["params"]
            critic_params = critic_def.init(rngs[1], observations, actions)["params"]
            temp_params = temp_def.init(rngs[2])["params"]
            tanh_params = (
                tanh_def.init(rngs[3], sample_actor_distribution)["params"]
                if do_update_limits
                else {}
            )
            q_entropy_lagrange_params = q_entropy_lagrange_def.init(rngs[4])["params"]

            critic_optimizer = (
                optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay)
                if critic_weight_decay is not None
                else optax.adam(learning_rate=critic_lr)
            )
            limits_optimizer = (
                optax.adamw(learning_rate=limits_lr, weight_decay=limits_weight_decay)
                if limits_weight_decay is not None
                else optax.adam(learning_rate=limits_lr)
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
                "limits": TrainState.create(
                    apply_fn=tanh_def.apply,
                    params=tanh_params,
                    tx=limits_optimizer,
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
                "q_entropy_lagrange": TrainState.create(
                    apply_fn=q_entropy_lagrange_def.apply,
                    params=q_entropy_lagrange_params,
                    tx=optax.adam(learning_rate=q_entropy_lagrange_lr),
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
            q_min=q_min,
            q_max=q_max,
            num_atoms=num_atoms,
            cvar_risk=cvar_risk,
            cvar_limits=cvar_limits,
            independent_ensemble=independent_ensemble,
            q_entropy_target_diff=q_entropy_target_diff,
            do_update_limits=learned_action_space_idx is not None,
            learned_action_space_idx=learned_action_space_idx,
            **make_train_states(rng),
        )

    def observation_keys(self):
        return "states"

    def reset(self, **kwargs) -> "DistributionalSACLearner":
        if len(kwargs) > 0:
            warnings.warn(
                "DistributionalSACLearner.reset() was called with arguments, ignoring them."
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
        output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
        output_range_next: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> Tuple["DistributionalSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        critic, target_critic, info = update_distributional_critic(
            self.critic,
            self.target_critic,
            self.actor,
            self.limits,
            batch,
            rng=key,
            tau=self.tau,
            discount=self.discount,
            num_qs=self.num_qs,
            num_min_qs=None if self.independent_ensemble else self.num_min_qs,
            target_reduction="independent" if self.independent_ensemble else "mix",
            entropy_bonus=self.q_entropy_lagrange.apply_fn(
                {"params": self.q_entropy_lagrange.params}
            ),
            output_range_next=output_range_next,
        )
        return self.replace(rng=rng, critic=critic, target_critic=target_critic), info

    def update_actor(
        self,
        batch: DatasetDict,
        output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> Tuple["DistributionalSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        temperature = self.temp.apply_fn({"params": self.temp.params})
        actor, info = update_distributional_actor(
            self.actor,
            self.critic,
            self.limits,
            batch,
            key,
            temperature,
            self.cvar_risk,
            output_range=output_range,
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

    def update_q_entropy_lagrange(
        self, q_entropy_diff: float
    ) -> Tuple[Agent, Dict[str, float]]:
        def q_entropy_loss_fn(temp_params):
            lagrange = self.temp.apply_fn({"params": temp_params})
            lagrange_loss = (
                lagrange * (self.q_entropy_target_diff - q_entropy_diff).mean()
            )
            return lagrange_loss, {
                "q_entropy_lagrange": lagrange,
                "q_entropy_lagrange_loss": lagrange_loss,
            }

        grads, temp_info = jax.grad(q_entropy_loss_fn, has_aux=True)(
            self.q_entropy_lagrange.params
        )
        q_entropy_lagrange = self.q_entropy_lagrange.apply_gradients(grads=grads)

        return self.replace(q_entropy_lagrange=q_entropy_lagrange), temp_info

    def update_limits(
        self, batch: DatasetDict, output_range: Optional[Tuple[jax.Array, jax.Array]] = None
    ) -> Tuple["DistributionalSACLearner", Dict[str, float]]:
        rng, key = jax.random.split(self.rng)
        temperature = self.temp.apply_fn({"params": self.temp.params})
        limits, info = update_distributional_limits(
            self.actor,
            self.critic,
            self.limits,
            batch,
            key,
            temperature,
            self.cvar_limits,
            self.learned_action_space_idx,
            output_range,
        )
        return self.replace(rng=rng, limits=limits), info

    def sample_actions(
        self,
        observation: Dict[str, jax.Array],
        output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jax.Array, "DistributionalSACLearner"]:
        if isinstance(observation, (dict, FrozenDict)):
            observation = observation["states"]
        actions, rng = _sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.limits.apply_fn,
            self.actor.params,
            self.limits.params,
            observation,
            output_range=output_range,
        )
        return actions, self.replace(rng=rng)

    @partial(jax.jit, static_argnames=("utd_ratio", "update_temperature"))
    def update(
        self,
        batch: DatasetDict,
        utd_ratio: int,
        update_temperature: bool = True,
        output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_range_next: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
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
            mini_batch_output_range = jax.tree_util.tree_map(partial(slice, i), output_range)
            mini_batch_output_range_next = jax.tree_util.tree_map(partial(slice, i), output_range_next)
            new_agent, critic_info = new_agent.update_critic(
                mini_batch, output_range_next=mini_batch_output_range_next
            )

        new_agent, actor_info = new_agent.update_actor(
            mini_batch,
            output_range=mini_batch_output_range,
        )
        if update_temperature:
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        else:
            temp_info = {}

        if self.do_update_limits:
            new_agent, limits_info = new_agent.update_limits(mini_batch, output_range=mini_batch_output_range)
        else:
            limits_info = {}

        if self.q_entropy_target_diff is not None:
            new_agent, q_entropy_lagrange_info = new_agent.update_q_entropy_lagrange(
                critic_info["entropy_diff"]
            )
        else:
            q_entropy_lagrange_info = {}

        return new_agent, {
            **actor_info,
            **critic_info,
            **temp_info,
            **limits_info,
            **q_entropy_lagrange_info,
        }
