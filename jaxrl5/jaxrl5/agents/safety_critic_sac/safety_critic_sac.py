"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Callable, List

import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from distrax import Distribution
import chex

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    subsample_ensemble,
    PixelMultiplexer,
)


def _reset_params(old, new, alpha=0.2, ensemble_idx=None):
    def _interp(key_path, old: jax.Array, new: jax.Array) -> jax.Array:
        result = new

        if "encoder" in jax.tree_util.keystr(key_path).lower():
            result = alpha * new + (1 - alpha) * old

        if (
            ensemble_idx is not None
            and "ensemble" in jax.tree_util.keystr(key_path).lower()
        ):
            result = old.at[ensemble_idx].set(result[ensemble_idx])

        return result

    # Use the source for everything except the encoder, which is interpolated:
    new_params = jax.tree_util.tree_map_with_path(_interp, old, new)

    return new_params


class SafetyCriticSACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    safety_critic: TrainState
    target_safety_critic: TrainState

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
    pixel_embeddings_key: Optional[str] = struct.field(pytree_node=False)

    safety_threshold: float = struct.field(pytree_node=False)
    safety_discount: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
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
        pixel_embeddings_key: Optional[str] = None,
        safety_threshold: float = 1.0,
        safety_discount: float = 0.9,
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

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        if pixel_embeddings_key is not None:
            actor_def = PixelMultiplexer(
                encoder_cls=None,
                network_cls=actor_cls,
                latent_dim=50,
                stop_gradient=True,
                pixel_keys=(pixel_embeddings_key,),
            )
        else:
            actor_def = actor_cls()

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        if pixel_embeddings_key is not None:
            critic_def = PixelMultiplexer(
                encoder_cls=None,
                network_cls=critic_cls,
                latent_dim=50,
                stop_gradient=True,
                pixel_keys=(pixel_embeddings_key,),
            )
        else:
            critic_def = critic_cls()

        temp_def = Temperature(init_temperature)

        # Initialize parameters
        def make_train_states(rng: jax.random.KeyArray) -> Dict[str, TrainState]:
            rngs = jax.random.split(rng, 4)
            actor_params = actor_def.init(rngs[0], observations)["params"]
            critic_params = critic_def.init(rngs[1], observations, actions)["params"]
            safety_critic_params = critic_def.init(rngs[2], observations, actions)["params"]
            temp_params = temp_def.init(rngs[3])["params"]
            return {
                "actor": TrainState.create(
                    apply_fn=actor_def.apply,
                    params=actor_params,
                    tx=optax.adam(learning_rate=actor_lr),
                ),
                "critic": TrainState.create(
                    apply_fn=critic_def.apply,
                    params=critic_params,
                    tx=optax.adam(learning_rate=critic_lr) if critic_weight_decay is None else optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay),
                ),
                "target_critic": TrainState.create(
                    apply_fn=critic_def.apply,
                    params=critic_params,
                    tx=optax.GradientTransformation(lambda _: None, lambda _: None),
                ),
                "safety_critic": TrainState.create(
                    apply_fn=critic_def.apply,
                    params=safety_critic_params,
                    tx=optax.adam(learning_rate=critic_lr) if critic_weight_decay is None else optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay),
                ),
                "target_safety_critic": TrainState.create(
                    apply_fn=critic_def.apply,
                    params=safety_critic_params,
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
            pixel_embeddings_key=pixel_embeddings_key,
            **make_train_states(rng),
            safety_threshold=safety_threshold,
            safety_discount=safety_discount,
        )

    def observation_keys(self):
        if self.pixel_embeddings_key is not None:
            return {"states", self.pixel_embeddings_key}
        else:
            return "states"

    def reset(
        self, ensemble_idx: Optional[int] = None, reset_actor: bool = True
    ) -> "SafetyCriticSACLearner":
        rng, key = jax.random.split(self.rng)
        train_states = self.initialize_params(key)
        del train_states["temp"]

        new_actor = train_states["actor"] if reset_actor else self.actor
        new_critic = train_states["critic"]
        new_target_critic = train_states["target_critic"]

        return self.replace(
            rng=rng,
            actor=new_actor.replace(
                params=_reset_params(
                    self.actor.params, new_actor.params, ensemble_idx=ensemble_idx
                )
            ),
            critic=new_critic.replace(
                params=_reset_params(
                    self.critic.params, new_critic.params, ensemble_idx=ensemble_idx
                )
            ),
            target_critic=new_target_critic.replace(
                params=_reset_params(
                    self.target_critic.params,
                    new_target_critic.params,
                    ensemble_idx=ensemble_idx,
                )
            ),
        )

    def update_actor(
        self,
        batch: FrozenDict[str, jax.Array],
        output_range: Optional[Tuple[jax.Array, jax.Array]],
    ) -> Tuple["SafetyCriticSACLearner", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jax.Array, Dict[str, float]]:
            dist: Distribution = self.actor.apply_fn(
                {"params": actor_params},
                batch["observations"],
                output_range=output_range,
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean(),
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * jnp.mean(entropy - self.target_entropy)
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(
        self,
        batch: DatasetDict,
        output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> Tuple["SafetyCriticSACLearner", Dict[str, float]]:
        rng = self.rng

        dist: Distribution = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"],
            output_range=output_range,
        )

        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs: jax.Array = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        chex.assert_equal_shape([target_q, next_q, batch["rewards"], batch["masks"]])

        if self.backup_entropy:
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            target_q_ = target_q[None].repeat(qs.shape[0], axis=0)
            chex.assert_equal_shape([qs, target_q_])
            critic_loss = ((qs - target_q_) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": jnp.mean(qs),
                "q_std": jnp.std(qs, axis=0).mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @jax.jit
    def sample_actions(
        self,
        observation: Dict[str, jax.Array],
        output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jax.Array, "SafetyCriticSACLearner"]:
        if isinstance(observation, (dict, FrozenDict)):
            observation = observation["states"]
        
        agent = self

        rng, key_sample_actions, key_rejection_sampling = jax.random.split(self.rng, 3)

        # Rejection sampling
        distribution = self.actor.apply_fn(
            {"params": self.actor.params},
            observation,
            output_range=output_range,
        )
        actions = distribution.sample(seed=key_sample_actions, sample_shape=(100,))
        unsafety_logits = self.safety_critic.apply_fn({"params": self.safety_critic.params}, observation[None].repeat(100, axis=0), actions, True).mean(axis=0)
        unsafety_probs = jax.nn.sigmoid(unsafety_logits) # HACK

        # If any are safe, sample uniformly from those
        adj_unsafety_logits = jnp.where(
            unsafety_probs < self.safety_threshold,
            -1000,
            unsafety_logits,
        )
        safety_logits = -adj_unsafety_logits

        idx = jax.random.categorical(key_rejection_sampling, safety_logits)

        jax.lax.cond(
            unsafety_probs[idx] < self.safety_threshold,
            lambda _: None,
            lambda s: jax.debug.print("No valid action found, best safety: {s}", s=s),
            unsafety_probs[idx],
        )
        return actions[idx], agent.replace(rng=rng)

    def update_safety_critic(
        self,
        batch: DatasetDict,
        output_range: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> Tuple["SafetyCriticSACLearner", Dict[str, float]]:
        rng = self.rng

        dist: Distribution = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"],
            output_range=output_range,
        )

        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key, sample_shape=(10,))

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_safety_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs: jax.Array = self.target_safety_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"][None].repeat(10, axis=0),
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_qs = jax.nn.sigmoid(next_qs) # HACK
        # MAX INSTEAD OF MIN (BECAUSE LOW SAFETY ERROR IS GOOD)
        # Max along ensemble
        next_qs = next_qs.max(axis=0)
        # Min along sampled actions
        assert next_qs.shape[0] == 10
        beta = 5
        next_q = (-jax.scipy.special.logsumexp(-beta*next_qs, axis=0) + jnp.log(10))/beta

        target_q = batch["safety"] + self.safety_discount * batch["masks"] * next_q

        chex.assert_shape([next_q, target_q], (batch["next_observations"].shape[0],))

        if self.backup_entropy:
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def safety_critic_loss_fn(safety_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.safety_critic.apply_fn(
                {"params": safety_critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            qs = jax.nn.sigmoid(qs) # HACK
            target_q_ = target_q[None].repeat(qs.shape[0], axis=0)
            chex.assert_equal_shape([qs, target_q_])
            safety_critic_loss = ((qs - target_q_) ** 2).mean()
            return safety_critic_loss, {
                "safety_critic_loss": safety_critic_loss,
                "safety_critic_q": jnp.mean(qs),
            }

        grads, info = jax.grad(safety_critic_loss_fn, has_aux=True)(self.safety_critic.params)
        safety_critic = self.safety_critic.apply_gradients(grads=grads)

        target_safety_critic_params = optax.incremental_update(
            safety_critic.params, self.target_safety_critic.params, self.tau
        )
        target_safety_critic = self.target_safety_critic.replace(params=target_safety_critic_params)

        return self.replace(safety_critic=safety_critic, target_safety_critic=target_safety_critic, rng=rng), info

    @partial(jax.jit, static_argnames=("utd_ratio", "update_temperature"))
    def update(
        self,
        batch: DatasetDict,
        utd_ratio: int,
        update_temperature: bool = True,
        output_range: Optional[Tuple[jax.Array]] = None,
        output_range_next: Optional[Tuple[jax.Array]] = None,
    ):
        new_agent = self
        # Setup
        batch_size = batch["actions"].shape[0]
        action_dim = batch["actions"].shape[-1]

        def fix_output_range(x):
            if x is None:
                x = (-jnp.ones(()), jnp.ones(()))

            if len(x[0].shape) == 0:
                x = jax.tree_map(lambda y: jnp.repeat(y[None], action_dim, axis=0), x)

            if len(x[0].shape) == 1:
                x = jax.tree_map(lambda y: jnp.repeat(y[None], batch_size, axis=0), x)

            return x

        output_range = fix_output_range(output_range)
        output_range_next = fix_output_range(output_range_next or output_range)

        chex.assert_shape(output_range, (batch_size, action_dim))

        # Critic update
        minibatch_size = batch_size // utd_ratio
        def reshape_first_dim(x: jax.Array):
            return jnp.reshape(x, (utd_ratio, minibatch_size, *x.shape[1:]))
        batched_batch = jax.tree_map(reshape_first_dim, batch)
        batched_output_range = jax.tree_map(reshape_first_dim, output_range)
        batched_output_range_next = jax.tree_map(reshape_first_dim, output_range_next)

        def update_step(agent: SafetyCriticSACLearner, data: Tuple[DatasetDict, Tuple[jax.Array, jax.Array]]) -> SafetyCriticSACLearner:
            batch, next_output_range = data
            new_agent, critic_info = agent.update_critic(batch, output_range=next_output_range)
            new_agent, safety_critic_info = new_agent.update_safety_critic(batch, output_range=next_output_range)
            return new_agent, (critic_info, safety_critic_info)

        new_agent, (critic_infos, safety_critic_infos) = jax.lax.scan(
            update_step,
            new_agent,
            (batched_batch, batched_output_range_next),
        )

        critic_info = jax.tree_map(
            lambda data: jnp.mean(data, axis=0), critic_infos
        )
        safety_critic_info = jax.tree_map(
            lambda data: jnp.mean(data, axis=0), safety_critic_infos
        )

        # Only update the actor on one minibatch
        new_agent, actor_info = new_agent.update_actor(
            jax.tree_map(lambda x: x[0], batched_batch),
            jax.tree_map(lambda x: x[0], batched_output_range)
        )
        new_agent, safety_critic_info = new_agent.update_safety_critic(
            jax.tree_map(lambda x: x[0], batched_batch),
            jax.tree_map(lambda x: x[0], batched_output_range)
        )

        if update_temperature:
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        else:
            temp_info = {}

        return new_agent, {**actor_info, **critic_info, **temp_info, **safety_critic_info}
