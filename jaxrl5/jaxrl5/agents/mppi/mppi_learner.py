"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Callable, List

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax import linen as nn
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
)


class DynamicsModel(nn.Module):
    model: Callable[[], nn.Module]

    def obs_to_state(
        self, obs: jax.Array, next_obs: Optional[jax.Array] = None
    ) -> jax.Array:
        if next_obs is None:
            return obs
        else:
            return obs, next_obs

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([state, action], axis=-1)
        x = self.model()(x)

        d_state = nn.Dense(state.shape[-1])(x)
        reward = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        is_terminal_logits = jnp.squeeze(nn.Dense(1)(x), axis=-1)

        return state + d_state, reward, is_terminal_logits


class CarDynamicsModel2D(nn.Module):
    model: Callable[[], nn.Module]

    def obs_to_state(
        self, obs: jax.Array, next_obs: Optional[jax.Array] = None
    ) -> jax.Array:
        from procedural_driving import STATES_STATES_KEYS

        assert STATES_STATES_KEYS == [
            "goal_relative",
            "car/sensors_vel",
            "car/sensors_gyro",
            "car/body_down_vector",
        ]
        chex.assert_shape(obs, [..., 3 + 3 + 3 + 3])
        goal_relative = obs[..., :3]
        velocity = obs[..., 3:6]
        angular_velocity = obs[..., 6:9]
        down_vector = obs[..., 9:12]

        yaw = jnp.zeros_like(goal_relative[..., 2])
        goal_position = goal_relative[..., :2] * goal_relative[..., 2:3]
        position = jnp.zeros_like(goal_position)
        linear_velocity = velocity[..., :2]
        angular_velocity = angular_velocity[..., 2]

        state = {
            "position": position,
            "yaw": yaw,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "goal": goal_position,
        }

        if next_obs is None:
            return state

        next_goal_relative = next_obs[..., :3]
        next_velocity = next_obs[..., 3:6]
        next_angular_velocity = next_obs[..., 6:9]
        next_down_vector = next_obs[..., 9:12]

        ACTUAL_DT = 0.1

        next_state = {
            "position": position + linear_velocity * ACTUAL_DT,
            "yaw": yaw + angular_velocity * ACTUAL_DT,
            "linear_velocity": next_velocity[..., :2],
            "angular_velocity": next_angular_velocity[..., 2],
            "goal": goal_position,
        }

        return state, next_state

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array):
        position = state["position"]
        yaw = state["yaw"]
        linear_velocity = state["linear_velocity"]
        angular_velocity = state["angular_velocity"]

        dt = jnp.exp(self.param("log_dt", lambda *args: jnp.log(0.01), ()))

        # Predict
        x = jnp.concatenate([linear_velocity, angular_velocity[..., None]], axis=-1)
        d_vel = nn.Dense(2)(x)
        d_avel = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        is_terminal_logits = jnp.squeeze(nn.Dense(1)(x), axis=-1)

        # Compute reward
        to_goal_vec = state["goal"] - state["position"]
        to_goal_vec /= jnp.linalg.norm(to_goal_vec, axis=-1, keepdims=True) + 1e-3
        to_goal_vel = jnp.sum(
            to_goal_vec[..., :2] * state["linear_velocity"][..., :2], axis=-1
        )

        new_position = (
            position
            + jnp.stack(
                [
                    linear_velocity[..., 0] * jnp.cos(yaw)
                    - linear_velocity[..., 1] * jnp.sin(yaw),
                    linear_velocity[..., 0] * jnp.sin(yaw)
                    + linear_velocity[..., 1] * jnp.cos(yaw),
                ],
                axis=-1,
            )
            * dt
        )

        return (
            {
                "position": new_position,
                "yaw": yaw + angular_velocity * dt,
                "linear_velocity": linear_velocity + d_vel * dt,
                "angular_velocity": angular_velocity + d_avel * dt,
                "goal": state["goal"],
            },
            to_goal_vel,
            is_terminal_logits,
        )


class CarDynamicsModel(nn.Module):
    model: Callable[[], nn.Module]

    def obs_to_state(
        self, obs: jax.Array, next_obs: Optional[jax.Array] = None
    ) -> jax.Array:
        from procedural_driving import STATES_STATES_KEYS

        assert STATES_STATES_KEYS == [
            "goal_relative",
            "car/sensors_vel",
            "car/sensors_gyro",
            "car/body_down_vector",
        ]
        chex.assert_shape(obs, [..., 3 + 3 + 3 + 3])
        goal_relative = obs[..., :3]
        velocity = obs[..., 3:6]
        angular_velocity = obs[..., 6:9]
        down_vector = obs[..., 9:12]

        # Construct a rotation matrix with the down vector as the z-axis and otherwise axis-aligned
        # This is the inverse of the rotation matrix that would rotate the z-axis to the down vector
        up_vector = -down_vector
        left_vector = jnp.broadcast_to(jnp.array([0, 1, 0]), down_vector.shape)
        forward_vector = jnp.cross(left_vector, up_vector)
        rotation_matrix = jnp.stack([forward_vector, left_vector, up_vector], axis=-1)

        # Rotate the goal relative vector into the local frame
        # Extract yaw from the rotation matrix
        goal_position = goal_relative[..., :2] * goal_relative[..., 2:3]
        position = jnp.zeros_like(goal_position)

        state = {
            "position": position,
            "orientation": rotation_matrix,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "goal": goal_position,
        }

        if next_obs is None:
            return state

        ACTUAL_DT = 0.1

        next_state = {
            "position": position + velocity * ACTUAL_DT,
            "orientation": rotation_matrix,
        }

        return state, next_state

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array):
        down_vector = state["orientation"][..., 2, :]
        x = jnp.concatenate(
            [state["velocity"], state["angular_velocity"], state["down_vector"]]
        )
        x = self.model()(x)

        d_vel = nn.Dense(3)(x)
        d_ang_vel = nn.Dense(3)(x)

        dt = jnp.exp(self.param("log_dt", lambda *args: jnp.log(0.01), ()))

        # Compute rotation from axis-angle
        rotation = d_ang_vel * dt
        theta = jnp.linalg.norm(rotation, axis=-1, keepdims=True)
        axis = rotation / theta
        cth = jnp.cos(theta)
        rotation_matrix = (
            axis[:, None] * axis[None, :] * (1 - cth)
            + jnp.eye(3) * cth
            + jnp.cross(axis[None, :], axis[:, None]) * jnp.sin(theta)
        )

        # Compute reward
        to_goal_vec = state["goal"] - state["position"]
        to_goal_vec /= jnp.linalg.norm(to_goal_vec, axis=-1, keepdims=True)
        to_goal_vel = jnp.sum(
            to_goal_vec[..., :2] * state["velocity"][..., :2], axis=-1
        )

        # Compute is_terminal
        is_terminal_logits = jnp.squeeze(nn.Dense(1)(x), axis=-1)

        # Batch matmul
        batch_dims = state["position"].shape[:-1]
        if len(batch_dims) == 0:
            new_orientation = jnp.matmul(state["orientation"], rotation_matrix)
        else:
            new_orientation = jax.vmap(jnp.matmul)(
                state["orientation"], rotation_matrix
            )

        return (
            {
                "position": state["position"] + state["velocity"] * dt,
                "orientation": new_orientation,
                "velocity": state["velocity"] + d_vel * dt,
                "angular_velocity": state["angular_velocity"] + d_ang_vel * dt,
                "goal": state["goal"],
            },
            to_goal_vel,
            is_terminal_logits,
        )


def do_rollout(
    model_fn: callable,
    rng: jax.random.PRNGKey,
    init_state: jax.Array,
    action_sequence: jax.Array,
    horizon: int,
    discount: float,
    terminal_value: float,
) -> jax.Array:
    # chex.assert_rank(init_state, 1)
    chex.assert_shape(action_sequence, (horizon, None))

    def step_fn(carry, action):
        state, rng = carry
        next_state, reward, is_terminal_logits = model_fn(state, action)
        is_terminal = jax.nn.sigmoid(is_terminal_logits)

        # if next_state.ndim == 2:
        #     # Sample from ensemble
        #     rng, key = jax.random.split(rng)
        #     next_state = next_state[
        #         jax.random.randint(key, shape=(), minval=0, maxval=next_state.shape[0])
        #     ]
        #     reward = reward.mean()
        #     is_terminal = jnp.mean(is_terminal)

        return (next_state, rng), (next_state, reward, is_terminal)

    _, (states, rewards, terminals) = jax.lax.scan(
        step_fn, (init_state, rng), action_sequence, length=horizon
    )
    # chex.assert_shape(states, (horizon, *init_state.shape))
    chex.assert_shape(rewards, (horizon,))

    def _compute_cumulative_returns(carry, rew_and_term):
        reward, is_terminal = rew_and_term
        next_value = carry * (1 - is_terminal) + is_terminal * terminal_value
        return reward + discount * next_value, None

    cumulative_returns, _ = jax.lax.scan(
        _compute_cumulative_returns,
        0.0,
        (rewards, terminals),
        reverse=True,
    )

    return states, cumulative_returns


def do_rollouts(
    model_fn: callable,
    rng: jax.random.PRNGKey,
    init_state: jax.Array,
    action_sequences: jax.Array,
    *,
    horizon: int,
    num_rollouts: int,
    action_dim: int,
    discount: float,
    terminal_value: float,
) -> jax.Array:
    # chex.assert_rank(init_state, 1)
    chex.assert_shape(action_sequences, (num_rollouts, horizon, action_dim))

    return jax.vmap(
        lambda action_sequence, rng: do_rollout(
            model_fn=model_fn,
            rng=rng,
            init_state=init_state,
            action_sequence=action_sequence,
            horizon=horizon,
            discount=discount,
            terminal_value=terminal_value,
        ),
    )(action_sequences, jax.random.split(rng, num_rollouts))


do_batch_rollouts = jax.vmap(do_rollouts, in_axes=(None, 0, 0, 0), out_axes=0)


def compute_action_sequences(
    action_sequences: jax.Array,
    rewards: jax.Array,
    beta: float,
    horizon: int,
):
    num_rollouts = action_sequences.shape[0]
    chex.assert_shape(action_sequences, (num_rollouts, horizon, None))
    chex.assert_shape(rewards, (num_rollouts,))

    # Standardize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-3)

    weights = jax.nn.softmax(beta * rewards)
    return jnp.sum(weights[:, None, None] * action_sequences, axis=0)


batch_compute_action_sequences = jax.vmap(
    compute_action_sequences, in_axes=(0, 0, None, None), out_axes=0
)


def do_mppi(
    model_fn: callable,
    init_state: jax.Array,
    last_actions: jax.Array,
    rng: jax.random.PRNGKey,
    *,
    horizon: int,
    num_rollouts: int,
    action_dim: int,
    discount: float,
    terminal_value: float,
    sampling_std: float,
    debug: bool = False,
    beta: float = 1.0,
):
    # Generate action sequences
    action_dim = last_actions.shape[-1]
    chex.assert_shape(last_actions, (horizon, action_dim))
    # chex.assert_rank(init_state, 1)

    rng, key = jax.random.split(rng)
    clip_value = 1 - 1e-3
    action_sequences = nn.tanh(
        jax.random.normal(key, shape=(num_rollouts, horizon, action_dim)) * sampling_std
        + jnp.arctanh(jnp.clip(last_actions[None, :, :], -clip_value, clip_value))
    )

    state_rollouts, returns = do_rollouts(
        model_fn,
        rng,
        init_state,
        action_sequences,
        horizon=horizon,
        num_rollouts=num_rollouts,
        action_dim=action_dim,
        discount=discount,
        terminal_value=terminal_value,
    )

    action_sequence = compute_action_sequences(
        action_sequences, returns, beta=beta, horizon=horizon
    )

    if debug:
        return action_sequence, {
            "action_rollouts": action_sequences,
            "state_rollouts": state_rollouts,
            "rewards": returns,
            "reward_mean": returns.mean(),
            "reward_std": returns.std(),
        }
    else:
        return action_sequence, {
            "reward_mean": returns.mean(),
            "reward_std": returns.std(),
        }


class MPPILearner(Agent):
    dynamics: TrainState = struct.field(pytree_node=True)
    obs_to_state: Callable[[jax.Array], jax.Array] = struct.field(pytree_node=False)

    action_trajectory: jax.Array = struct.field(pytree_node=True)
    action_trajectory_valid: jax.Array = struct.field(pytree_node=True)

    action_dim: int = struct.field(pytree_node=False)
    observation_dim: int = struct.field(pytree_node=False)

    beta: float = struct.field(pytree_node=True)
    discount: float = struct.field(pytree_node=True)

    ema: float = struct.field(pytree_node=True, default=0.5)
    ema_discount: float = struct.field(pytree_node=True, default=1.0)

    horizon: int = struct.field(pytree_node=False, default=5)
    num_rollouts: int = struct.field(pytree_node=False, default=128)
    sampling_std: float = struct.field(pytree_node=False, default=0.3)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        dynamics_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        dynamics_ensemble_size: int = None,
        beta: float = 2.0,
        discount: float = 0.99,
        ema: float = 0.0,
        ema_discount: float = 0.9,
        horizon: int = 5,
        num_rollouts: int = 128,
        sampling_std: float = 0.3,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        dynamics_model = partial(
            # CarDynamicsModel2D,
            DynamicsModel,
            model=partial(
                MLP,
                hidden_dims=hidden_dims,
                activations=nn.relu,
                activate_final=True,
                use_layer_norm=True,
            ),
        )

        if dynamics_ensemble_size is not None:
            dynamics_model = Ensemble(dynamics_model, dynamics_ensemble_size)

        rng, init_key = jax.random.split(jax.random.PRNGKey(seed))
        dynamics_model: DynamicsModel = dynamics_model()
        dynamics_params = dynamics_model.init(
            init_key,
            dynamics_model.obs_to_state(jnp.zeros(observation_space.shape[-1])),
            jnp.zeros(action_space.shape[-1]),
        )["params"]
        dynamics_train_state = TrainState.create(
            apply_fn=dynamics_model.apply,
            params=dynamics_params,
            tx=optax.adam(dynamics_lr),
        )

        return cls(
            actor=None,
            action_trajectory=jnp.zeros((horizon, action_space.shape[-1])),
            action_trajectory_valid=jnp.zeros(()),
            dynamics=dynamics_train_state,
            obs_to_state=dynamics_model.obs_to_state,
            rng=rng,
            action_dim=action_space.shape[-1],
            observation_dim=observation_space.shape[-1],
            beta=beta,
            discount=discount,
            ema=ema,
            ema_discount=ema_discount,
            horizon=horizon,
            num_rollouts=num_rollouts,
            sampling_std=sampling_std,
        )

    def observation_keys(self):
        if self.pixel_embeddings_key is not None:
            return {"states", self.pixel_embeddings_key}
        else:
            return "states"

    def update(
        self,
        batch: DatasetDict,
        *,
        utd_ratio=None,
        output_range=None,
    ) -> Tuple["MPPILearner", Dict[str, float]]:
        def loss_fn(params, batch):
            state, next_state = self.obs_to_state(
                batch["observations"], batch["next_observations"]
            )

            (
                predicted_states,
                predicted_rewards,
                predicted_is_terminal_logits,
            ) = self.dynamics.apply_fn(
                {"params": params},
                state,
                batch["actions"],
            )

            rewards = batch["rewards"]
            terminals = 1 - batch["masks"]

            # if predicted_states.ndim == 3:
            #     next_obs = next_obs[None]
            #     rewards = rewards[None]
            #     terminals = terminals[None]

            if isinstance(predicted_states, jax.Array):
                state_mse = jnp.mean((predicted_states - next_state) ** 2)
                state_mse_verbose = {
                    f"verbose/state_{i}_mse": jnp.mean(
                        (predicted_states[..., i] - next_state[..., i]) ** 2
                    )
                    for i in range(predicted_states.shape[-1])
                }
            elif isinstance(predicted_states, Mapping):
                state_mse_verbose = {
                    f"{k}_mse": jnp.mean((predicted_states[k] - next_state[k]) ** 2)
                    for k in predicted_states.keys()
                }
                state_mse = sum(state_mse_verbose.values())
            else:
                raise NotImplementedError

            reward_mse = jnp.mean((predicted_rewards - rewards) ** 2)
            is_terminal_bce = jnp.mean(
                optax.sigmoid_binary_cross_entropy(
                    predicted_is_terminal_logits, terminals
                )
            )

            return state_mse + reward_mse + is_terminal_bce, {
                "state_mse": state_mse,
                "reward_mse": reward_mse,
                "is_terminal_bce": is_terminal_bce,
                **state_mse_verbose,
            }

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(self.dynamics.params, batch)
        dynamics = self.dynamics.apply_gradients(grads=grads)

        return self.replace(dynamics=dynamics), metrics

    def env_reset(
        self,
        observations: np.ndarray,
    ):
        return self.replace(
            action_trajectory=jnp.zeros((self.horizon, self.action_dim)),
            action_trajectory_valid=jnp.zeros(()),
        )

    @partial(jax.jit, static_argnames=("debug",))
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        output_range=None,
        debug: bool = False
    ) -> np.ndarray:
        rng, key = jax.random.split(self.rng)

        def model_fn(state, action):
            return self.dynamics.apply_fn(
                {"params": self.dynamics.params},
                state,
                action,
            )

        action_trajectory = self.action_trajectory
        new_action_trajectory, info = do_mppi(
            model_fn,
            self.obs_to_state(observations),
            action_trajectory,
            key,
            horizon=self.horizon,
            num_rollouts=self.num_rollouts,
            action_dim=self.action_dim,
            discount=self.discount,
            terminal_value=jnp.zeros(()),
            sampling_std=self.sampling_std,
            beta=self.beta,
        )

        ema = self.action_trajectory_valid * self.ema
        ema = ema * self.ema_discount ** jnp.arange(self.horizon)[:, None]
        action_trajectory = ema * action_trajectory + (1 - ema) * new_action_trajectory

        action = action_trajectory[0]
        agent = self.replace(
            rng=rng,
            action_trajectory=jnp.concatenate(
                [action_trajectory[1:], jnp.zeros_like(action[None])], axis=0
            ),
            action_trajectory_valid=jnp.ones(()),
        )

        if debug:
            return action, agent, info
        else:
            return action, agent

    @jax.jit
    def eval_actions(self, observations: np.ndarray, output_range=None) -> np.ndarray:
        return self.sample_actions(observations)
