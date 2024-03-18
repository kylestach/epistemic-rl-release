#! /usr/bin/env python
import os
import pickle

import collections
from typing import Tuple
import gym
from gym.wrappers import TimeLimit, RecordEpisodeStatistics
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
import numpy as np
import quaternion as npq
import moviepy.editor
from PIL import Image, ImageDraw
import procedural_driving
from jax import numpy as jnp

from jaxrl5.agents import *
from jaxrl5.data import ReplayBuffer

# Offroad stuff
import warnings

warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_name", "procedural_driving/CarTask-states-v0", "Environment name."
)
flags.DEFINE_string("wandb_project", "offroad_procedural_states", "Project for W&B")
flags.DEFINE_string("comment", "", "Comment for W&B")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string(
    "expert_replay_buffer", "", "(Optional) Expert replay buffer pickle file."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string(
    "world_name",
    "default",
    "World name, corresponds to a yaml file in procedural_driving/worlds",
)
flags.DEFINE_integer("eval_episodes", 16, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 100, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 128, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("replay_buffer_size", 100000, "Capacity of the replay buffer.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
flags.DEFINE_integer("save_buffer_interval", 50000, "Save buffer interval.")
flags.DEFINE_integer("utd_ratio", 8, "Updates per data point")
flags.DEFINE_integer(
    "reset_interval",
    None,
    "Parameter reset interval, in network updates (= env steps * UTD)",
)
flags.DEFINE_enum(
    "ramp_action",
    None,
    ["linear", "step"],
    "Should the max action be ramped up?",
    required=False,
)
flags.DEFINE_float("action_penalty_start", None, "Start value for ramping up action", required=False)
flags.DEFINE_float("action_penalty_end", 0.0, "End value for ramping up action")
flags.DEFINE_boolean("reset_ensemble", False, "Reset one ensemble member at a time")
config_flags.DEFINE_config_file(
    "config",
    "configs/redq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string("group_name_suffix", None, "Group name suffix")

flags.DEFINE_boolean("unstable_car", False, "Use the unstable car model")

def safety_reward_fn(next_obs):
    quat = next_obs["car/body_rotation"]
    quat = npq.as_quat_array(quat)
    up_world = np.array([0, 0, 1])
    up_local = npq.as_vector_part(
        quat * npq.from_vector_part(up_world) * np.conjugate(quat)
    )
    error = np.linalg.norm(up_local - up_world, axis=-1) / np.sqrt(2)
    dot_product = np.sum(up_local * up_world, axis=-1)

    # assert dot_product.shape == (), dot_product.shape

    rewards_smooth = -error
    # Smoothly interpolate from -1 to 0 as dot product goes from 0 to 1, using a cosine function
    # rewards_smooth = (-1 - np.cos(dot_product * np.pi)) / 2
    rewards = np.where(
        dot_product < 0,
        -1,
        0*rewards_smooth,
    )

    return rewards


def run_trajectory(agent, env: gym.Env, max_steps=1000, video: bool = False, output_range: Tuple[float, float] = None):
    obs, _ = env.reset()
    if hasattr(agent, "env_reset"):
        agent = agent.env_reset(obs["states"])

    images = []
    episode_return = 0
    episode_length = 0

    for _ in range(max_steps):
        action, agent = agent.sample_actions(obs["states"], output_range=output_range)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, done, truncated, info = env.step(action)

        # safety_bonus = safety_reward_fn(obs)
        # reward += safety_bonus

        if video:
            image = env.render(camera_id="car/overhead_track", width=100, height=100)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            goal_vector = obs["goal_absolute"][:2] - obs["car/body_pose_2d"][:2]
            a = -np.rad2deg(np.arctan2(goal_vector[1], goal_vector[0]))
            d = np.linalg.norm(goal_vector)
            draw.arc(
                (5, 5, 95, 95),
                start=a - (100 + 2 * d) / (0.01 + d),
                end=a + (100 + 2 * d) / (0.01 + d),
                fill=(0, 255, 0),
                width=4,
            )
            image = np.asarray(image)
            image = np.transpose(image, (2, 0, 1))
            images.append(image)

        episode_return += reward
        episode_length += 1

        if done or truncated:
            break

    return images, episode_return, episode_length


def evaluate(agent, env: gym.Env, num_episodes: int, output_range: Tuple[float, float]):
    episode_videos = []
    episode_returns = []
    episode_lengths = []
    for i in range(num_episodes):
        images, episode_return, episode_length = run_trajectory(agent, env, video=i<4, output_range=output_range)
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        if i < 4:
            episode_videos.append(images)

    max_video_length = max([len(v) for v in episode_videos])
    episode_videos = [
        np.stack(v + [v[-1]] * (max_video_length - len(v))) for v in episode_videos
    ]

    wandb.log(
        {
            "evaluation/episode_returns_histogram": wandb.Histogram(episode_returns),
            "evaluation/episode_return": np.mean(episode_returns),
            "evaluation/episode_length_histogram": wandb.Histogram(episode_lengths),
            "evaluation/episode_length": np.mean(episode_lengths),
            "evaluation/videos": wandb.Video(
                np.stack(episode_videos), fps=10, format="mp4"
            ),
        }
    )


def max_action_schedule(i):
    return min(1, 3 * i / FLAGS.max_steps - 0.5)


def main(_):
    ## offroad_env stuff
    env = gym.make(FLAGS.env_name, world_seed=0, world_name=FLAGS.world_name, use_alt_car=FLAGS.unstable_car)
    env = TimeLimit(env, max_episode_steps=100000)
    env = RecordEpisodeStatistics(env, deque_size=1)

    eval_env = gym.make(FLAGS.env_name, world_seed=1, world_name=FLAGS.world_name, use_alt_car=FLAGS.unstable_car)
    eval_env = TimeLimit(eval_env, max_episode_steps=300)
    ## offroad_env stuff

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")

    safety_bonus_coeff = kwargs.pop("safety_penalty", 0.0)
    wandb_group_name = kwargs.pop("group_name", "???")

    agent: SACLearner = globals()[model_cls].create(
        FLAGS.seed, env.observation_space["states"], env.action_space, **kwargs
    )

    if FLAGS.action_penalty_start and FLAGS.action_penalty_start < FLAGS.action_penalty_end:
        warnings.warn(
            "Action penalty ramp is backwards. This is probably not what you want."
        )

    if FLAGS.expert_replay_buffer:
        with open(FLAGS.expert_replay_buffer, "rb") as f:
            expert_replay_buffer = pickle.load(f)

    replay_buffer_size = FLAGS.replay_buffer_size
    replay_buffer = ReplayBuffer(
        env.observation_space["states"], env.action_space, replay_buffer_size, extra_fields=["safety"]
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
        }
    )
    if FLAGS.expert_replay_buffer:
        expert_replay_buffer_iterator = expert_replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
            }
        )

    observation, done = env.reset()

    wandb.init(
        project=FLAGS.wandb_project, notes=FLAGS.comment, group=f"{wandb_group_name}-{FLAGS.world_name}" if FLAGS.group_name_suffix is None else f"{wandb_group_name}-{FLAGS.group_name_suffix}-{FLAGS.world_name}"
    )
    config_for_wandb = {
        **FLAGS.flag_values_dict(),
        "config": dict(FLAGS.config),
    }
    wandb.config.update(config_for_wandb)

    task = env.unwrapped._env._task

    start_positions = [tuple(observation["car/body_pose_2d"][:2]) + (0,)]
    observation = observation["states"]

    reset_interval = FLAGS.reset_interval
    if FLAGS.reset_ensemble:
        reset_interval = reset_interval // agent.num_qs

    action_min: np.ndarray = env.action_space.low
    action_max: np.ndarray = env.action_space.high

    speed_ema = 0.0
    safety_ema = 0.0

    ema_beta = 3e-4

    if FLAGS.ramp_action:
        action_max[1] = max_action_schedule(0)

    pbar = tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm, dynamic_ncols=True
    )
    for i in pbar:
        if hasattr(agent, "target_entropy"):
            agent = agent.replace(target_entropy=agent.target_entropy - 2e-5)
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            result = agent.sample_actions(observation, output_range=(action_min, action_max), debug=True)
            if len(result) == 3:
                action, agent, action_info = result
            else:
                action, agent = result
                action_info = {}
            action = np.clip(action, env.action_space.low, env.action_space.high)

        if FLAGS.ramp_action == "linear":
            action_max[1] = max_action_schedule(i)

        next_observation, reward, done, truncated, info = env.step(action)

        safety_bonus = safety_reward_fn(next_observation)
        reward += safety_bonus * safety_bonus_coeff

        speed_ema = (1-ema_beta) * speed_ema + ema_beta * np.linalg.norm(next_observation["car/body_vel_2d"][:2])
        safety_ema = (1-ema_beta) * safety_ema + ema_beta * safety_bonus

        next_observation = next_observation["states"]

        if not done or truncated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
                safety=-safety_bonus,
            )
        )

        observation = next_observation

        if done or truncated:
            observation, done = env.reset()
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb_log = {f"training/{decode[k]}": v}
                start_positions.append(
                    tuple(observation["car/body_pose_2d"][:2]) + (len(start_positions),)
                )
                start_pos_table = wandb.Table(
                    data=start_positions, columns=["x", "y", "t"]
                )
                wandb_log["start_pos"] = wandb.plot.scatter(
                    start_pos_table, "x", "y", title="Start positions"
                )
                if FLAGS.ramp_action:
                    wandb_log["training/action_max"] = action_max[1]
                wandb.log(wandb_log, step=i)
            observation = observation["states"]

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)

            output_range = (
                jnp.repeat(action_min[None, :], FLAGS.utd_ratio*FLAGS.batch_size, axis=0),
                jnp.repeat(action_max[None, :], FLAGS.utd_ratio*FLAGS.batch_size, axis=0),
            )

            agent, update_info = agent.update(batch, utd_ratio=FLAGS.utd_ratio, output_range=output_range)

            if hasattr(agent, "update_safety"):
                agent, safety_info = agent.update_safety(-safety_ema)
                update_info = {**update_info, **safety_info}

            def get_logged_value(k, v):
                if k.endswith("_hist"):
                    probs, atoms = v
                    atoms = np.asarray(atoms)
                    probs = np.asarray(probs)
                    return wandb.Histogram(
                        np_histogram=np.histogram(atoms, weights=probs, bins=atoms)
                    )
                elif np.prod(v.shape) == 1:
                    return v.item()
                else:
                    return wandb.Histogram(list(np.asarray(v)))

            update_info = update_info | action_info
            update_info = {k: get_logged_value(k, v) for k, v in update_info.items()}

            if FLAGS.action_penalty_start:
                action_penalty = max(0, FLAGS.action_penalty_start + (FLAGS.action_penalty_end - FLAGS.action_penalty_start) * i / FLAGS.max_steps * 2)
                agent = agent.replace(action_penalty=action_penalty)
                update_info["action_penalty"] = action_penalty

            update_info_expert = {}
            if FLAGS.expert_replay_buffer:
                batch_expert = next(expert_replay_buffer_iterator)
                agent, update_info_expert = agent.update(
                    batch_expert,
                    utd_ratio=FLAGS.utd_ratio,
                    update_temperature=False,
                    output_range=output_range
                )

            if i % FLAGS.log_interval == 0:
                wandb_log = {
                    "training/running_return": info["running_return"],
                }
                if hasattr(agent, "target_entropy"):
                    wandb_log["training/target_entropy"] = float(agent.target_entropy)

                wandb_log["num_flips"] = task.num_flips
                wandb_log["num_timeouts"] = task.num_timeouts
                wandb_log["num_stuck"] = task.num_stuck
                for k, v in update_info.items():
                    wandb_log.update({f"training/{k}": v})
                for k, v in update_info_expert.items():
                    wandb_log.update({f"training/expert/{k}": v})
                wandb_log["speed_ema"] = speed_ema
                wandb_log["safety_ema"] = safety_ema
                wandb.log(wandb_log, step=i)

        if i > 1 and i % FLAGS.save_buffer_interval == 0:
            if FLAGS.save_buffer:
                dataset_folder = os.path.join("datasets", FLAGS.env_name)
                os.makedirs(dataset_folder, exist_ok=True)
                dataset_file = os.path.join(dataset_folder, f"{wandb.run.name}_{i}.pkl")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)

        if i % FLAGS.eval_interval == 0 or i == 100:
            """
            try:
                policy_folder = os.path.join("policies", wandb.run.name)
                os.makedirs(policy_folder, exist_ok=True)
                param_dict = {
                    "actor": agent.actor,
                    "critic": agent.critic,
                    "target_critic_params": agent.target_critic,
                    "temp": agent.temp,
                    "rng": agent.rng,
                }
                if hasattr(agent, "limits"):
                    param_dict["limits"] = agent.limits
                if hasattr(agent, "q_entropy_lagrange"):
                    param_dict["q_entropy_lagrange"] = agent.q_entropy_lagrange
                checkpoints.save_checkpoint(policy_folder, param_dict, step=i, keep=10)
            except Exception as e:
                print(f"Cannot save checkpoints: {e}")
            """

            # evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes, output_range=(action_min, action_max))
            pass

        if (
            reset_interval
            and (i - 1) % (reset_interval // FLAGS.utd_ratio) == 0
            and i > 1
        ):
            if FLAGS.ramp_action == "step":
                action_max[1] = max_action_schedule(i)

            if FLAGS.reset_ensemble:
                agent = agent.reset(
                    ensemble_idx=next_ensemble_member, reset_actor=False
                )
                next_ensemble_member = (next_ensemble_member + 1) % agent.num_qs
            else:
                agent = agent.reset()


if __name__ == "__main__":
    app.run(main)
