import warnings
from typing import Union
import gym
from gym import spaces
from gym.wrappers import TimeLimit
import numpy as np
from procedural_driving import make_car_task_gym, make_car_task_gym_states
from jaxrl5.agents import (
    SACLearner,
    DrQLearner,
    DistributionalDrQLearner,
    DistributionalSACLearner,
)
import jax
from ml_collections import config_flags
from flax.training.checkpoints import restore_checkpoint
from dmcgym.env import DMCGYM
from dm_control.composer import Environment as DMCEnvironment
from tqdm import trange
from pathlib import Path
from PIL import Image, ImageDraw
import os
from flax.core.frozen_dict import FrozenDict
from moviepy.editor import ImageSequenceClip

from absl import app, flags

jax.config.update("jax_platform_name", "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_file", None, "Path to the policy file")
flags.DEFINE_string("video_output_dir", None, "Path to the video output directory")
flags.DEFINE_integer("num_trajectories", 1, "Number of trajectories to run")
flags.DEFINE_integer("world_seed", 2, "Random seed")
flags.DEFINE_boolean(
    "pixels", True, "True if the agent is from pixels or False for from states"
)
config_flags.DEFINE_config_file(
    "config_pixels",
    "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_states",
    "configs/redq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


class FilterObs(gym.ObservationWrapper):
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys
        self.observation_space = spaces.Dict(
            {k: env.observation_space[k] for k in keys}
        )

    def observation(self, obs):
        return {k: obs[k] for k in self.keys}


class SelectObs(gym.ObservationWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self.key = key
        self.observation_space = env.observation_space[self.key]

    def observation(self, obs):
        return obs[self.key]


warnings.filterwarnings("ignore")


def load_agent(agent: Union[DrQLearner, SACLearner], policy_file: str):
    """
    Load the agent from a checkpoint.
    """

    param_dict = {
        "actor": agent.actor,
        "critic": agent.critic,
        "target_critic_params": agent.target_critic,
        "temp": agent.temp,
        "rng": agent.rng,
    }
    param_dict = restore_checkpoint(policy_file, target=param_dict)
    return agent.replace(
        actor=param_dict["actor"],
        critic=param_dict["critic"],
        target_critic=param_dict["target_critic_params"],
        temp=param_dict["temp"],
        rng=param_dict["rng"],
    )


def run_trajectory(agent, env: gym.Env, max_steps=5000, render_fn=None):
    """
    Run a trajectory with the agent and render it to a video.
    """

    obs, _ = env.reset()
    images = []
    q_logits_seq = []
    q_atoms_seq = []
    states_seq = []
    action_seq = []
    for i in trange(max_steps):
        action, agent = agent.sample_actions(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        q_logits, q_atoms = agent.critic.apply_fn({"params": agent.critic.params}, obs, action)
        q_logits_seq.append(q_logits)
        q_atoms_seq.append(q_atoms)
        states_seq.append(obs)
        action_seq.append(action)

        for _ in range(3):
            step = env.step(action)
            if len(step) == 5:
                obs, reward, done, truncated, info = step
            else:
                obs, rewawrd, done, info = step
                truncated = False

            if render_fn is not None:
                image = Image.fromarray(render_fn(env))
                draw = ImageDraw.Draw(image)
                state = obs["states"] if FLAGS.pixels else obs
                vel = obs["car/sensors_vel"] if FLAGS.pixels else [0, 0]
                draw.text(
                    (10, 10),
                    f"Goal: {state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}",
                    fill=(255, 255, 255),
                )
                draw.text(
                    (10, 30), f"Vel: {vel[0]:.2f}, {vel[1]:.2f}", fill=(255, 255, 255)
                )
                images.append(np.asarray(image))

            if done or truncated:
                break

    # Save Q-values as npz
    np.savez(
        os.path.join(FLAGS.video_output_dir, "q_values.npz"),
        q_logits=q_logits_seq,
        q_atoms=q_atoms_seq,
        states=states_seq,
        actions=action_seq,
    )

    return images


def render_fn(env: DMCGYM):
    img_a = env.render(camera_id="car/overhead_track", width=480, height=480)
    img_b = env.render(camera_id="car/offroad_realsense_d435i", width=480, height=480)
    img = np.concatenate([img_a, img_b], axis=1)
    return img


def main(_):
    if FLAGS.pixels:
        env_gym = gym.make(
            "procedural_driving/CarTask-v0",
            control_timestep=0.1 / 3,
            physics_timestep=0.001 / 3,
            world_seed=FLAGS.world_seed,
        )
        env_gym = TimeLimit(env_gym, max_episode_steps=1000, new_step_api=True)
        kwargs = dict(FLAGS.config_pixels)
    else:
        env_gym = gym.make(
            "procedural_driving/CarTask-states-v0",
            world_seed=1,
            world_name="hyperflat",
            control_timestep=0.1 / 3,
            physics_timestep=0.001 / 3,
        )
        env_gym = SelectObs(env_gym, "states")
        kwargs = dict(FLAGS.config_states)

    model_cls = kwargs.pop("model_cls")
    agent: SACLearner = globals()[model_cls].create(
        FLAGS.world_seed,
        env_gym.observation_space,
        env_gym.action_space,
        **kwargs,
        # depth_keys=("depth",),
    )

    agent = load_agent(agent, FLAGS.policy_file)
    Path(FLAGS.video_output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(FLAGS.num_trajectories):
        images = run_trajectory(agent, env_gym, render_fn=render_fn)
        ImageSequenceClip(sequence=images, fps=30).write_videofile(
            os.path.join(
                FLAGS.video_output_dir,
                f"video-{os.path.basename(FLAGS.policy_file)}-{i}.mp4",
            )
        )


if __name__ == "__main__":
    app.run(main)