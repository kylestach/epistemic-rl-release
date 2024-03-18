import warnings
from typing import Union
import numpy as np
from procedural_driving import make_car_task_gym, make_car_task_gym_states
from jaxrl5.agents import SACLearner, DrQLearner
import jax
from ml_collections import config_flags
from flax.training.checkpoints import restore_checkpoint
from tqdm import trange
from pathlib import Path
from PIL import Image, ImageDraw
import os
from flax.core.frozen_dict import FrozenDict
from moviepy.editor import ImageSequenceClip

from procedural_driving.task import CarTask
from dm_control import composer
from dm_control.composer import Environment

from absl import app, flags
import cv2
from moviepy.editor import ImageSequenceClip

jax.config.update("jax_platform_name", "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_string("world_name", "default", "World name")
flags.DEFINE_string("video_output", None, "Path to the video output directory")

warnings.filterwarnings("ignore")


def render_fn(env, observation):
    view_a = env.physics.render(
        camera_id="car/offroad_realsense_d435i", width=320, height=320
    )
    view_b = env.physics.render(camera_id="car/overhead_track", width=320, height=320)
    view = np.concatenate([view_a, view_b], axis=1)
    view = Image.fromarray(view)
    draw = ImageDraw.Draw(view)
    a = np.rad2deg(
        np.arctan2(-observation["goal_relative"][0], -observation["goal_relative"][1])
    )
    d = np.linalg.norm(
        observation["goal_absolute"] - observation["car/body_pose_2d"][:2]
    )
    draw.arc(
        (0, 0, 320, 320),
        start=a - (100 + 2 * d) / (0.01 + d),
        end=a + (100 + 2 * d) / (0.01 + d),
        fill=(0, 255, 0),
        width=10,
    )
    state = observation
    #draw.text(
        #(10, 10),
        #f"Position: {state['car/body_pose_2d'][:2]}, Velocity: {state['car/body_vel_2d'][:2]}",
        #fill=(255, 255, 255),
    #)
    return view


def get_action_from_key(key, last_action, action_space):
    # Escape
    if key == 27:
        return None

    action = last_action.copy()

    if key == 82:
        if action[1] >= 0.0:
            action[1] += 0.1 * 5
        else:
            action[1] = 0.0
    elif key == 84:
        if action[1] <= 0.0:
            action[1] -= 0.1 * 5
        else:
            action[1] = 0.0
    elif key == 83:
        if action[0] <= 0.0:
            action[0] -= 0.2 * 0.38
        else:
            action[0] = 0.0
    elif key == 81:
        if action[0] >= 0.0:
            action[0] += 0.2 * 0.38
        else:
            action[0] = 0.0

    action = np.clip(action, action_space.minimum, action_space.maximum)
    return action


def run_manual(env: Environment):
    # Run the simulation
    obs = env.reset().observation

    images = []
    running = True
    action = np.zeros(2)
    for i in trange(300):
        for _ in range(3):
            k = cv2.waitKey(33)

            action = get_action_from_key(k, action, env.action_spec())
            if action is None:
                running = False
                break

            ts = env.step(action)
            obs = ts.observation

            if ts.last():
                obs = env.reset().observation

            # Render the simulation
            view = render_fn(env, obs)
            images.append(np.asarray(view))
            view = np.flip(np.asarray(view), axis=-1)

            cv2.imshow("view", view)

        if not running:
            break

    if FLAGS.video_output is not None:
        clip = ImageSequenceClip(images, fps=30)
        clip.write_videofile(FLAGS.video_output)


def main(_):
    # Run at 3x FPS (30 rather than 10) for visualization
    task = CarTask(
        control_timestep=0.1 / 3,
        physics_timestep=0.001 / 3,
        world_seed=1,
        use_alt_car=False,
        world_name=FLAGS.world_name,
    )
    env = composer.Environment(
        task,
        raise_exception_on_physics_error=False,
        strip_singleton_obs_buffer_dim=True,
    )
    run_manual(env)


if __name__ == "__main__":
    app.run(main)
