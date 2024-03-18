from copy import deepcopy

import gym
from gym.wrappers.rescale_action import RescaleAction
from dm_control import composer
import numpy as np

from dmcgym.env import DMCGYM

from .task import CarTask
from .wrappers import KeysToStates, PermuteImage, RunningReturnInfo, ReplaceKey


PIXELS_STATES_KEYS = ['goal_relative', 'goal_relative_next', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc', 'car/wheel_speeds', 'car/steering_pos', 'car/steering_vel', 'car/body_down_vector', 'car/sensors_suspension']
# STATES_STATES_KEYS = ['goal_relative', 'goal_relative_next', 'car/sensors_vel', 'car/sensors_gyro', 'car/sensors_acc', 'car/wheel_speeds', 'car/steering_pos', 'car/steering_vel', 'car/body_down_vector', 'car/sensors_suspension']
STATES_STATES_KEYS = ['goal_relative', 'car/sensors_vel', 'car/sensors_gyro', 'car/body_down_vector']

def make_car_task_gym(*args, **kwargs):
    task = CarTask(*args, **kwargs)
    original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
    env = DMCGYM(original_env, new_step_api=True)
    env = ReplaceKey(env, 'car/realsense_camera', 'pixels')
    env = ReplaceKey(env, 'car/realsense_depth', 'depth')
    env = PermuteImage(env, 'pixels')
    env = PermuteImage(env, 'depth')
    env = KeysToStates(env, PIXELS_STATES_KEYS)
    env = RescaleAction(env, -np.ones(2), np.ones(2))
    env = RunningReturnInfo(env)
    return env

def make_car_task_gym_states(*args, **kwargs):
    task = CarTask(*args, include_camera=False, **kwargs)
    original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
    env = DMCGYM(original_env, new_step_api=True)
    env = KeysToStates(env, STATES_STATES_KEYS)
    env = RescaleAction(env, -np.ones(2), np.ones(2))
    env = RunningReturnInfo(env)
    return env


gym.register('procedural_driving/CarTask-v0', entry_point='procedural_driving:make_car_task_gym')
gym.register('procedural_driving/CarTask-states-v0', entry_point='procedural_driving:make_car_task_gym_states')
