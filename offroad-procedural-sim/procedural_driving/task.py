import collections
from dm_control import composer
from dm_control.composer.observation import observable as observable_lib
import numpy as np
from dm_control.utils import transformations
from dm_control.mjcf.physics import Physics
import quaternion as npq
from typing import Dict, Tuple

from procedural_driving.car import Car
from procedural_driving.procedural_arena import ProceduralArena
from procedural_driving import env_helpers
from procedural_driving.goal_graph import RandomGoalGraph

DEFAULT_CONTROL_TIMESTEP = 0.1
DEFAULT_PHYSICS_TIMESTEP = 0.001


def batch_compute_reward(
    car_quat: np.ndarray,
    car_pose_2d: np.ndarray,
    car_vel_2d: np.ndarray,
    goal_absolute: np.ndarray,
    should_timeout: np.ndarray,
    local_vel: np.ndarray,
):
    """
    Compute the reward over a batch of robot data.

    For a single instance, add an additional leading dimension with size 1.
    """

    # If it's upside down, terminate with negative reward
    directions_to_goal = goal_absolute - car_pose_2d[:, :2]
    directions_to_goal /= (
        np.linalg.norm(directions_to_goal, axis=-1, keepdims=True) + 1e-6
    )

    velocities_to_goal = np.sum(car_vel_2d * directions_to_goal, axis=-1)

    r = velocities_to_goal

    return r


def batch_compute_reward_from_observation(
    observation: Dict[str, np.ndarray],
    action: np.ndarray,
    next_observation: Dict[str, np.ndarray],
):
    """
    Compute the reward over a batch of observations.
    """

    # For some reason dm_control computes the reward after the step, so we need to do the same
    car_quat = next_observation["car/body_rotation"]
    car_pose_2d = next_observation["car/body_pose_2d"]
    car_vel_2d = next_observation["car/body_vel_2d"]
    goal_absolute = next_observation["goal_absolute"]
    should_timeout = next_observation["timeout"]
    local_vel = next_observation["car/sensors_vel"]

    return batch_compute_reward(
        car_quat, car_pose_2d, car_vel_2d, goal_absolute, should_timeout, local_vel
    )


def sample_goals_future(
    observations: Dict[str, np.ndarray],
    next_observations: Dict[str, np.ndarray],
    future_observations: Dict[str, np.ndarray],
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Sample goals for both current and next observations from the given future observations.

    The absolute goals are the same for both current and next observations, but the relative goals are different (because the car has moved)
    """
    # Sample goals relative to the current pose
    car_pose_t0 = observations["car/body_pose_2d"]
    car_pose_t1 = next_observations["car/body_pose_2d"]

    goal_absolute = future_observations["car/body_pose_2d"][:, :2]

    goal_polar_t0 = env_helpers.batch_relative_goal_polarcoord(
        car_pose_t0, goal_absolute
    )
    goal_polar_t1 = env_helpers.batch_relative_goal_polarcoord(
        car_pose_t1, goal_absolute
    )

    return ((goal_absolute, goal_polar_t0), (goal_absolute, goal_polar_t1))


def sample_goals_random(self, batch_size, observations, next_observations):
    """
    Sample random goals
    """

    # Sample goals relative to the current pose
    car_pose_t0 = observations["car/body_pose_2d"]
    car_pos_t0 = observations["car/body_pose_2d"][:, :2]
    car_yaw_t0 = observations["car/body_pose_2d"][:, 2]
    car_pose_t1 = next_observations["car/body_pose_2d"]

    theta = np.random.normal(0, 1.0, size=(batch_size))
    distance = np.random.normal(2.5, 1.5, size=(batch_size))

    # The car to goal vector, in world coordinates, at t=0
    car_to_goal_world_t0 = np.stack(
        [np.cos(theta + car_yaw_t0) * distance, np.sin(theta + car_yaw_t0) * distance],
        axis=-1,
    )

    # The absolute goal position, in world coordinates, at t=0 and t=1
    goal_absolute = car_pos_t0 + car_to_goal_world_t0

    goal_polar_t0 = env_helpers.batch_relative_goal_polarcoord(
        car_pose_t0, goal_absolute
    )
    goal_polar_t1 = env_helpers.batch_relative_goal_polarcoord(
        car_pose_t1, goal_absolute
    )

    return ((goal_absolute, goal_polar_t0), (goal_absolute, goal_polar_t1))


class CarTask(composer.Task):
    def __init__(
        self,
        physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
        control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
        scale: float = 20.0,
        include_camera: bool = True,
        world_seed=0,
        use_alt_car=False,
        world_name="default",
    ):
        self._arena = ProceduralArena(seed=world_seed, world_name=world_name)

        if use_alt_car:
            self._car = Car(model_name="offroad_alt.xml")
        else:
            self._car = Car(model_name="offroad.xml")
        self._arena.add_free_entity(self._car)

        self._car.observables.enable_all()
        if not include_camera:
            self._car.observables.get_observable("realsense_camera").enabled = False
            self._car.observables.get_observable("realsense_depth").enabled = False

        self.goal_graph = RandomGoalGraph(scale=10, arena=self._arena)

        self.set_timesteps(control_timestep, physics_timestep)

        self._last_positions = np.empty((500, 2), dtype=np.float32)
        self._last_positions.fill(np.inf)

        self.num_flips = 0
        self.num_timeouts = 0
        self.num_stuck = 0

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        def goal_polar(physics: Physics, goal):
            car_pose_2d = self._car.observables.get_observable("body_pose_2d")(physics)
            return env_helpers.relative_goal_polarcoord(
                car_pose_2d, goal,
            )

        relative_polar_coords = observable_lib.Generic(lambda physics: goal_polar(physics, self.goal_graph.current_goal))
        relative_polar_coords.enabled = True

        relative_polar_coords_next = observable_lib.Generic(lambda physics: goal_polar(physics, self.goal_graph.next_goal))
        relative_polar_coords_next.enabled = True

        absolute = observable_lib.Generic(lambda _: self.goal_graph.current_goal)
        absolute.enabled = True

        absolute_next = observable_lib.Generic(lambda _: self.goal_graph.next_goal)
        absolute_next.enabled = True

        should_timeout = observable_lib.Generic(self.should_timeout)
        should_timeout.enabled = True

        task_obs = collections.OrderedDict(
            {
                "goal_relative": relative_polar_coords,
                "goal_absolute": absolute,
                "goal_relative_next": relative_polar_coords_next,
                "goal_absolute_next": absolute_next,
                "timeout": should_timeout,
            }
        )

        return task_obs

    def initialize_episode(self, physics: Physics, random_state: np.random.RandomState):
        super().initialize_episode(physics, random_state)
        self._arena.initialize_episode(physics, random_state)

        # Find a reset position near the robot's last position
        last_position = (
            self._last_positions[-1]
            if self._last_positions[-1][0] != np.inf
            else np.array([0.0, 0.0])
        )
        self._arena._set_chunk(physics, self._arena._lookup_chunk(last_position))
        start_pos = np.asarray(self._arena._find_nearby_reset_position(last_position))
        start_quat = transformations.euler_to_quat(
            [0, 0, np.random.uniform(0, 2 * np.pi)]
        )
        self._car.set_pose(physics, start_pos, start_quat)

        # Reset to pick a new goal
        self.goal_graph.reset(start_pos[:2])
        self._last_positions.fill(np.inf)

    def should_terminate_episode(self, physics: Physics):
        """
        Should the episode terminate? Called after the step (i.e. with next_observation).
        """

        _, car_quat = self._car.get_pose(physics)

        if env_helpers.is_upside_down(car_quat):
            self.num_flips += 1
            return True

        if self.should_timeout(physics):
            self.num_timeouts += 1
            return True

        if self.goal_graph.is_failed():
            self.num_stuck += 1
            return True

        return False

    def before_step(
        self, physics: Physics, action, random_state: np.random.RandomState
    ):
        """
        Called before the physics step.
        """

        # We've probably moved, so update the set of loaded chunks accordingly.
        car_pos, _ = self._car.get_pose(physics)
        car_chunk = self._arena._lookup_chunk(car_pos[:2])
        self._arena._set_chunk(physics, car_chunk)

        # We have to call the super method to propagate the action to the car
        super().before_step(physics, action, random_state)

    def after_step(self, physics: Physics, random_state: np.random.RandomState):
        """
        Called after the step, but before the reward is computed.
        """

        # Update the goal graph and the last positions.
        car_pos, _ = self._car.get_pose(physics)
        self.goal_graph.tick(car_pos[:2])

        self._last_positions = np.roll(self._last_positions, -1, axis=0)
        self._last_positions[-1] = car_pos[:2]

    def get_reward(self, physics: Physics):
        """
        Compute the reward for the current state. Called after the step (i.e. with next_observation).
        """

        _, car_quat = self._car.get_pose(physics)
        car_pose_2d = self._car.observables.get_observable("body_pose_2d")(physics)
        car_vel_2d = self._car.observables.get_observable("body_vel_2d")(physics)
        local_vel = self._car.observables.get_observable("sensors_vel")(physics)
        goal_absolute = self.goal_graph.current_goal
        should_timeout = self.should_timeout(physics)

        reward = batch_compute_reward(
            car_quat[None],
            car_pose_2d[None],
            car_vel_2d[None],
            goal_absolute[None],
            should_timeout[None],
            local_vel[None],
        )[0]

        return reward

    def should_timeout(self, physics: Physics):
        car_pos, _ = self._car.get_pose(physics)
        return np.array(
            np.linalg.norm(car_pos[None, :2] - self._last_positions).max() < 1.0,
            dtype=np.float32,
        )
