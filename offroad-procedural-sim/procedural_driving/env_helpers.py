import numpy as np
from dm_control.utils.transformations import quat_to_euler
import quaternion as npq


def batch_is_upside_down(quat):
    """Returns true if the car is upside down."""
    quat = npq.as_quat_array(quat)
    up_world = np.array([0, 0, 1])
    up_local = npq.as_vector_part(
        quat * npq.from_vector_part(up_world) * np.conjugate(quat))
    return np.sum(up_world[None] * up_local, axis=-1) < 0


def is_upside_down(quat):
    """Returns true if the car is upside down."""
    return batch_is_upside_down(quat[None])[0]


def batch_get_projected_forward_vel_mag(car_vel_local, goal_polar):
    return np.sum(car_vel_local[:, :2] * goal_polar[:, :2], axis=-1)


def make_batch_rotation_matrix(yaw):
    """Returns a batch of rotation matrices."""
    return np.stack([
        np.stack([np.cos(yaw), -np.sin(yaw)], axis=-1),
        np.stack([np.sin(yaw), np.cos(yaw)], axis=-1),
    ], axis=-2)


def batch_rotate_2d(vectors, yaw):
    """Rotates a batch of 2d vectors."""
    batch_size = vectors.shape[0]
    assert vectors.shape == (batch_size, 2)
    rot = make_batch_rotation_matrix(yaw)
    return np.matmul(rot, vectors[:, :, None])[:, :, 0]


def batch_world_displacement_to_relative_xy(car_pose_2d, goal):
    """Returns the displacement of the goal in the car's frame."""
    batch_size = car_pose_2d.shape[0]
    assert car_pose_2d.shape == (batch_size, 3)
    assert goal.shape == (batch_size, 2)

    pos = car_pose_2d[:, :2]
    yaw = car_pose_2d[:, 2]
    return batch_rotate_2d(goal - pos, -yaw)


def world_displacement_to_relative_xy(car_pose_2d, goal_relative):
    """Returns the displacement of the goal in the world frame."""
    return batch_world_displacement_to_relative_xy(car_pose_2d[None, :], goal_relative[None, :])[0]


def batch_relative_displacement_to_world_xy(car_pose_2d, goal_relative):
    """Returns the displacement of the goal in the world frame."""
    batch_size = car_pose_2d.shape[0]
    assert car_pose_2d.shape == (batch_size, 3)
    assert goal_relative.shape == (batch_size, 2)

    pos = car_pose_2d[:, :2]
    yaw = car_pose_2d[:, 2]
    rot = make_batch_rotation_matrix(yaw)
    return np.matmul(rot, goal_relative) + pos


def relative_displacement_to_world_xy(car_pose_2d, goal_relative):
    """Returns the displacement of the goal in the world frame."""
    return batch_relative_displacement_to_world_xy(car_pose_2d[None, :], goal_relative[None, :])[0]


def batch_relative_goal_polarcoord(car_pose_2d, goal):
    delta = batch_world_displacement_to_relative_xy(car_pose_2d, goal[:, :2])

    magnitude = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.concatenate((delta / (magnitude + 1e-6), 1.0 / (magnitude + 1e-6)), axis=-1)


def relative_goal_polarcoord(car_pose_2d, goal):
    return batch_relative_goal_polarcoord(car_pose_2d[None, :], goal[None, :])[0]
