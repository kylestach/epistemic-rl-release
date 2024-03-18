from offroad_learning.inference.state_machine import State as StateMachineState
import rospy
from typing import Dict, Any
import numpy as np


class StatsTracker:
    def __init__(self):
        self.last_goal_idx = 0

        self.distance_traveled_autonomous = 0.0
        self.lap_count = 0
        self.lap_times = []
        self.lap_distances = []
        self.lap_collisions = []
        self.lap_interventions = []
        self.lap_stucks = []
        self.lap_flips = []
        self.lap_average_speeds = []

        self.last_update_time = None
        self.last_position = None
        self.last_state_machine_state = None

        self.lap_distance = 0.0
        self.lap_time = 0.0
        self.lap_num_collisions = 0
        self.lap_num_interventions = 0
        self.lap_num_stuck = 0
        self.lap_num_flips = 0

        self.distance_traveled_total = 0.0
        self.distance_traveled_autonomous = 0.0

        self.time_total = 0.0
        self.time_autonomous = 0.0

        self.time_to_first_collision_free_lap = None

    def register_lap(self):
        self.lap_count += 1
        self.lap_times.append(self.lap_time)
        self.lap_distances.append(self.lap_distance)
        self.lap_collisions.append(self.lap_num_collisions)
        self.lap_interventions.append(self.lap_num_interventions)
        self.lap_stucks.append(self.lap_num_stuck)
        self.lap_flips.append(self.lap_num_flips)
        self.lap_average_speeds.append(self.lap_distance / self.lap_time)

        if self.time_to_first_collision_free_lap is None and self.lap_num_collisions == 0:
            self.time_to_first_collision_free_lap = self.time_total

        self.lap_time = 0
        self.lap_distance = 0
        self.lap_num_collisions = 0
        self.lap_num_interventions = 0
        self.lap_num_stuck = 0
        self.lap_num_flips = 0

    def update_time(self, state_machine_state: StateMachineState):
        time = rospy.Time.now()
        if self.last_update_time is None:
            self.last_update_time = time
            return

        dt = (time - self.last_update_time).to_sec()

        if state_machine_state == StateMachineState.LEARNING:
            self.time_autonomous += dt

        self.time_total += dt
        self.lap_time += dt

        self.last_update_time = time

    def update_distance_traveled(
        self, state_machine_state: StateMachineState, observation: Dict[str, Any]
    ):
        position = observation["pose_2d"][:2]
        if self.last_position is None:
            self.last_position = position
            return

        distance = np.linalg.norm(position - self.last_position)

        if state_machine_state == StateMachineState.LEARNING:
            self.distance_traveled_autonomous += distance

        self.distance_traveled_total += distance
        self.lap_distance += distance

        self.last_position = position

    def update_state(self, state_machine_state: StateMachineState):
        if self.last_state_machine_state is None:
            self.last_state_machine_state = state_machine_state
            return

        if self.last_state_machine_state == StateMachineState.LEARNING:
            if state_machine_state == StateMachineState.COLLISION:
                self.lap_num_collisions += 1
            elif state_machine_state == StateMachineState.STUCK:
                self.lap_num_stuck += 1
            elif state_machine_state == StateMachineState.INVERTED:
                self.lap_num_flips += 1
            elif state_machine_state == StateMachineState.TELEOP:
                self.lap_num_interventions += 1
            elif state_machine_state == StateMachineState.TELEOP_RECORD:
                self.lap_num_interventions += 1
            elif state_machine_state == StateMachineState.LEARNING:
                pass
            else:
                self.last_state_machine_state = state_machine_state
                raise ValueError(f"Invalid transition to {state_machine_state}")

        self.last_state_machine_state = state_machine_state

    def update(
        self,
        goal_idx: int,
        state_machine_state: StateMachineState,
        observation: Dict[str, Any],
    ):
        self.update_time(state_machine_state)
        self.update_distance_traveled(state_machine_state, observation)
        self.update_state(state_machine_state)

        if goal_idx != self.last_goal_idx and goal_idx == 0:
            self.register_lap()

        self.last_goal_idx = goal_idx

    def get_stats(self):
        return {
            "logging": {
                "lap_count": self.lap_count,
                "lap_times": self.lap_times,
                "lap_distances": self.lap_distances,
                "lap_collisions": self.lap_collisions,
                "lap_interventions": self.lap_interventions,
                "lap_stucks": self.lap_stucks,
                "lap_flips": self.lap_flips,
                "lap_average_speeds": self.lap_average_speeds,
                "distance_traveled_total": self.distance_traveled_total,
                "distance_traveled_autonomous": self.distance_traveled_autonomous,
                "time_total": self.time_total,
                "time_autonomous": self.time_autonomous,
            },
            "summary": {
                "time_to_first": self.time_to_first_collision_free_lap,
                "best_time": np.min(self.lap_times),
                "median_time": np.median(self.lap_times),
                "median_collisions": np.median(self.lap_collisions),
            } if len(self.lap_times) > 0 else {},
        }
