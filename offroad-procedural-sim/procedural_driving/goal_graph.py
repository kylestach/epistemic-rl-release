import numpy as np
import random
import collections

GOAL_THRESHOLD = 1.5
TICKS_WITHOUT_PROGRESS = 100
PROGRESS_THRESHOLD = 0.5


class RandomGoalGraph():
    """
    A graph of goals that the car can drive to. Once the car arrives at a goal,
    the goal will be changed to one of its successors.
    """

    def __init__(self, scale, arena, goal_threshold=GOAL_THRESHOLD):
        self.goal_threshold = goal_threshold
        self.scale = scale
        self.max_tries_before_reset = 3
        self.goal_distance = 5

        self.current_goal = np.array([0., 0.])
        self.next_goal = np.array([0., 0.])
        self._last_distances = collections.deque(maxlen=TICKS_WITHOUT_PROGRESS)
        self._arena = arena

    def is_complete(self, car_pos):
        return np.linalg.norm(np.array(car_pos)[:2] - self.current_goal) < self.goal_threshold

    def _select_new_goal(self, car_pos):
        valid_goal = None
        while valid_goal is None:
            new_goal = np.random.normal(
                loc=car_pos, scale=self.goal_distance, size=(2,))
            valid_goal = self._arena._find_nearby_reset_position(new_goal)
        self.current_goal = self.next_goal
        self.next_goal = np.asarray(valid_goal[:2])
        self._last_distances.clear()

    def is_failed(self):
        return self._failed_tries >= self.max_tries_before_reset

    def tick(self, car_pos: np.ndarray):
        """
        Update the goal if reached.
        """
        assert car_pos.shape == (2,)

        distance = np.linalg.norm(car_pos - self.current_goal)
        self._last_distances.append(distance)

        if self.is_complete(car_pos):
            self._select_new_goal(car_pos)
            self._failed_tries = 0
        else:
            progress = self._last_distances[0] - self._last_distances[-1]
            if progress < PROGRESS_THRESHOLD and len(self._last_distances) == TICKS_WITHOUT_PROGRESS:
                self._failed_tries += 1
                self._select_new_goal(car_pos)

        return self._failed_tries >= self.max_tries_before_reset

    def reset(self, car_pos: np.ndarray):
        assert car_pos.shape == (2,)

        self._select_new_goal(car_pos)
        self._failed_tries = 0
