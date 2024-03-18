import rospy
from typing import Type, Callable, Optional, Tuple, Any, Dict, List
import numpy as np
from functools import partial
from threading import RLock

from threading import RLock

PyTree = Any


class AggregatedTopic:
    def __init__(
        self,
        name: str,
        topic_name: str,
        topic_type: Type[rospy.AnyMsg],
        data_shape: PyTree,
        estimated_latency: rospy.Duration,
        converter: Callable,
        history_length: int = 100,
    ):
        self.name = name
        self.topic_name = topic_name
        self.topic_type = topic_type
        self.estimated_latency = estimated_latency
        self.stamps = np.zeros(history_length, dtype=np.int64)
        self.data_shape = {}
        self.buffer = {}

        for k, v in data_shape.items():
            if isinstance(v, dict):
                self.data_shape[k] = tuple(v["shape"])
                self.buffer[k] = np.zeros(
                    (history_length, *v["shape"]), dtype=v["dtype"]
                )
            elif isinstance(v, list):
                self.data_shape[k] = tuple(v)
                self.buffer[k] = np.zeros((history_length, *v), dtype=np.float64)

        self.to_rosmsg = converter
        self.num_data = 0

        self.extra_callbacks: List[Callable[[rospy.Time, Any], None]] = []
        self.lock = RLock()

        self.lock = RLock()

    def setup_ros(self):
        """
        Setup ROS subscribers for all topics
        """
        if self.topic_name is not None:
            self.subscription = rospy.Subscriber(
                self.topic_name, self.topic_type, self.callback
            )

    def __repr__(self) -> str:
        return f"AggregatedTopic({self.subscription.name}, {self.subscription.type}, {self.to_rosmsg})"

    def callback(self, msg: rospy.AnyMsg):
        """
        Callback function for the subscriber. This function is called whenever a new message is received.
        """
        with self.lock:
            insert_result = self.insert_ros(msg)

        if insert_result is None:
            rospy.logwarn(
                f"Failed to convert message to numpy array for topic {self.name}"
            )
            return

        stamp, data = insert_result

        for callback in self.extra_callbacks:
            callback(stamp, data)

    def insert_ros(self, msg: rospy.AnyMsg):
        """
        Manually insert a ROS message into the buffer, without calling callbacks.
        """
        result = self.to_rosmsg(msg)
        if result is None:
            return None

        stamp, data = result
        self.insert_data(stamp, data)
        return stamp, data

    def insert_data(self, stamp: int, data: Dict[str, np.ndarray]):
        """
        Insert numpy data into the buffer, without calling callbacks.
        """
        with self.lock:
            assert (
                data.keys() == self.buffer.keys()
            ), f"{data.keys()} != {self.buffer.keys()}"

        with self.lock:
            if self.num_data < self.stamps.shape[0]:
                self.stamps[self.num_data] = stamp
                for k, buffer in self.buffer.items():
                    buffer[self.num_data] = data[k]

                self.num_data += 1
            else:
                self.stamps = np.roll(self.stamps, -1, axis=0)
                self.stamps[-1] = stamp

                self.buffer = {
                    k: np.roll(buffer, -1, axis=0) for k, buffer in self.buffer.items()
                }
                for k, buffer in self.buffer.items():
                    buffer[-1] = data[k]

    def get(self, query_stamp: rospy.Time) -> Tuple[rospy.Time, Dict[str, np.ndarray]]:
        """
        Get the most recent data before the query stamp.
        """
        with self.lock:
            if self.num_data == 0:
                return rospy.Time(0), {k: None for k in self.buffer}

            lookup_idx = (
                np.searchsorted(
                    self.stamps[: self.num_data],
                    (query_stamp + self.estimated_latency).to_nsec(),
                    side="right",
                )
                - 1
            )
            return rospy.Time(nsecs=self.stamps[lookup_idx]) - self.estimated_latency, {
                k: b[lookup_idx] for k, b in self.buffer.items()
            }

    def get_latest(self) -> Tuple[rospy.Time, Dict[str, np.ndarray]]:
        """
        Get the most recent data.
        """
        with self.lock:
            if self.num_data == 0:
                return rospy.Time(0), {k: None for k in self.buffer}

            assert (
                self.stamps[self.num_data - 1] > 0
            ), f"Stamp is negative in {self.name}: {self.stamps}"

            return rospy.Time(
                nsecs=self.stamps[self.num_data - 1]
            ) - self.estimated_latency, {
                k: b[self.num_data - 1] for k, b in self.buffer.items()
            }

    def get_nearest(
        self, query_stamp_ros: rospy.Time
    ) -> Tuple[rospy.Time, Dict[str, np.ndarray]]:
        """
        Get the nearest data to the query stamp.
        """
        with self.lock:
            if self.num_data == 0:
                return rospy.Time(0), {k: None for k in self.buffer}

            query_stamp = (query_stamp_ros + self.estimated_latency).to_nsec()

            lookup_idx = (
                np.searchsorted(
                    self.stamps[: self.num_data],
                    query_stamp,
                    side="right",
                )
                - 1
            )

            if lookup_idx == -1 or lookup_idx >= self.num_data:
                return rospy.Time(0), {k: None for k in self.buffer}

            # Check what's closer: the data before or after the query stamp
            if lookup_idx < self.num_data - 1:
                assert (
                    self.stamps[lookup_idx] <= query_stamp
                ), f"stamps[{lookup_idx}] ({self.stamps[lookup_idx]}) > {query_stamp} in topic {self.name}"
                assert (
                    self.stamps[lookup_idx + 1] >= query_stamp
                ), f"stamps[{lookup_idx + 1}] ({self.stamps[lookup_idx + 1]}) < {query_stamp} in topic {self.name}"

                err_l = abs(self.stamps[lookup_idx] - query_stamp)
                err_r = abs(self.stamps[lookup_idx + 1] - query_stamp)

                if err_l > err_r:
                    lookup_idx += 1

            return rospy.Time(nsecs=self.stamps[lookup_idx]) - self.estimated_latency, {
                k: b[lookup_idx] for k, b in self.buffer.items()
            }

    def add_callback(self, callback: Callable[[rospy.Time, Any], None]):
        """
        Add a callback function to the subscriber.
        """
        self.extra_callbacks.append(callback)

    def has_received(self, query_stamp_ros: rospy.Time) -> bool:
        """
        Check if we've already received data for the given query stamp, adjusted for latency.
        """
        with self.lock:
            query_stamp = (query_stamp_ros + self.estimated_latency).to_nsec()

            return query_stamp <= self.stamps[self.num_data - 1]

    def has_lost(self, query_stamp_ros: rospy.Time) -> bool:
        """
        Check if we've lost data for the given query stamp, adjusted for latency.
        """
        with self.lock:
            query_stamp = (query_stamp_ros + self.estimated_latency).to_nsec()

            return query_stamp < self.stamps[0]
