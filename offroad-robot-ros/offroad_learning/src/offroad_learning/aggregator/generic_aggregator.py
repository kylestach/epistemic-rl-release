from enum import Enum
from offroad_learning.aggregator.aggregated_topic import AggregatedTopic
from offroad_learning.aggregator import converters
import yaml
import importlib
import rospy
from typing import Optional, List, Dict, Callable, Any, Tuple
import warnings
import numpy as np

"""
Aggregator for multiple topics.

Configured via yaml file, e.g.:

topics:
  - name: "image"
    type: "sensor_msgs.msg.Image"
    topic: "/camera/image_raw"
    history_length: 30
    # Positive latency means that the receive time is AFTER the source time (so retrieval will look into the future)
    # This is the case for normal sensors
    estimated_latency_ms: 40
    data_shape:
      pixels:
        shape: [128, 128, 3]
        dtype: "uint8"
    converter:
      converter: "ImageConverter"
      config:
        width: 128
        height: 128
  - name: "imu"
    type: "sensor_msgs.msg.Imu"
    topic: "/imu/data"
    history_length: 200
    estimated_latency_ms: 40
    data_shape:
      gyro: [3]
      accel: [3]
    converter:
      converter: "ImuConverter"
...
  - name: "prev_action" # Same as action, but shifted by one timestep (latency=100ms)
    topic: null
    history_length: 100
    # Negative latency means that the receive time is BEFORE the source time (so retrieval will look into the past)
    # This allows retrieving previous states
    estimated_latency_ms: -100
    data_shape:
      prev_action: [2]
"""


class TopicStatus(Enum):
    VALID = 0
    NOT_YET = 1
    LOST = 2


class Aggregator:
    """
    Aggregator for multiple topics.
    """

    topics: List[AggregatedTopic]

    def __init__(self, config_file: str = None, config: Dict[str, Any] = None):
        if config_file is not None:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

        elif config is None:
            raise ValueError("Must specify either config_file or config")

        self.topics = []
        for topic_config in config["topics"]:
            topic_name = topic_config.get("topic", None)

            if topic_name is None:
                topic_type = None
                converter = None
            else:
                topic_type_string = topic_config.get("type", None).split(".")
                topic_module = ".".join(topic_type_string[:-1])
                topic_classname = topic_type_string[-1]
                topic_type = getattr(
                    importlib.import_module(topic_module), topic_classname
                )

                converter_config = topic_config.get("converter", None)
                converter_name = converter_config.get("converter", None)
                converter_type = getattr(converters, converter_name)
                converter = converter_type(**converter_config.get("config", {}))

            topic_aggregator = AggregatedTopic(
                name=topic_config["name"],
                topic_name=topic_name,
                converter=converter,
                topic_type=topic_type,
                history_length=topic_config["history_length"],
                data_shape=topic_config["data_shape"],
                estimated_latency=rospy.Duration.from_sec(
                    topic_config["estimated_latency_ms"] / 1000
                ),
            )

            self.topics.append(topic_aggregator)

    def setup_ros(self):
        """
        Setup ROS subscribers for all topics
        """
        for topic in self.topics:
            topic.setup_ros()

    def insert(self, stamp: rospy.Time, data: Dict[str, Any]):
        """
        Manually insert data into a particular topic stream.
        """
        data = data.copy()

        for topic in self.topics:
            data_for_this_topic = {}
            for k in topic.data_shape.keys():
                if k in data:
                    data_for_this_topic[k] = data.pop(k)

            if len(data_for_this_topic) > 0:
                topic.insert_data(stamp.to_nsec(), data_for_this_topic)

        assert len(data) == 0, f"Data {data} not inserted"

    def get(self, query_stamp: rospy.Time, tolerance: Optional[rospy.Duration] = None):
        """
        Get the data from all topics nearest to a given timestamp
        """

        result = {}
        for topic in self.topics:
            stamp, data = topic.get(query_stamp)

            if tolerance is not None:
                if abs(query_stamp - stamp) > tolerance:
                    warnings.warn("Topics {} are stale".format(data.keys()))
                    return None

            result.update(data)
        return result

    def get_latest(self, tolerance: Optional[rospy.Duration] = None):
        """
        Get the latest data from all topics
        """

        result = {}
        for topic in self.topics:
            stamp, data = topic.get_latest()

            if tolerance is not None:
                if abs(rospy.Time.now() - stamp) > tolerance:
                    warnings.warn("Topics {} are stale".format(data.keys()))
                    return None

            result.update(data)
        return result

    def get_nearest(
        self, query_stamp: rospy.Time, tolerance: Optional[rospy.Duration] = None
    ):
        """
        Get the data from all topics nearest to a given timestamp
        """

        result = {}
        for topic in self.topics:
            stamp, data = topic.get_nearest(query_stamp)

            if tolerance is not None:
                if abs(query_stamp - stamp) > tolerance:
                    print("Topics {} are stale".format(data.keys()))
                    return None

            result.update(data)
        return result

    def get_latest_synced(
        self, query_key: str, tolerance: Optional[rospy.Duration] = None
    ):
        """
        Get the latest data from all topics, nearest to the latest data from the given key
        """
        stamp = None
        for topic in self.topics:
            if query_key in topic.data_shape:
                stamp = topic.get_latest()[0]
                break
        if stamp is not None:
            return self.get_nearest(stamp, tolerance)

    def get_nearest_synced(
        self,
        query_key: str,
        query_stamp: rospy.Time,
        tolerance: Optional[rospy.Duration] = None,
    ):
        """
        Get the latest data from all topics, nearest to the latest data from the given key
        """
        stamp = None
        for topic in self.topics:
            if query_key in topic.data_shape:
                stamp = topic.get_nearest(query_stamp)[0]
                break
        if stamp is not None:
            return self.get_nearest(stamp, tolerance)

    def add_callback(self, key, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback for a given key
        """

        def callback_wrapper(stamp, data):
            callback(self.get_latest())

        for topic in self.topics:
            if key in topic.data_shape:
                topic.add_callback(callback_wrapper)

    def zeros(self):
        result = {}

        for topic in self.topics:
            for k, v in topic.data_shape.items():
                result[k] = np.zeros_like(topic.buffer[k][0])

        return result

    def status(self, query_stamp: rospy.Time) -> TopicStatus:
        """
        Check if all topics are up to date
        """
        for topic in self.topics:
            if topic.has_lost(query_stamp):
                return TopicStatus.LOST

        for topic in self.topics:
            if not topic.has_received(query_stamp):
                return TopicStatus.NOT_YET

        return TopicStatus.VALID

    def status_verbose(self, query_stamp: rospy.Time) -> Dict[str, TopicStatus]:
        """
        Check if each topics is up to date
        """

        def status_for_topic(topic: AggregatedTopic) -> TopicStatus:
            if topic.has_lost(query_stamp):
                return TopicStatus.LOST

            if not topic.has_received(query_stamp):
                return TopicStatus.NOT_YET

            return TopicStatus.VALID

        return {topic.name: status_for_topic(topic) for topic in self.topics}

    def data_shape(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get the data shape of all topics
        """
        result = {}

        for topic in self.topics:
            result.update(topic.data_shape)

        return result
