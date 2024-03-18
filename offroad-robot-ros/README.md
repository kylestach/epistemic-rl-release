# Real robot setup
> [!WARNING]  
> These instructions are still mostly untested. Please open an issue if you run into any problems.
See the release of [FastRLAP](https://github.com/kylestach/fastrlap-release/) for general setup instructions.

On the robot side, install `jaxrl5` and put the `offroad-learning` directory into `~/ros_ws/src`. On the training side, install everything normally (ROS is not explicitly required on the training computer).

Once you've got everything set up, you'll need to collect a set of goal checkpoints defining your course:
```bash
rosrun offroad_learning goal_graph_recorder_node _fixed_frame:="map"
```
Drive the course, pressing the `B` button at each goal location. This will print out a set of goals, which you can put in `offroad_learning/config/goals/<environment name>/goals.yaml`.

To launch robot-side inference, you can use `roslaunch offroad_bringup real_inference.launch`. On the training computer, run `python offroad_learning/src/offroad_learning/training/training.py`.