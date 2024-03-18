cmake_minimum_required(VERSION 3.0.2)
project(offroad_bringup)
find_package(catkin REQUIRED offroad_learning)
catkin_package(CATKIN_DEPENDS offroad_learning)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)