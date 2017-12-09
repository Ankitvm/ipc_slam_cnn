cmake_minimum_required(VERSION 2.8.3)
project(powerbot_model)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  kdl_parser
  sensor_msgs
  std_msgs
  tf
  urdf
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs kdl_parser sensor_msgs std_msgs tf urdf
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(powerbot_publisher src/publisher.cc)
target_link_libraries(powerbot_publisher ${catkin_LIBRARIES})

install(TARGETS powerbot_publisher
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


add_executable(publisher_powerbot src/publisher_powerbot.cc)
target_link_libraries(publisher_powerbot ${catkin_LIBRARIES})

install(TARGETS publisher_powerbot
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
   launch defs meshes models
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
 )
