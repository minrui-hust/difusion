
file(GLOB_RECURSE PROJECT_SRCS *.cpp)

add_library(${PROJECT_NAME} ${PROJECT_SRCS})

target_link_libraries(${PROJECT_NAME}
  ${YAML_CPP_LIBRARIES}
  ${G3LOG_LIBRARIES}
  ${G3SINKS_LIBRARIES}
  ${Boost_LIBRARIES}
  Ceres::ceres
)
