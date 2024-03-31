cmake_minimum_required(VERSION 3.15.0)
project(difusion)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -D_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DMANIF_NO_DEBUG")

include_directories(${PROJECT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
# message("EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(manif REQUIRED)
include_directories(${manif_INCLUDE_DIRS})
# message("MANIF_INCLUDE_DIRS: ${manif_INCLUDE_DIRS}")

find_package(Ceres REQUIRED)

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})
# message("YAML_CPP_INCLUDE_DIR:${YAML_CPP_INCLUDE_DIR}")

find_package(g3log REQUIRED)
include_directories(${G3LOG_INCLUDE_DIR})
# message("G3LOG_INCLUDE_DIR:${G3LOG_INCLUDE_DIR}")

set(G3SINKS_INCLUDE_DIR /usr/local/include)
set(G3SINKS_LIBRARIES /usr/local/lib/libg3logrotate.so
                      /usr/local/lib/libg3syslog.so)

find_package(
  Boost
  COMPONENTS filesystem serialization
  REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
# message("Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS}")

enable_testing()

add_subdirectory(difusion)
add_subdirectory(test)
add_subdirectory(app)

install(DIRECTORY config DESTINATION ${PROJECT_NAME})
