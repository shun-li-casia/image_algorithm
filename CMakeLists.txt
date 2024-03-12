cmake_minimum_required(VERSION 3.1)
project(image_algorithm)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g -ggdb -fPIC")

## Find catkin macros and libraries
find_package(catkin REQUIRED
    roscpp
    std_msgs
    cv_bridge
    utility_tool
    camera_models
    )

## System dependencies are found with CMake's conventions
find_package(OpenCV 4 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)
message(WARNING ${PCL_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    3rdparty/libSGM/include
    )

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp std_msgs cv_bridge
    LIBRARIES ${PROJECT_NAME}
    # DEPENDS system_lib
    )

link_directories("3rdparty/libSGM/build/src")

add_library(${PROJECT_NAME}
    src/modules/stereo_rectifier.cc
    src/modules/disparity_calculator.cc
    )

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${PCL_LIBRARIES}
    sgm
    yaml-cpp
    )

add_executable(test_stereo_match src/app/test_stereo_match.cc)
target_link_libraries(test_stereo_match ${PROJECT_NAME})

add_executable(tune_sgbm src/tools/tune_sgbm.cc)
target_link_libraries(tune_sgbm ${PROJECT_NAME})

add_executable(tune_sgbm_pcl src/tools/tune_sgbm_pcl.cc)
target_link_libraries(tune_sgbm_pcl ${PROJECT_NAME})