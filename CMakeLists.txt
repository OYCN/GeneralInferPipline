cmake_minimum_required(VERSION 3.11)

project(GeneralInferTemplate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(glog
    GIT_REPOSITORY
    https://github.com/google/glog.git
)
FetchContent_MakeAvailable(glog)

FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY
    https://github.com/jbeder/yaml-cpp.git
)
FetchContent_MakeAvailable(yaml-cpp)

find_package(OpenCV REQUIRED)

find_library(TRT_NVINFER NAMES "nvinfer" "/usr" PATH_SUFFIXES lib)

FILE(GLOB_RECURSE CORE_SRCS "core/*.cpp")
FILE(GLOB_RECURSE CORE_HDRS "core/*.hpp")

FILE(GLOB_RECURSE NODE_SRCS "node/*.cpp")
FILE(GLOB_RECURSE NODE_HDRS "node/*.hpp")

FILE(GLOB_RECURSE COMMON_SRCS "common/*.cpp")
FILE(GLOB_RECURSE COMMON_HDRS "common/*.hpp")

include_directories(/usr/local/cuda/include)

add_executable(main
    ${NODE_SRCS}
    ${CORE_SRCS}
    ${COMMON_SRCS}
    foo.cu
    demo/main.cpp
)

target_include_directories(main
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(main
    glog::glog
    yaml-cpp::yaml-cpp
    ${TRT_NVINFER}
    ${OpenCV_LIBRARIES}
)
