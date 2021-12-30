cmake_minimum_required(VERSION 3.14)

project(GeneralInferTemplate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

find_package (PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

list(APPEND PLUGINS "nvinfer")
# list(APPEND PLUGINS "nvonnxparser")
# list(APPEND PLUGINS "nvparsers")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

FILE(GLOB_RECURSE CORE_SRCS "core/*.cpp")
FILE(GLOB_RECURSE CORE_HDRS "core/*.hpp")

FILE(GLOB_RECURSE NODE_SRCS "node/*.cpp")
FILE(GLOB_RECURSE NODE_HDRS "node/*.hpp")

FILE(GLOB_RECURSE COMMON_SRCS "common/*.cpp")
FILE(GLOB_RECURSE COMMON_HDRS "common/*.hpp")

# message("UTILS_SRCS: " ${UTILS_SRCS})
# message("UTILS_HDRS: " ${UTILS_HDRS})
# message("NODE_SRCS: " ${NODE_SRCS})
# message("NODE_HDRS: " ${NODE_HDRS})

include_directories(/usr/local/cuda/include)

link_directories(
    ${YAMLCPP_LIBRARY_DIRS}
)

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
    ${YAMLCPP_LIBRARIES}
    ${PLUGIN_LIBS}
    ${OpenCV_LIBRARIES}
)
