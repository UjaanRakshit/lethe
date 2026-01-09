cmake_minimum_required(VERSION 3.20)
project(lethe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build types.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Warnings.
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Optional dependencies. RDMA is W5–6; gate via -DLETHE_ENABLE_RDMA=ON.
option(LETHE_ENABLE_RDMA "Build the ibverbs/RDMA transport" OFF)
option(LETHE_ENABLE_METRICS "Build prometheus-cpp metrics" ON)
option(LETHE_BUILD_TESTS "Build unit tests" ON)

# gRPC + protobuf (required).
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Generate proto stubs into the build directory.
set(LETHE_PROTO_SRC ${CMAKE_SOURCE_DIR}/proto/lethe.proto)
set(LETHE_PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${LETHE_PROTO_GEN_DIR})

# (Real protoc invocation deferred; see cache_server/CMakeLists.txt.)

add_subdirectory(cache_server)

if(LETHE_BUILD_TESTS)
  enable_testing()
  # add_subdirectory(tests)   # W3 onward
endif()
