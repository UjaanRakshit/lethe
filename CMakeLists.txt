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
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)

# Generate proto stubs into the build directory.
set(LETHE_PROTO_SRC ${CMAKE_SOURCE_DIR}/proto/lethe.proto)
set(LETHE_PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${LETHE_PROTO_GEN_DIR})

set(LETHE_PROTO_PB_CC ${LETHE_PROTO_GEN_DIR}/lethe.pb.cc)
set(LETHE_PROTO_PB_H  ${LETHE_PROTO_GEN_DIR}/lethe.pb.h)
set(LETHE_PROTO_GRPC_CC ${LETHE_PROTO_GEN_DIR}/lethe.grpc.pb.cc)
set(LETHE_PROTO_GRPC_H  ${LETHE_PROTO_GEN_DIR}/lethe.grpc.pb.h)

add_custom_command(
  OUTPUT ${LETHE_PROTO_PB_CC} ${LETHE_PROTO_PB_H}
         ${LETHE_PROTO_GRPC_CC} ${LETHE_PROTO_GRPC_H}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
          --cpp_out=${LETHE_PROTO_GEN_DIR}
          --grpc_out=${LETHE_PROTO_GEN_DIR}
          --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
          -I ${CMAKE_SOURCE_DIR}/proto
          ${LETHE_PROTO_SRC}
  DEPENDS ${LETHE_PROTO_SRC}
  COMMENT "Generating C++ proto stubs for lethe.proto"
  VERBATIM)

# Per-target source list. Both cache_server and tests link these.
set(LETHE_PROTO_GENERATED_SRCS ${LETHE_PROTO_PB_CC} ${LETHE_PROTO_GRPC_CC})

add_subdirectory(cache_server)

if(LETHE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
