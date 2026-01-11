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

# Two-path gRPC discovery. Prefer the upstream CMake CONFIG package
# (provided by vcpkg, Conan, or a source build with -DgRPC_INSTALL=ON
# — this is what PACE will have). Fall back to pkg-config for
# Debian/Ubuntu system packages, whose `libgrpc++-dev` historically
# ships no gRPCConfig.cmake. Both paths land on the same
# `gRPC::grpc++` link target so downstream target_link_libraries calls
# are environment-agnostic.
find_package(gRPC CONFIG QUIET)
if(gRPC_FOUND)
  message(STATUS "Lethe: found gRPC via CMake CONFIG package "
                 "(version=${gRPC_VERSION})")
else()
  message(STATUS "Lethe: gRPC CMake CONFIG package not found; "
                 "falling back to pkg-config")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  add_library(gRPC::grpc++ ALIAS PkgConfig::GRPCPP)
  message(STATUS "Lethe: found gRPC via pkg-config "
                 "(version=${GRPCPP_VERSION})")
endif()

# Plugin discovery — works for both paths (apt installs to /usr/bin;
# source builds with -DgRPC_INSTALL install to the chosen prefix).
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)
message(STATUS "Lethe: grpc_cpp_plugin=${GRPC_CPP_PLUGIN_EXECUTABLE}")

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
