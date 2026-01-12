# cache_server/CMakeLists.txt

set(LETHE_SRCS
  src/main.cpp
  src/cache.cpp
  src/block_store.cpp
  src/tiered_store.cpp
  src/routing.cpp
  src/replication.cpp
  src/eviction.cpp
  src/membership.cpp
  src/metrics.cpp
  # GrpcStreamTransport is the default bulk-KV transport and is built
  # unconditionally; IbverbsTransport (below) is the LETHE_ENABLE_RDMA
  # alternative.
  src/grpc_stream_transport.cpp
)

if(LETHE_ENABLE_RDMA)
  list(APPEND LETHE_SRCS src/ibverbs_transport.cpp)
endif()

add_executable(lethe_server ${LETHE_SRCS} ${LETHE_PROTO_GENERATED_SRCS})

# Ensure the proto-gen step has run before lethe_server compiles. The
# top-level CMakeLists wraps protoc invocation in the `lethe_proto_gen`
# phony target precisely for this cross-subdirectory dependency.
add_dependencies(lethe_server lethe_proto_gen)

# -Werror under GCC/Clang only. MSVC's warning set is different and
# would surface noise that doesn't reflect real code issues. -Wall and
# -Wextra are already in the top-level CMakeLists. The generated proto
# .pb.cc files would also fall under this, so we drop -Werror back to
# -Wno-error on them via set_source_files_properties below.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lethe_server PRIVATE -Werror)
endif()

# Generated proto sources can carry warnings we can't fix (auto-emitted
# code, version-dependent). Keep -Werror off for them so the rest of
# cache_server can stay strict. -Wno-error here strips the promotion
# without disabling the warnings themselves.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(
    ${LETHE_PROTO_GENERATED_SRCS}
    PROPERTIES COMPILE_OPTIONS "-Wno-error")
endif()

target_include_directories(lethe_server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${LETHE_PROTO_GEN_DIR}
)

target_link_libraries(lethe_server PRIVATE
  gRPC::grpc++
  protobuf::libprotobuf
  Threads::Threads
)

if(LETHE_ENABLE_RDMA)
  target_link_libraries(lethe_server PRIVATE ibverbs rdmacm)
  target_compile_definitions(lethe_server PRIVATE LETHE_ENABLE_RDMA=1)
endif()

if(LETHE_ENABLE_METRICS)
  # find_package(prometheus-cpp CONFIG REQUIRED)
  # target_link_libraries(lethe_server PRIVATE prometheus-cpp::pull)
  target_compile_definitions(lethe_server PRIVATE LETHE_ENABLE_METRICS=1)
endif()

find_package(Threads REQUIRED)
