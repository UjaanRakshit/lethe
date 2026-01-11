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
