# cache_server/CMakeLists.txt

# Vendored BLAKE3 1.5.4 (portable C only, no SIMD dispatch). Single
# static library shared with the hash_compat_driver tool and the unit
# tests. The Python side uses the pyblake3 wrapper around the upstream
# Rust impl; both are BLAKE3-spec-compliant so digests agree
# bit-for-bit. Cross-language equivalence is asserted in
# tests/correctness/test_hash_compat.py via the driver subprocess.
set(LETHE_BLAKE3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blake3)
add_library(lethe_blake3 STATIC
  ${LETHE_BLAKE3_DIR}/blake3.c
  ${LETHE_BLAKE3_DIR}/blake3_dispatch.c
  ${LETHE_BLAKE3_DIR}/blake3_portable.c
)
target_compile_definitions(lethe_blake3 PRIVATE
  BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512)
target_include_directories(lethe_blake3 PUBLIC ${LETHE_BLAKE3_DIR})
# The vendored .c sources use a few non-portable patterns the global
# -Wpedantic flags us on. Suppress for the third-party tree only.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lethe_blake3 PRIVATE -w)
endif()

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
  lethe_blake3
)
target_include_directories(lethe_server PRIVATE ${LETHE_BLAKE3_DIR})

if(LETHE_ENABLE_RDMA)
  target_compile_definitions(lethe_server PRIVATE LETHE_ENABLE_RDMA=1)
  # NOTE: ibverbs / rdmacm link deps deferred until the IbverbsTransport
  # implementation actually uses them. The current ibverbs_transport.cpp
  # is a defined-symbol stub that aborts at runtime (gate #1 fallback
  # per docs/decisions/W5_rdma_fallback.md). Linking against
  # libibverbs/librdmacm when the .cpp uses none of their symbols would
  # add a build-host dependency for zero runtime benefit. Re-add the
  # following when the W12 PACE implementation lands:
  #   find_library(LIBIBVERBS ibverbs REQUIRED)
  #   find_library(LIBRDMACM  rdmacm  REQUIRED)
  #   target_link_libraries(lethe_server PRIVATE ${LIBIBVERBS} ${LIBRDMACM})
endif()

if(LETHE_ENABLE_METRICS)
  # find_package(prometheus-cpp CONFIG REQUIRED)
  # target_link_libraries(lethe_server PRIVATE prometheus-cpp::pull)
  target_compile_definitions(lethe_server PRIVATE LETHE_ENABLE_METRICS=1)
endif()

find_package(Threads REQUIRED)

# hash_compat_driver — small CLI that reads (prev_hash_hex, token_ids)
# on stdin and prints BLAKE3-chained-block-hash hex per line. Used by
# tests/correctness/test_hash_compat.py to assert the C++ and Python
# implementations produce bit-identical digests (W0 invariant).
add_executable(hash_compat_driver
  ${CMAKE_SOURCE_DIR}/tools/hash_compat_driver.cpp)
target_link_libraries(hash_compat_driver PRIVATE lethe_blake3 Threads::Threads)
target_include_directories(hash_compat_driver PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${LETHE_BLAKE3_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(hash_compat_driver PRIVATE -Werror)
endif()
