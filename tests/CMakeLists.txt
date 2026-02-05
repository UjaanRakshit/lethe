# tests/CMakeLists.txt — W1 unit tests.
#
# Built when -DLETHE_BUILD_TESTS=ON (default). Each test is a
# standalone main() with assert()s — no GoogleTest until W3+ when the
# test surface gets big enough to justify a real harness.
#
# ThreadSanitizer wiring: pass -DLETHE_ENABLE_TSAN=ON. On clang/gcc
# this adds -fsanitize=thread to the unit tests only (the production
# server is unaffected). MSVC's TSan is not supported here.

option(LETHE_ENABLE_TSAN "Build unit tests with ThreadSanitizer" OFF)

set(_lethe_test_compile_options "")
set(_lethe_test_link_options "")
if(LETHE_ENABLE_TSAN)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
     CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_lethe_test_compile_options -fsanitize=thread -g -O1)
    set(_lethe_test_link_options    -fsanitize=thread)
  else()
    message(WARNING
      "LETHE_ENABLE_TSAN ignored: only Clang/GCC supported "
      "(detected ${CMAKE_CXX_COMPILER_ID}).")
  endif()
endif()

# Shared list of subsystem .cpp files that unit tests link against.
# Tests do not pull in main.cpp (no gRPC server in unit tests).
set(_lethe_test_link_srcs
  ${CMAKE_SOURCE_DIR}/cache_server/src/block_store.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/tiered_store.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/ssd_block_store.cpp  # W7
  ${CMAKE_SOURCE_DIR}/cache_server/src/cache.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/membership.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/routing.cpp
  # W4: replication.cpp pulls in lethe.grpc.pb.h, so test targets need
  # the generated proto sources + include dir + gRPC link too. Wired
  # below per-target.
  ${CMAKE_SOURCE_DIR}/cache_server/src/replication.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/eviction.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/metrics.cpp
  # W5-6: GrpcStreamTransport is now constructed by LetheCache's ctor,
  # so every test that builds a LetheCache (test_cache, future
  # test_tiered_store) needs the transport TU linked.
  ${CMAKE_SOURCE_DIR}/cache_server/src/grpc_stream_transport.cpp
)

function(lethe_add_unit_test name source)
  # The generated proto sources are added as inputs to every unit
  # test because replication.cpp (in _lethe_test_link_srcs) calls
  # the gRPC stubs. We depend on lethe_proto_gen so the .cc files
  # exist before this target compiles.
  add_executable(${name} ${source} ${_lethe_test_link_srcs}
    ${LETHE_PROTO_GENERATED_SRCS})
  add_dependencies(${name} lethe_proto_gen)
  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/cache_server/include
    # routing.cpp (compiled into every unit test via _lethe_test_link_srcs)
    # includes blake3_helper.hpp which #includes blake3.h. Match the
    # include path used by the lethe_server target.
    ${CMAKE_SOURCE_DIR}/cache_server/third_party/blake3
    ${CMAKE_SOURCE_DIR}/cache_server/src
    # W4: generated proto headers, same path lethe_server uses.
    ${LETHE_PROTO_GEN_DIR})
  target_link_libraries(${name} PRIVATE
    Threads::Threads
    lethe_blake3
    # W4: gRPC + protobuf for the replication.cpp client calls.
    gRPC::grpc++
    protobuf::libprotobuf)
  # Generated proto sources can carry warnings auto-emitted code we
  # can't fix; strip -Werror promotion for them (mirrors the
  # lethe_server target).
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
      ${LETHE_PROTO_GENERATED_SRCS}
      PROPERTIES COMPILE_OPTIONS "-Wno-error")
  endif()
  # Force NDEBUG off so assert() actually executes under RelWithDebInfo.
  # Top-level CMakeLists defaults to RelWithDebInfo which defines NDEBUG,
  # which would compile every assert() in the test bodies to a no-op
  # and leave us with tests that "pass" trivially. -UNDEBUG keeps the
  # surrounding -O2 -g code-gen but re-enables the assertions.
  target_compile_options(${name} PRIVATE -UNDEBUG)
  if(_lethe_test_compile_options)
    target_compile_options(${name} PRIVATE ${_lethe_test_compile_options})
    target_link_options(${name}    PRIVATE ${_lethe_test_link_options})
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

lethe_add_unit_test(test_block_store  unit/test_block_store.cpp)
lethe_add_unit_test(test_cache        unit/test_cache.cpp)
lethe_add_unit_test(test_membership   unit/test_membership.cpp)
lethe_add_unit_test(test_routing      unit/test_routing.cpp)
lethe_add_unit_test(test_tiered_store unit/test_tiered_store.cpp)  # W7
lethe_add_unit_test(test_eviction     unit/test_eviction.cpp)      # W8
lethe_add_unit_test(test_metrics      unit/test_metrics.cpp)       # W10
