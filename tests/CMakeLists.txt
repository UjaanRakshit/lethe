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
  ${CMAKE_SOURCE_DIR}/cache_server/src/cache.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/membership.cpp
  # Subsystem stubs included so cache.cpp's null unique_ptr<X> dtors
  # link cleanly without requiring the full impl.
  ${CMAKE_SOURCE_DIR}/cache_server/src/routing.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/replication.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/eviction.cpp
  ${CMAKE_SOURCE_DIR}/cache_server/src/metrics.cpp
)

function(lethe_add_unit_test name source)
  add_executable(${name} ${source} ${_lethe_test_link_srcs})
  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/cache_server/include
    # routing.cpp (compiled into every unit test via _lethe_test_link_srcs)
    # includes blake3_helper.hpp which #includes blake3.h. Match the
    # include path used by the lethe_server target.
    ${CMAKE_SOURCE_DIR}/cache_server/third_party/blake3
    ${CMAKE_SOURCE_DIR}/cache_server/src)
  target_link_libraries(${name} PRIVATE Threads::Threads lethe_blake3)
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

lethe_add_unit_test(test_block_store unit/test_block_store.cpp)
lethe_add_unit_test(test_cache       unit/test_cache.cpp)
lethe_add_unit_test(test_membership  unit/test_membership.cpp)
