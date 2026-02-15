#pragma once
// Lethe — thin BLAKE3 helper around the vendored C reference impl
// at cache_server/third_party/blake3/. Hides the C struct dance so
// the rest of the codebase says blake3_full(bytes) → Hash256.
//
// The vendored sources are BLAKE3 1.5.4 portable-only (no SIMD). Could swap
// in a SIMD build if hashing ever shows up in profiles; today the routing
// hash is one call per Lookup, well off the hot path.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

extern "C" {
#include "blake3.h"
}

namespace lethe {

inline std::array<std::uint8_t, 32> blake3_full(
    std::span<const std::uint8_t> input) noexcept {
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, input.data(), input.size());
  std::array<std::uint8_t, 32> out{};
  blake3_hasher_finalize(&h, out.data(), out.size());
  return out;
}

inline std::array<std::uint8_t, 32> blake3_full(
    std::string_view input) noexcept {
  return blake3_full(std::span<const std::uint8_t>{
      reinterpret_cast<const std::uint8_t*>(input.data()), input.size()});
}

}  // namespace lethe
