// Lethe — hash compatibility driver (W3).
//
// Cross-language equivalence harness for chained_block_hash. Reads
// lines from stdin in either of two formats and prints the resulting
// 32-byte BLAKE3 digest as 64-character lowercase hex per line.
//
// Modes:
//   default      — each line: "<prev_hash_hex> <token_id> <token_id> ..."
//                  Computes BLAKE3(prev_bytes || pack<u32 LE>(tokens))
//                  for one block; mirrors
//                  lethe_client.routing.chained_block_hash.
//   --mode=ring_key — each line: "<peer> <vn>"
//                  Computes BLAKE3(f"{peer}#{vn}".encode("utf-8"));
//                  mirrors lethe_client.routing.HashRing's per-vnode
//                  digest. Used by the cross-language ring-key test.
//
// Used by tests/correctness/test_hash_compat.py via subprocess. The
// W0 invariant requires bit-equal output between this driver and the
// Python reference.
//
// Build: see cache_server/CMakeLists.txt → add_executable
// hash_compat_driver. The binary lands at
// build/cache_server/hash_compat_driver and the Python test looks for
// it at build/tests/hash_compat_driver — symlinked at runtime by
// tests/CMakeLists.txt or copied during the build.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include "blake3.h"
}

namespace {

bool hex_to_bytes(const std::string& hex, std::array<std::uint8_t, 32>& out) {
  if (hex.size() != 64) return false;
  auto nyb = [](char c, int& v) {
    if (c >= '0' && c <= '9') { v = c - '0'; return true; }
    if (c >= 'a' && c <= 'f') { v = c - 'a' + 10; return true; }
    if (c >= 'A' && c <= 'F') { v = c - 'A' + 10; return true; }
    return false;
  };
  for (std::size_t i = 0; i < 32; ++i) {
    int hi, lo;
    if (!nyb(hex[2 * i], hi) || !nyb(hex[2 * i + 1], lo)) return false;
    out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
  }
  return true;
}

std::string bytes_to_hex(const std::uint8_t* data, std::size_t n) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    out[2 * i]     = kHex[data[i] >> 4];
    out[2 * i + 1] = kHex[data[i] & 0xF];
  }
  return out;
}

int run_default_mode() {
  // Each input line: <prev_hash_hex> [<token_id> ...]
  // Emit BLAKE3(prev_bytes || pack<u32 LE>(tokens)) hex.
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string prev_hex;
    iss >> prev_hex;
    std::array<std::uint8_t, 32> prev{};
    if (!hex_to_bytes(prev_hex, prev)) {
      std::cerr << "hash_compat_driver: bad prev_hash_hex on line: "
                << line << "\n";
      return 1;
    }
    std::vector<std::uint32_t> tokens;
    std::uint32_t t;
    while (iss >> t) tokens.push_back(t);

    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, prev.data(), prev.size());
    // Tokens packed little-endian uint32, matching Python's
    // struct.pack("<I", t). On x86_64 this is a memcpy.
    for (auto v : tokens) {
      std::uint8_t buf[4];
      buf[0] = v & 0xFF;
      buf[1] = (v >> 8) & 0xFF;
      buf[2] = (v >> 16) & 0xFF;
      buf[3] = (v >> 24) & 0xFF;
      blake3_hasher_update(&h, buf, 4);
    }
    std::array<std::uint8_t, 32> out{};
    blake3_hasher_finalize(&h, out.data(), out.size());
    std::cout << bytes_to_hex(out.data(), out.size()) << "\n";
  }
  return 0;
}

int run_ring_key_mode() {
  // Each input line: <peer> <vn>
  // Emit BLAKE3(f"{peer}#{vn}".encode("utf-8")) hex — same key format
  // as Python HashRing.set_peers.
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string peer;
    std::uint32_t vn;
    if (!(iss >> peer >> vn)) {
      std::cerr << "hash_compat_driver --ring_key: bad line: " << line << "\n";
      return 1;
    }
    std::string key = peer + "#" + std::to_string(vn);
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, key.data(), key.size());
    std::array<std::uint8_t, 32> out{};
    blake3_hasher_finalize(&h, out.data(), out.size());
    std::cout << bytes_to_hex(out.data(), out.size()) << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool ring_key = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode=ring_key") == 0) ring_key = true;
  }
  return ring_key ? run_ring_key_mode() : run_default_mode();
}
