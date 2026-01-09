#pragma once
// Lethe — common types shared across cache subsystems.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lethe {

// Storage tier. Order matters: lower index = faster + smaller.
enum class Tier : std::uint8_t {
  HBM = 0,    // GPU memory, fastest, smallest
  DRAM = 1,   // host memory
  SSD = 2,    // disk-backed, slowest, largest
};

// 32-byte content hash. BLAKE3 — see DESIGN.md §1 and CLAUDE.md "What to
// never do." The C++ and Python implementations of `chained_block_hash`
// MUST produce identical bytes for the same `(prev_hash, tokens)`.
using Hash256 = std::array<std::byte, 32>;

// Identifies a single KV block.
//
// Routing note: the consistent-hash router (see routing.hpp) hashes
// `hash` and `hash` only. `layer`, `head_group`, and `model_id` are
// disambiguators for the in-memory store map — they keep KV bytes for
// different layers from colliding under the same content key — but they
// do NOT participate in routing. All layers of one prefix block therefore
// share a primary owner. This is intentional: it preserves
// "blocks sharing a primary → single Lookup RPC" batching.
struct BlockId {
  Hash256 hash;         // chained prefix hash, see proto/lethe.proto
  std::uint32_t layer = 0;
  std::uint32_t head_group = 0;
  std::uint32_t model_id = 0;

  bool operator==(const BlockId&) const noexcept = default;
};

// Hash functor for unordered containers. NOT a content hash — this is an
// in-memory combiner over the already-strong BLAKE3 content digest plus
// the (layer, head_group, model_id) disambiguators. The two language
// runtimes do not need to agree on this; see CLAUDE.md.
struct BlockIdHash {
  std::size_t operator()(const BlockId& b) const noexcept {
    std::size_t h;
    std::memcpy(&h, b.hash.data(), sizeof(h));
    // boost-style hash_combine — order-sensitive, unlike a bare XOR.
    auto combine = [](std::size_t& seed, std::size_t v) noexcept {
      seed ^= v + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
    };
    combine(h, static_cast<std::size_t>(b.layer));
    combine(h, static_cast<std::size_t>(b.head_group));
    combine(h, static_cast<std::size_t>(b.model_id));
    return h;
  }
};

// A materialized KV block — owned bytes + identity + tier of origin.
struct KvBlock {
  BlockId id;
  std::vector<std::byte> data;
  Tier tier = Tier::DRAM;
  std::uint64_t inserted_epoch = 0;
};

// Mirrors proto/lethe.proto `message PeerStatus`. Field-for-field, in
// declaration order matching the proto field numbers.
struct PeerStatus {
  std::string node_id;            // proto field 1
  std::uint64_t last_seen_epoch = 0;  // proto field 2
  bool suspected = false;         // proto field 3
};

// Reply payload for a Heartbeat RPC. Mirrors proto/lethe.proto
// `message HeartbeatResponse` field-for-field:
//
//   alive_peers   ←→  repeated PeerStatus alive_peers = 1;
//   cluster_epoch ←→  uint64 cluster_epoch = 2;
//
// The gRPC shim copies fields one-to-one; no information is lost in
// either direction.
struct HeartbeatReply {
  std::vector<PeerStatus> alive_peers;
  std::uint64_t cluster_epoch = 0;
};

// Per-block index entry kept by the block store (no payload, just metadata).
//
// `visited` is the SIEVE bit: set by Evictor::MarkVisited() (called from
// every BlockStore::Get hit), cleared by the eviction hand as it scans.
// See eviction.hpp for the algorithm.
struct BlockMeta {
  BlockId id;
  Tier tier;
  std::size_t size_bytes;
  std::uint64_t last_access_epoch;
  std::uint64_t insert_epoch;
  bool visited = false;
};

}  // namespace lethe
