// Consistent-hash router.
//
// Implements the bit-compatibility contract with
// client/lethe_client/routing.py — same hash function (BLAKE3), same
// virtual-node count (128 default), same per-vnode key format (`f"{peer}#{vn}"`
// UTF-8 bytes), same uint64 derivation (first 8 bytes of digest,
// little-endian). Cross-language equivalence is asserted by
// tests/correctness/test_hash_compat.py via the hash_compat_driver tool.
//
// Routing of a BlockId:
//   - HashBlock takes the FIRST 8 BYTES of id.hash (already a 32-byte BLAKE3
//     digest from the chained-prefix hash), little-endian uint64. No
//     additional hashing. The Python side does the same:
//     int.from_bytes(block_hash[:8], "little").
//   - id.layer / head_group / model_id are intentionally NOT consulted. See
//     DESIGN.md §1 "What the router hashes" and types.hpp BlockId.
//
// Ring lookup: binary search of the sorted vector for the first entry whose
// hash >= target, with wraparound. Replicas are the next R-1 DISTINCT peers
// walking the ring clockwise.

#include "lethe/routing.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>

#include "blake3_helper.hpp"

namespace lethe {

namespace {

std::uint64_t le_u64_from_digest(const std::array<std::uint8_t, 32>& d) {
  // First 8 bytes, little-endian. memcpy guards against strict-
  // aliasing UB on platforms where unaligned loads would otherwise
  // be invoked through a uint64_t*. On x86_64 it compiles to a
  // single mov.
  std::uint64_t v = 0;
  std::memcpy(&v, d.data(), sizeof(v));
  return v;
}

}  // namespace

Router::Router(std::string local_node_id,
               std::uint32_t virtual_nodes_per_peer,
               std::uint32_t replication_factor)
    : local_node_id_(std::move(local_node_id)),
      vnodes_per_peer_(virtual_nodes_per_peer),
      replication_factor_(replication_factor) {}

std::uint64_t Router::HashBlock(const BlockId& id) const {
  // First 8 bytes of the content hash; little-endian uint64. The
  // BLAKE3 digest is already cryptographically uniform, so any 8
  // bytes are equally good — we pick the leading 8 to match the
  // Python client.
  std::uint64_t v = 0;
  std::memcpy(&v, id.hash.data(), sizeof(v));
  return v;
}

std::uint64_t Router::HashVirtualNode(const std::string& peer,
                                       std::uint32_t vn) const {
  // Key format: literal f-string f"{peer}#{vn}" UTF-8 — verified bit-for-bit
  // against the Python HashRing's encoding. Changing this breaks the
  // cross-language invariant.
  std::string key;
  key.reserve(peer.size() + 1 + 10);
  key += peer;
  key += '#';
  key += std::to_string(vn);
  auto digest = blake3_full(std::string_view{key});
  return le_u64_from_digest(digest);
}

void Router::SetPeers(std::vector<std::string> peer_ids) {
  // Build the new ring outside the lock; swap atomically. Lookup
  // calls are reader-favored via shared_mutex, so building eagerly
  // means writers don't stall readers any longer than the swap.
  std::vector<RingEntry> ring;
  ring.reserve(peer_ids.size() * vnodes_per_peer_);
  for (std::uint32_t i = 0; i < peer_ids.size(); ++i) {
    for (std::uint32_t vn = 0; vn < vnodes_per_peer_; ++vn) {
      RingEntry e;
      e.hash = HashVirtualNode(peer_ids[i], vn);
      e.peer_idx = i;
      ring.push_back(e);
    }
  }
  std::sort(ring.begin(), ring.end(),
            [](const RingEntry& a, const RingEntry& b) {
              return a.hash < b.hash;
            });

  std::unique_lock<std::shared_mutex> lock(mu_);
  peers_ = std::move(peer_ids);
  ring_ = std::move(ring);
}

RouteResult Router::Route(const BlockId& id) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  RouteResult out;
  if (ring_.empty() || peers_.empty()) {
    return out;
  }
  const std::uint64_t target = HashBlock(id);

  // Binary search for first ring entry with hash >= target. Python
  // mirror in HashRing.route uses the same lo/hi split.
  std::size_t lo = 0;
  std::size_t hi = ring_.size();
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (ring_[mid].hash < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  std::size_t idx = lo % ring_.size();

  // Walk clockwise picking the next R DISTINCT peers (primary +
  // R-1 replicas). std::unordered_set ensures we don't double-pick
  // the same peer via two of its virtual nodes.
  std::vector<std::string> picked;
  std::unordered_set<std::string> seen;
  const std::uint32_t want = std::max<std::uint32_t>(1, replication_factor_);
  const std::size_t ring_size = ring_.size();
  for (std::size_t step = 0;
       step < ring_size && picked.size() < want;
       ++step) {
    const auto& peer = peers_[ring_[(idx + step) % ring_size].peer_idx];
    if (seen.insert(peer).second) {
      picked.push_back(peer);
    }
  }
  if (picked.empty()) {
    return out;
  }
  out.primary = picked.front();
  out.replicas.assign(picked.begin() + 1, picked.end());
  return out;
}

bool Router::IsLocalPrimary(const BlockId& id) const {
  const auto r = Route(id);
  return r.primary == local_node_id_;
}

bool Router::IsLocalReplica(const BlockId& id) const {
  const auto r = Route(id);
  if (r.primary == local_node_id_) return true;
  for (const auto& p : r.replicas) {
    if (p == local_node_id_) return true;
  }
  return false;
}

std::vector<std::string> Router::peers() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return peers_;
}

}  // namespace lethe
