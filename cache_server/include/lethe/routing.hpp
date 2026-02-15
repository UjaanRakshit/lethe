#pragma once
// Prefix-aware consistent hashing router.
//
// Each peer is mapped to V virtual nodes on a 64-bit hash ring. To resolve a
// block: hash the block's content hash and walk the ring to find the owner.
// Because block IDs are prefix-chained, two requests sharing a prefix produce
// identical block IDs for the first k blocks and route to the same node,
// which is the entire point.
//
// Membership changes trigger a ring rebuild; the rebuild is kept cheap by
// using a sorted std::vector<std::pair<uint64_t, peer_idx>> rather than a tree.

#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

struct RouteResult {
  std::string primary;                  // first replica
  std::vector<std::string> replicas;    // next R-1 distinct peers on ring
};

class Router {
 public:
  Router(std::string local_node_id,
         std::uint32_t virtual_nodes_per_peer,
         std::uint32_t replication_factor);

  // Replace the active peer set. Rebuilds the ring atomically.
  void SetPeers(std::vector<std::string> peer_ids);

  // Routes a block ID to its primary owner and replica successors.
  RouteResult Route(const BlockId& id) const;

  // Returns true iff the local node is the primary owner of `id`.
  bool IsLocalPrimary(const BlockId& id) const;

  // Returns true iff the local node is among the R replicas for `id`.
  bool IsLocalReplica(const BlockId& id) const;

  // Snapshot of the peer set for introspection.
  std::vector<std::string> peers() const;

 private:
  struct RingEntry {
    std::uint64_t hash;
    std::uint32_t peer_idx;
  };

  // HashBlock hashes `id.hash` and only `id.hash`. The (layer, head_group,
  // model_id) disambiguators are deliberately ignored so that every layer
  // of a prefix block routes to the same primary owner; this is what makes
  // "blocks sharing a primary → single Lookup RPC" batching work. The
  // Python client mirror in lethe_client/routing.py must follow the same
  // rule. See types.hpp BlockId for the longer note.
  std::uint64_t HashBlock(const BlockId& id) const;
  std::uint64_t HashVirtualNode(const std::string& peer, std::uint32_t vn) const;

  std::string local_node_id_;
  std::uint32_t vnodes_per_peer_;
  std::uint32_t replication_factor_;

  mutable std::shared_mutex mu_;
  std::vector<std::string> peers_;
  std::vector<RingEntry> ring_;   // sorted by hash; binary search on lookup
};

}  // namespace lethe
