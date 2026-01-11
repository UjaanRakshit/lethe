#pragma once
// Lethe — replication and read-repair (W4).
//
// On Insert, the owner streams the block to its R-1 replica successors on
// the consistent hash ring. On Lookup miss at the local node, we consult
// the router for replica peers and issue parallel StreamBlocks pulls; if
// any returns the block, we repair our local copy (read-repair).

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

class TieredStore;
class Router;

// Thin client handle for talking to a peer's LetheCache gRPC service. Owned
// by Replicator; pooled per peer for connection reuse.
class PeerClient;

class Replicator {
 public:
  Replicator(std::string local_node_id,
             Router* router,
             TieredStore* store);
  ~Replicator();

  // Push a block to all replica successors. Called from Insert after the
  // local tier write succeeds. Returns the list of peers that ACKed.
  std::vector<std::string> ReplicateOut(const KvBlock& block);

  // Attempt to fetch a block from any of the named replica peers. Returns
  // the materialized block on success, nullopt on full miss. On success the
  // caller may write the block locally to repair.
  std::optional<KvBlock> FetchFromAny(const BlockId& id,
                                      const std::vector<std::string>& peers);

  // Re-replicate blocks owned-but-under-replicated after a peer is suspected
  // dead. Called by Membership on cluster epoch change (W8).
  void TriggerReReplication(const std::vector<std::string>& lost_peers);

  // Connection pool management.
  void EnsurePeerClient(const std::string& peer_id, const std::string& address);
  void DropPeerClient(const std::string& peer_id);

 private:
  std::string local_node_id_;
  Router* router_;          // not owned
  TieredStore* store_;      // not owned

  std::mutex pool_mu_;
  std::unordered_map<std::string, std::unique_ptr<PeerClient>> peer_clients_;
};

}  // namespace lethe
