#pragma once
// Lethe — replication and read-repair (W4).
//
// On Insert, the owner streams the block to its R-1 replica successors on
// the consistent hash ring. On Lookup miss at the local node, we consult
// the router for replica peers and issue parallel StreamBlocks pulls; if
// any returns the block, we repair our local copy (read-repair).

#include <optional>
#include <string>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

class TieredStore;
class Router;
class KvTransport;
class Metrics;

class Replicator {
 public:
  // W5-6: transport may be null for tests / single-node bring-up that
  // don't exercise replication. When null, ReplicateOut returns an
  // empty list and FetchFromAny returns nullopt without any RPC.
  // W10: metrics may be null (tests); Record* calls are guarded.
  Replicator(std::string local_node_id,
             Router* router,
             TieredStore* store,
             KvTransport* transport,
             Metrics* metrics = nullptr);
  ~Replicator();

  // Push a block to all replica successors. Called from Insert after the
  // local tier write succeeds. Returns the list of peers the push was
  // QUEUED to (not ACKed; the W4 async-replication policy is
  // fire-and-forget — see docs/DECISIONS.md "Async replication policy
  // realized").
  std::vector<std::string> ReplicateOut(const KvBlock& block);

  // Attempt to fetch a block from any of the named replica peers. Returns
  // the materialized block on success, nullopt on full miss. On success the
  // caller may write the block locally to repair.
  std::optional<KvBlock> FetchFromAny(const BlockId& id,
                                      const std::vector<std::string>& peers);

  // Re-replicate blocks owned-but-under-replicated after a peer is suspected
  // dead. Called by Membership on cluster epoch change (W8).
  void TriggerReReplication(const std::vector<std::string>& lost_peers);

  // Connection pool management. Delegates to the underlying KvTransport
  // (each transport implementation manages its own per-peer state —
  // gRPC channels for GrpcStreamTransport, RC QPs for IbverbsTransport).
  // These remain on Replicator's public surface because LetheCache's
  // ctor uses them to pre-populate the pool from the static seed list
  // (so the first Insert doesn't pay a cold-channel-open RTT).
  void EnsurePeerClient(const std::string& peer_id, const std::string& address);
  void DropPeerClient(const std::string& peer_id);

 private:
  // W11.1: dispatch the next bounded batch of the in-progress
  // re-replication round (populated by TriggerReReplication). Driven on a
  // cadence by an internal sweep thread so a working set larger than one
  // batch fully reconverges across successive ticks — Finding B fix. Also
  // called once inline by TriggerReReplication for a fast first batch.
  // No-op when no round is active.
  void DrainReReplication();

  std::string local_node_id_;
  Router* router_;            // not owned
  TieredStore* store_;        // not owned
  KvTransport* transport_;    // not owned; nullable for single-node tests
  Metrics* metrics_;          // not owned; nullable
};

}  // namespace lethe
