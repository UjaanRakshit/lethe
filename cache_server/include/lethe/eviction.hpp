#pragma once
// Lethe — eviction (W8).
//
// Per-node SIEVE eviction (NSDI '24).
//
// SIEVE keeps blocks in FIFO insertion order with a 1-bit `visited` flag
// per block. A single "hand" pointer walks the FIFO from oldest toward
// newest. On each step:
//
//   if hand->visited:        clear visited; advance hand          (give a 2nd chance)
//   else:                    evict hand; advance hand to next     (the victim)
//
// `visited` is set by Evictor::MarkVisited(), which is called from every
// BlockStore::Get hit (see block_store.hpp). It is cleared only by the
// scanning hand. This is the real SIEVE — NOT "approximate LRU by
// timestamp" — and that distinction matters for the comparison vs. LRU we
// want to be able to defend in the design doc.
//
// Cluster-wide coordination: when a block is evicted, broadcast (gossip)
// the eviction so that peers' read-repair routing tables don't try to
// fetch from us. Best-effort; the worst case is one wasted RPC.

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

class TieredStore;
class Membership;

struct EvictionConfig {
  std::size_t high_watermark_pct = 90;  // begin eviction
  std::size_t low_watermark_pct = 75;   // stop eviction
  std::chrono::milliseconds scan_interval{500};
  bool broadcast_evictions = true;
};

class Evictor {
 public:
  Evictor(EvictionConfig cfg,
          TieredStore* store,
          Membership* membership,
          std::string local_node_id);
  ~Evictor();

  Evictor(const Evictor&) = delete;
  Evictor& operator=(const Evictor&) = delete;

  // Idempotent. Spawns one eviction thread per real tier (HBM is only
  // started if hbm capacity > 0; SSD only if ssd capacity > 0).
  void Start();
  void Shutdown();

  // Force a single eviction pass for a specific tier; returns blocks
  // evicted from that tier. Used by tests; production callers wait for
  // the scan-interval-driven loop.
  struct PassResult {
    std::size_t hbm_evicted = 0;
    std::size_t dram_evicted = 0;
    std::size_t ssd_evicted = 0;
    std::size_t bytes_freed = 0;
    std::size_t blocks_broadcast = 0;
  };
  PassResult RunPassForTier(Tier tier);

  // Called from Cache::OnEvictBroadcast — record that a peer evicted
  // these blocks. Read-repair (Replicator::FetchFromAny) currently
  // does NOT consult this; the W8 wire is in place and the optimization
  // is a small follow-up if profiling shows wasted cross-cluster
  // fetches against just-evicted nodes.
  void OnPeerEviction(const std::vector<BlockId>& evicted,
                      const std::string& peer);

  // Test-only seam: count of how many block IDs are tracked as
  // "this peer recently evicted." Returns 0 if peer is unknown.
  std::size_t peer_evicted_count_for_testing(const std::string& peer) const;

 private:
  // Helper: fire one EvictBroadcast RPC per known peer carrying ALL
  // the block IDs evicted in the current pass. Best-effort; failures
  // are swallowed per CLAUDE.md rule 2.
  void BroadcastEvictionsToPeers(const std::vector<BlockId>& evicted);

  EvictionConfig cfg_;
  TieredStore* store_;        // not owned
  Membership* membership_;    // not owned
  std::string local_node_id_;

  // Per-instance state (threads, hand pointers, peer stubs, peer-
  // eviction tracking) lives in a TU-local registry keyed by
  // `this` so gRPC and std::thread types stay out of this header.
  // See eviction.cpp `struct EvictorState` + `Registry`. Same pattern
  // Replicator uses (replication.cpp `ReplicatorPoolState`).
};

}  // namespace lethe
