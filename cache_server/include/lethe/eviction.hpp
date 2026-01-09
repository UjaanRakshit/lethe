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
#include <thread>
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

  void Start();
  void Shutdown();

  // Force a single eviction pass; returns blocks evicted from each tier.
  struct PassResult {
    std::size_t hbm_evicted = 0;
    std::size_t dram_evicted = 0;
    std::size_t ssd_evicted = 0;
    std::size_t bytes_freed = 0;
  };
  PassResult RunPass();

  // Set the SIEVE visited bit on a block. Called from BlockStore::Get on
  // every hit. O(1); thread-safe under the same mutex that guards the
  // tier's block map.
  void MarkVisited(const BlockId& id, Tier tier);

  // Called from Cache::OnEvictBroadcast — record that a peer evicted these
  // blocks so we don't try to read-repair from them.
  void OnPeerEviction(const std::vector<BlockId>& evicted,
                      const std::string& peer);

 private:
  void Loop();
  std::vector<BlockId> PickVictims(Tier tier, std::size_t bytes_needed);
  void BroadcastEvictions(const std::vector<BlockId>& evicted);

  EvictionConfig cfg_;
  TieredStore* store_;
  Membership* membership_;
  std::string local_node_id_;

  std::atomic<bool> running_{false};
  std::thread thread_;

  // SIEVE state per tier: a circular pointer into the snapshot list.
  std::size_t sieve_hand_hbm_ = 0;
  std::size_t sieve_hand_dram_ = 0;
  std::size_t sieve_hand_ssd_ = 0;
};

}  // namespace lethe
