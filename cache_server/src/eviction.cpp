// Lethe — eviction implementation (W8).
//
// W1 ships only the destructor stub so LetheCache's unique_ptr<Evictor>
// is destructible. Real SIEVE pass, MarkVisited, and gossip handler
// land in W8.

#include "lethe/eviction.hpp"

namespace lethe {

Evictor::Evictor(EvictionConfig cfg,
                 TieredStore* store,
                 Membership* membership,
                 std::string local_node_id)
    : cfg_(cfg),
      store_(store),
      membership_(membership),
      local_node_id_(std::move(local_node_id)) {}

Evictor::~Evictor() {
  // Stop the loop if it was running. W1 never Starts it; this is for
  // future safety once Evictor is actually constructed.
  running_.store(false, std::memory_order_relaxed);
  if (thread_.joinable()) thread_.join();
}

// W8: Start, Shutdown, RunPass, OnPeerEviction, MarkVisited, Loop,
// PickVictims, BroadcastEvictions.

}  // namespace lethe
