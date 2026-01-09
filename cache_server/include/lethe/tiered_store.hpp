#pragma once
// Lethe — tiered storage (W7).
//
// Composes three BlockStores: HBM (tier 0), DRAM (tier 1), SSD (tier 2).
// On Get, blocks may be promoted to a faster tier; on eviction pressure,
// demoted to a slower one before being dropped entirely.
//
// Tier selection on Put: tier_hint from the client is a request; the store
// may override based on current pressure and (eventually) access frequency.
//
// HBM allocator strategy:
//   - Default build (no `LETHE_ENABLE_CUDA`): the HBM tier is backed by
//     pinned host memory (`cudaMallocHost`-shaped allocator, or plain
//     mlock'd pages if CUDA isn't linked at all). This lets the tier
//     plumbing be exercised without a GPU; capacity is still bounded by
//     `hbm_bytes`, and benchmarks must note "HBM=pinned-host" in their
//     output so the number isn't passed off as device-memory performance.
//   - With `-DLETHE_ENABLE_CUDA=ON`: the HBM BlockStore uses real device
//     memory via `cudaMalloc` and zero-copy paths to vLLM's PagedAttention
//     buffers. This path is the W7 stretch goal, not a W1 requirement.
//   - If `hbm_bytes == 0`, the tier is disabled entirely and the HBM
//     BlockStore pointer is null.

#include <memory>
#include <span>

#include "lethe/block_store.hpp"
#include "lethe/types.hpp"

namespace lethe {

struct TieredStoreConfig {
  std::size_t hbm_bytes = 0;
  std::size_t dram_bytes = 32ULL << 30;
  std::size_t ssd_bytes = 256ULL << 30;
  std::string ssd_path;
  bool enable_promotion = true;
  std::uint32_t promotion_access_threshold = 2;  // accesses before promote
};

struct GetResult {
  std::span<const std::byte> data;
  Tier tier_found;
  bool promoted;
};

class TieredStore {
 public:
  explicit TieredStore(TieredStoreConfig cfg);
  ~TieredStore();

  // Returns nullopt on miss. May promote the block to a faster tier as a
  // side effect; callers should treat the returned span as ephemeral.
  std::optional<GetResult> Get(const BlockId& id);

  // Inserts a block at the best-fit tier given the hint and current pressure.
  // Returns the tier where the block actually landed.
  Tier Put(KvBlock block, Tier hint);

  // Demotes the block to the next slower tier. Used by Evictor before drop.
  // Returns true on success, false if already at SSD (drop instead).
  bool Demote(const BlockId& id);

  // Removes the block from all tiers AND wipes its entry from
  // access_counts_. Callers (Evictor, replication drop) must use this
  // rather than reaching into individual BlockStores, so the access-count
  // map never outlives the block.
  std::size_t Erase(const BlockId& id);

  // Per-tier introspection for metrics + eviction decisions.
  std::size_t used_bytes(Tier t) const;
  std::size_t capacity_bytes(Tier t) const;
  std::vector<BlockMeta> Snapshot(Tier t) const;

 private:
  TieredStoreConfig cfg_;
  std::unique_ptr<BlockStore> hbm_;
  std::unique_ptr<BlockStore> dram_;
  std::unique_ptr<BlockStore> ssd_;       // SSD-backed; see ssd_store.hpp

  // Per-block access counter, used for promotion decisions. Hashed in a
  // separate map to keep BlockStore lean.
  std::unordered_map<BlockId, std::uint32_t, BlockIdHash> access_counts_;
  mutable std::shared_mutex counts_mu_;
};

}  // namespace lethe
