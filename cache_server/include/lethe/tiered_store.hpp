#pragma once
// Tiered storage composing three layers: HBM (tier 0), DRAM (tier 1), SSD
// (tier 2). HBM and DRAM use BlockStore (in-memory hashtable backed by
// pinned-host or device memory); SSD uses SsdBlockStore (mmap-backed slot
// allocator).
//
// On Get, blocks may be promoted to a faster tier; under eviction pressure,
// demoted to a slower one before being dropped entirely.
//
// Tier selection on Put: tier_hint from the client is a request; the store
// may override based on current pressure.
//
// HBM allocator strategy:
//   - Default build (no `LETHE_ENABLE_CUDA`): the HBM tier is backed by
//     plain heap memory (the same BlockStore the DRAM tier uses). The
//     benchmark output MUST tag the result "HBM=pinned-host" so it isn't
//     passed off as device-memory performance.
//   - With `-DLETHE_ENABLE_CUDA=ON`: the HBM BlockStore uses real device
//     memory via `cudaMalloc` and zero-copy paths to vLLM's PagedAttention
//     buffers. The goal, but not required.
//   - If `hbm_bytes == 0`, the tier is disabled entirely and the HBM
//     BlockStore pointer is null.
//
// Lifetime contract: GetResult.data is OWNED bytes (a vector), not a
// borrowed span. The SSD tier can't safely lend spans into mmap'd memory
// across Erase/Put churn — the slot may be reused for an unrelated block
// before the caller serializes the bytes. Uniform ownership is the simplest
// invariant; the per-Get copy is one memcpy of <= slot_bytes (default
// 64 KiB), negligible at our scale.

#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lethe/block_store.hpp"
#include "lethe/types.hpp"

namespace lethe {

class SsdBlockStore;

struct TieredStoreConfig {
  std::size_t hbm_bytes = 0;
  std::size_t dram_bytes = 32ULL << 30;
  std::size_t ssd_bytes = 256ULL << 30;
  std::string ssd_path;
  bool enable_promotion = true;
  std::uint32_t promotion_access_threshold = 2;  // accesses before promote
  std::size_t ssd_slot_bytes = 64ULL * 1024;     // per-slot size on disk
};

struct GetResult {
  std::vector<std::byte> data;   // owned copy of the block payload
  Tier tier_found;
  bool promoted;
};

class TieredStore {
 public:
  explicit TieredStore(TieredStoreConfig cfg);
  ~TieredStore();

  TieredStore(const TieredStore&) = delete;
  TieredStore& operator=(const TieredStore&) = delete;

  // Returns nullopt on miss. May promote the block to a faster tier as
  // a side effect when access count crosses promotion_access_threshold.
  std::optional<GetResult> Get(const BlockId& id);

  // Inserts a block at the best-fit tier given the hint and current
  // pressure. Returns the tier where the block actually landed. If no
  // tier has space, returns nullopt.
  std::optional<Tier> Put(KvBlock block, Tier hint);

  // Demotes the block to the next slower tier. Used by Evictor before
  // drop. Returns true on success, false if already at SSD (caller
  // should drop), or if the next tier has no space (caller may try a
  // different victim or proceed to drop).
  bool Demote(const BlockId& id);

  // Removes the block from all tiers AND wipes its entry from
  // access_counts_. Callers (Evictor, replication drop) must use this
  // rather than reaching into individual BlockStores, so the access-
  // count map never outlives the block.
  std::size_t Erase(const BlockId& id);

  // Per-tier introspection for metrics + eviction decisions. Snapshot's
  // BlockMeta carries the SIEVE visited bit; the Evictor reads it on each
  // scan pass.
  std::size_t used_bytes(Tier t) const;
  std::size_t capacity_bytes(Tier t) const;
  std::vector<BlockMeta> Snapshot(Tier t) const;

  // SIEVE support. visited_ is a single set of BlockIds the cache has seen
  // recently; the Evictor's scan reads it via Snapshot (which overlays the
  // bit onto each BlockMeta). MarkVisited fires from Get on every hit;
  // ClearVisited fires from the Evictor when SIEVE gives a block a second
  // chance. The visited entry is wiped on Erase per the same access_counts_
  // contract.
  void MarkVisited(const BlockId& id);
  void ClearVisited(const BlockId& id);

  // Test-only seam: read the access counter for a block. Returns 0 if
  // the block has no entry (either never seen or wiped on Erase).
  std::uint32_t access_count_for_testing(const BlockId& id) const;
  // Test-only seam for the SIEVE visited bit.
  bool visited_for_testing(const BlockId& id) const;

 private:
  // Try to insert into `tier`'s BlockStore (HBM/DRAM) or SsdBlockStore.
  // Returns true on success.
  bool PutToTier(KvBlock block, Tier tier);

  // Erase from the named tier only; doesn't touch other tiers or
  // access_counts_.
  std::size_t EraseFromTier(const BlockId& id, Tier tier);

  TieredStoreConfig cfg_;
  std::unique_ptr<BlockStore> hbm_;
  std::unique_ptr<BlockStore> dram_;
  std::unique_ptr<SsdBlockStore> ssd_;

  // Per-block access counter, used for promotion decisions. Hashed in a
  // separate map to keep BlockStore lean.
  std::unordered_map<BlockId, std::uint32_t, BlockIdHash> access_counts_;
  // SIEVE visited bits. Same shared_mutex as access_counts_ — both are
  // touched on every Get, so one lock acquisition handles both. Membership
  // in `visited_` ⇒ "visited bit set"; absence ⇒ cleared.
  std::unordered_set<BlockId, BlockIdHash> visited_;
  mutable std::shared_mutex counts_mu_;
};

}  // namespace lethe
