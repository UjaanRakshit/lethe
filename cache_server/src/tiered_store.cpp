// Lethe — tiered storage (W1: DRAM-only).
//
// W1 scope: only the DRAM tier is actually backed by storage. HBM is
// declared in the header but stays null unless LETHE_ENABLE_CUDA + a
// non-zero hbm_bytes — neither is W1 work. SSD is W7. For W1 the
// composition reduces to a thin wrapper around a single BlockStore.
//
// The class's full behavior (promotion on Get, demotion before
// eviction, tier-hint negotiation) lights up in W7 when there's more
// than one real tier to move blocks between. For W1 we keep the API
// shape and route everything to DRAM.

#include "lethe/tiered_store.hpp"

#include <utility>

namespace lethe {

TieredStore::TieredStore(TieredStoreConfig cfg) : cfg_(std::move(cfg)) {
  // W1: only DRAM is constructed. HBM and SSD remain null.
  // W7 will replace this with real allocator selection and
  // (cfg_.hbm_bytes > 0 && LETHE_ENABLE_CUDA) tier construction.
  dram_ = std::make_unique<BlockStore>(Tier::DRAM, cfg_.dram_bytes);
}

TieredStore::~TieredStore() = default;

std::optional<GetResult> TieredStore::Get(const BlockId& id) {
  // W1: no HBM, no SSD, no promotion — just try DRAM.
  if (auto data = dram_->Get(id); data.has_value()) {
    // Bump the access counter under its own lock. W7 reads this to
    // decide whether to promote on the next eviction-pressure event.
    {
      std::unique_lock<std::shared_mutex> lock(counts_mu_);
      ++access_counts_[id];
    }
    GetResult r;
    r.data = *data;
    r.tier_found = Tier::DRAM;
    r.promoted = false;
    return r;
  }
  return std::nullopt;
}

Tier TieredStore::Put(KvBlock block, Tier /*hint*/) {
  // W1: hint is ignored; we only have DRAM. Capacity overflow at the
  // BlockStore level returns false; the caller sees the rejection via
  // used_bytes not changing. W7 will expand this to honor the hint
  // and route across tiers.
  block.tier = Tier::DRAM;
  dram_->Put(std::move(block));
  return Tier::DRAM;
}

bool TieredStore::Demote(const BlockId& /*id*/) {
  // W1: only DRAM exists; nothing to demote to. Real impl lands W7.
  return false;
}

std::size_t TieredStore::Erase(const BlockId& id) {
  // Erase from every tier and wipe the per-block access counter so
  // access_counts_ never outlives the block (W0 audit contract carried
  // forward into the implementation).
  std::size_t freed = 0;
  if (dram_) freed += dram_->Erase(id);
  {
    std::unique_lock<std::shared_mutex> lock(counts_mu_);
    access_counts_.erase(id);
  }
  return freed;
}

std::size_t TieredStore::used_bytes(Tier t) const {
  switch (t) {
    case Tier::DRAM: return dram_ ? dram_->used_bytes() : 0;
    case Tier::HBM:  return 0;  // W1: not constructed.
    case Tier::SSD:  return 0;  // W7.
  }
  return 0;
}

std::size_t TieredStore::capacity_bytes(Tier t) const {
  switch (t) {
    case Tier::DRAM: return dram_ ? dram_->capacity_bytes() : 0;
    case Tier::HBM:  return 0;
    case Tier::SSD:  return 0;
  }
  return 0;
}

std::vector<BlockMeta> TieredStore::Snapshot(Tier t) const {
  if (t == Tier::DRAM && dram_) return dram_->Snapshot();
  return {};
}

}  // namespace lethe
