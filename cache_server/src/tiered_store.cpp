// Lethe — tiered storage (W7).
//
// Composes HBM + DRAM (in-memory BlockStore) + SSD (SsdBlockStore,
// mmap-backed). Tier-aware Get/Put/Demote/Erase. Promotion on Get when
// access_counts_[id] crosses the configured threshold AND the faster
// tier has space (best-effort — no eviction triggering from here; W8's
// Evictor handles capacity pressure).
//
// Threading: access_counts_ has its own shared_mutex. Each underlying
// BlockStore / SsdBlockStore manages its own mutex. We never hold two
// of these at once.
//
// Why HBM uses plain BlockStore in the default build:
//   The W0 design says the HBM tier is pinned-host memory unless
//   LETHE_ENABLE_CUDA=ON pulls in cudaMalloc. For W7 we don't have CUDA
//   wired; the default-build HBM is therefore "fast in-memory tier
//   logically, regular heap physically." Benchmarks tag this as
//   `HBM=pinned-host` per the DESIGN.md note so the number isn't
//   misrepresented.

#include "lethe/tiered_store.hpp"

#include <filesystem>
#include <utility>

#include "lethe/ssd_block_store.hpp"

namespace lethe {

TieredStore::TieredStore(TieredStoreConfig cfg) : cfg_(std::move(cfg)) {
  // HBM: only constructed if non-zero capacity. The "pinned host vs
  // cudaMalloc" switch lives in the BlockStore allocator at the
  // LETHE_ENABLE_CUDA=ON path — not in W7 scope.
  if (cfg_.hbm_bytes > 0) {
    hbm_ = std::make_unique<BlockStore>(Tier::HBM, cfg_.hbm_bytes);
  }
  // DRAM is the always-on tier; the cache makes no sense without it.
  dram_ = std::make_unique<BlockStore>(Tier::DRAM, cfg_.dram_bytes);
  // SSD: only constructed if non-zero capacity AND a path is set. The
  // single-node W1.4 + W4 unit tests run with ssd_bytes=0; they go
  // through this path unchanged.
  if (cfg_.ssd_bytes > 0 && !cfg_.ssd_path.empty()) {
    std::filesystem::path file = std::filesystem::path(cfg_.ssd_path);
    // Append a stable filename so a single ssd_path can host the data
    // file directly OR contain it (caller passes a dir or a file path
    // and we DTRT).
    if (std::filesystem::is_directory(file)) {
      file /= "lethe_ssd.bin";
    }
    ssd_ = std::make_unique<SsdBlockStore>(file, cfg_.ssd_bytes,
                                           cfg_.ssd_slot_bytes);
  }
}

TieredStore::~TieredStore() = default;

bool TieredStore::PutToTier(KvBlock block, Tier tier) {
  switch (tier) {
    case Tier::HBM:
      return hbm_ ? hbm_->Put(std::move(block)) : false;
    case Tier::DRAM:
      return dram_ ? dram_->Put(std::move(block)) : false;
    case Tier::SSD:
      return ssd_ ? ssd_->Put(std::move(block)) : false;
  }
  return false;
}

std::size_t TieredStore::EraseFromTier(const BlockId& id, Tier tier) {
  switch (tier) {
    case Tier::HBM:  return hbm_  ? hbm_->Erase(id)  : 0;
    case Tier::DRAM: return dram_ ? dram_->Erase(id) : 0;
    case Tier::SSD:  return ssd_  ? ssd_->Erase(id)  : 0;
  }
  return 0;
}

std::optional<GetResult> TieredStore::Get(const BlockId& id) {
  // Lookup order: HBM → DRAM → SSD. The first hit wins; promotion
  // decision happens AFTER we know which tier had the block.
  Tier found = Tier::SSD;
  std::vector<std::byte> bytes;

  if (hbm_) {
    if (auto span = hbm_->Get(id); span.has_value()) {
      found = Tier::HBM;
      bytes.assign(span->begin(), span->end());
    }
  }
  if (bytes.empty() && dram_) {
    if (auto span = dram_->Get(id); span.has_value()) {
      found = Tier::DRAM;
      bytes.assign(span->begin(), span->end());
    }
  }
  if (bytes.empty() && ssd_) {
    if (auto blk = ssd_->GetCopy(id); blk.has_value()) {
      found = Tier::SSD;
      bytes = std::move(blk->data);
    }
  }
  if (bytes.empty() && found != Tier::HBM && found != Tier::DRAM) {
    return std::nullopt;
  }

  // Bump the access counter AND set the SIEVE visited bit under
  // counts_mu_ (its own lock; not held anywhere else above). Both
  // pieces of bookkeeping fire on every Get hit, so one lock
  // acquisition does both. Eviction reads visited via Snapshot,
  // which takes shared locks on the underlying BlockStores; the
  // visited bit lookup happens against this map under shared_lock,
  // serialized against future MarkVisited calls.
  std::uint32_t count = 0;
  {
    std::unique_lock<std::shared_mutex> lock(counts_mu_);
    count = ++access_counts_[id];
    visited_.insert(id);
  }

  bool promoted = false;
  // Promote on the leading edge: when count >= threshold AND we're not
  // already at HBM AND the faster tier has space. Best-effort — if the
  // faster tier's Put returns false (capacity exhausted), we leave the
  // block where it is. W8's eviction will eventually free space.
  if (cfg_.enable_promotion &&
      count >= cfg_.promotion_access_threshold &&
      found != Tier::HBM) {
    Tier target = (found == Tier::SSD) ? Tier::DRAM : Tier::HBM;
    // Skip HBM target if HBM tier doesn't exist.
    if (target == Tier::HBM && !hbm_) {
      target = Tier::DRAM;
      // And if we're already at DRAM, no promotion target.
      if (found == Tier::DRAM) {
        target = Tier::DRAM;  // sentinel: no-op below
      }
    }
    if (target != found) {
      KvBlock promo;
      promo.id = id;
      promo.data = bytes;  // copy — we still return the original bytes
      promo.tier = target;
      // Set inserted_epoch from the existing snapshot if available; we
      // don't have it cheaply here, so leave it 0. Promotion isn't a
      // re-insert from the cluster's perspective.
      promo.inserted_epoch = 0;
      if (PutToTier(std::move(promo), target)) {
        // Promotion succeeded: drop the cold copy. Note this only fires
        // when we actually moved bytes to a faster tier; on failure
        // the original copy stays put.
        EraseFromTier(id, found);
        promoted = true;
      }
    }
  }

  GetResult r;
  r.data = std::move(bytes);
  r.tier_found = found;
  r.promoted = promoted;
  return r;
}

std::optional<Tier> TieredStore::Put(KvBlock block, Tier hint) {
  // Try the hinted tier first; fall through to slower tiers if it's
  // full or doesn't exist. We never fall UP — the hint is a maximum
  // "where the caller thinks this block should land" and the policy
  // is to demote on pressure, not promote on insert.
  static constexpr Tier kFallthrough[3][3] = {
      {Tier::HBM,  Tier::DRAM, Tier::SSD},   // hint=HBM
      {Tier::DRAM, Tier::SSD,  Tier::SSD},   // hint=DRAM (last entry unused)
      {Tier::SSD,  Tier::SSD,  Tier::SSD},   // hint=SSD
  };
  const auto& row = kFallthrough[static_cast<std::size_t>(hint)];
  for (Tier candidate : row) {
    KvBlock copy = block;  // each PutToTier consumes a moved KvBlock
    if (PutToTier(std::move(copy), candidate)) {
      return candidate;
    }
    if (candidate == Tier::SSD) break;  // exhausted the chain
  }
  return std::nullopt;
}

bool TieredStore::Demote(const BlockId& id) {
  // Find current tier by trying each store. We could maintain a
  // tier-of-residence map, but the Get-style fan-out is cheap (three
  // hash lookups under shared locks) and avoids the consistency
  // headaches of a separate index.
  if (hbm_) {
    if (auto span = hbm_->Get(id); span.has_value()) {
      KvBlock blk;
      blk.id = id;
      blk.data.assign(span->begin(), span->end());
      blk.tier = Tier::DRAM;
      // Drop from HBM first to free its capacity, then attempt the
      // DRAM Put. If the DRAM Put fails (DRAM also full), the block is
      // lost — caller (Evictor) should have checked DRAM space first.
      if (!PutToTier(std::move(blk), Tier::DRAM)) {
        // DRAM full. Try SSD before giving up.
        KvBlock blk2;
        blk2.id = id;
        blk2.data.assign(span->begin(), span->end());
        blk2.tier = Tier::SSD;
        if (!PutToTier(std::move(blk2), Tier::SSD)) {
          return false;  // no target had space
        }
      }
      hbm_->Erase(id);
      return true;
    }
  }
  if (dram_) {
    if (auto span = dram_->Get(id); span.has_value()) {
      KvBlock blk;
      blk.id = id;
      blk.data.assign(span->begin(), span->end());
      blk.tier = Tier::SSD;
      if (!PutToTier(std::move(blk), Tier::SSD)) {
        return false;
      }
      dram_->Erase(id);
      return true;
    }
  }
  // Already at SSD (or not present): no slower tier to demote to.
  return false;
}

std::size_t TieredStore::Erase(const BlockId& id) {
  std::size_t freed = 0;
  if (hbm_)  freed += hbm_->Erase(id);
  if (dram_) freed += dram_->Erase(id);
  if (ssd_)  freed += ssd_->Erase(id);
  // Wipe per-block bookkeeping so a re-Insert behaves as a fresh block,
  // per the W0 contract. Both access_counts_ and the SIEVE visited
  // bit live under counts_mu_; one lock acquisition wipes both.
  {
    std::unique_lock<std::shared_mutex> lock(counts_mu_);
    access_counts_.erase(id);
    visited_.erase(id);
  }
  return freed;
}

std::size_t TieredStore::used_bytes(Tier t) const {
  switch (t) {
    case Tier::HBM:  return hbm_  ? hbm_->used_bytes()  : 0;
    case Tier::DRAM: return dram_ ? dram_->used_bytes() : 0;
    case Tier::SSD:  return ssd_  ? ssd_->used_bytes()  : 0;
  }
  return 0;
}

std::size_t TieredStore::capacity_bytes(Tier t) const {
  switch (t) {
    case Tier::HBM:  return hbm_  ? hbm_->capacity_bytes()  : 0;
    case Tier::DRAM: return dram_ ? dram_->capacity_bytes() : 0;
    case Tier::SSD:  return ssd_  ? ssd_->capacity_bytes()  : 0;
  }
  return 0;
}

std::vector<BlockMeta> TieredStore::Snapshot(Tier t) const {
  std::vector<BlockMeta> out;
  switch (t) {
    case Tier::HBM:  out = hbm_  ? hbm_->Snapshot()  : std::vector<BlockMeta>{}; break;
    case Tier::DRAM: out = dram_ ? dram_->Snapshot() : std::vector<BlockMeta>{}; break;
    case Tier::SSD:  out = ssd_  ? ssd_->Snapshot()  : std::vector<BlockMeta>{}; break;
  }
  // Overlay the W8 SIEVE visited bit. The underlying BlockStore
  // doesn't know about visited; we paint it here so Evictor consumers
  // see a unified BlockMeta.
  if (!out.empty()) {
    std::shared_lock<std::shared_mutex> lock(counts_mu_);
    for (auto& m : out) {
      m.visited = (visited_.count(m.id) > 0);
    }
  }
  return out;
}

void TieredStore::MarkVisited(const BlockId& id) {
  std::unique_lock<std::shared_mutex> lock(counts_mu_);
  visited_.insert(id);
}

void TieredStore::ClearVisited(const BlockId& id) {
  std::unique_lock<std::shared_mutex> lock(counts_mu_);
  visited_.erase(id);
}

std::uint32_t TieredStore::access_count_for_testing(const BlockId& id) const {
  std::shared_lock<std::shared_mutex> lock(counts_mu_);
  auto it = access_counts_.find(id);
  return it == access_counts_.end() ? 0u : it->second;
}

bool TieredStore::visited_for_testing(const BlockId& id) const {
  std::shared_lock<std::shared_mutex> lock(counts_mu_);
  return visited_.count(id) > 0;
}

}  // namespace lethe
