// Single-tier block store.
//
// Thread-safety: `shared_mutex` for reader-favored hot paths. Writers take a
// unique_lock; readers take a shared_lock. `used_bytes_` is atomic for cheap
// unlocked metrics reads, but updates happen inside the write lock so
// Put/Erase races can't corrupt the accounting.
//
// Lifetime: Get returns a span into the stored KvBlock.data. unordered_map
// doesn't invalidate references on rehash, so concurrent Puts that grow the
// map keep existing spans valid; Erase does invalidate. The only caller above
// this layer is TieredStore::Get, which copies the span into an owned vector
// before returning - so the lend/borrow window stays inside TieredStore.

#include "lethe/block_store.hpp"

#include <cstring>
#include <utility>

namespace lethe {

BlockStore::BlockStore(Tier tier, std::size_t capacity_bytes)
    : tier_(tier), capacity_bytes_(capacity_bytes) {}

bool BlockStore::Put(KvBlock block) {
  // Force tier consistency: callers shouldn't smuggle a different tier
  // through Put; the store knows its own tier.
  block.tier = tier_;
  const std::size_t new_size = block.data.size();

  std::unique_lock<std::shared_mutex> lock(mu_);

  if (auto it = blocks_.find(block.id); it != blocks_.end()) {
    // Idempotent on key. BlockId is a content hash, so two blocks with
    // the same id have the same bytes; nothing to do, and used_bytes_
    // already counts them.
    return true;
  }

  // Pre-flight capacity check before mutating state.
  if (used_bytes_.load(std::memory_order_relaxed) + new_size > capacity_bytes_) {
    return false;
  }

  // Move the block in so we don't copy the payload.
  const BlockId id_copy = block.id;
  blocks_.emplace(id_copy, std::move(block));
  used_bytes_.fetch_add(new_size, std::memory_order_relaxed);
  return true;
}

std::optional<std::span<const std::byte>> BlockStore::Get(const BlockId& id) {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = blocks_.find(id);
  if (it == blocks_.end()) {
    return std::nullopt;
  }
  // .data()/.size() are stable across unordered_map rehashes; the span is
  // valid until an Erase or replacing Put on this same id.
  const auto& kv = it->second;
  return std::span<const std::byte>(kv.data.data(), kv.data.size());
}

std::size_t BlockStore::Erase(const BlockId& id) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  auto it = blocks_.find(id);
  if (it == blocks_.end()) {
    return 0;
  }
  const std::size_t freed = it->second.data.size();
  blocks_.erase(it);
  used_bytes_.fetch_sub(freed, std::memory_order_relaxed);
  return freed;
}

std::size_t BlockStore::used_bytes() const noexcept {
  return used_bytes_.load(std::memory_order_relaxed);
}

std::vector<BlockMeta> BlockStore::Snapshot() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::vector<BlockMeta> out;
  out.reserve(blocks_.size());
  for (const auto& [id, kv] : blocks_) {
    BlockMeta m;
    m.id = id;
    m.tier = tier_;
    m.size_bytes = kv.data.size();
    // last_access_epoch is seeded to insert_epoch so the field is
    // well-defined; the SIEVE hand manages `visited`.
    m.insert_epoch = kv.inserted_epoch;
    m.last_access_epoch = kv.inserted_epoch;
    m.visited = false;
    out.push_back(m);
  }
  return out;
}

}  // namespace lethe
