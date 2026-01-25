// Lethe — single-tier block store (W1).
//
// Thread-safety: `shared_mutex` for reader-favored hot paths. Writers
// (Put, Erase) take a unique_lock; readers (Get, Snapshot) take a
// shared_lock. `used_bytes_` is std::atomic so cheap unlocked reads
// from metrics work; all *updates* still happen inside the write lock
// so Put+Erase races can't race-corrupt the accounting.
//
// Lifetime contract: Get returns a span into the stored KvBlock.data
// vector. unordered_map does NOT invalidate references to elements
// on rehash (per the standard), so concurrent Puts that grow the map
// don't invalidate existing spans. Erase DOES invalidate the element;
// callers must either copy the span before any mutation on that
// BlockId or hold some other guarantee that no Erase will land. The
// only direct caller at and above this layer is TieredStore::Get,
// which copies the span into an owned vector before returning to
// upper layers (W7 owned-bytes contract on LookupResult::Entry::
// local_data) — so the lend/borrow window stays inside TieredStore.

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
    // the same id MUST have the same bytes; nothing to do. We don't
    // touch used_bytes_ either — the previously-counted bytes are
    // still there.
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
  // KvBlock.data is std::vector<std::byte>; .data()/.size() are stable
  // across unordered_map rehashes (the standard guarantees no
  // reference invalidation on rehash). The span is valid until an
  // Erase or replacing Put on this same id.
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
    // W1: no per-access bookkeeping yet (the SIEVE visited bit lands
    // when Evictor::MarkVisited fires in W8). last_access_epoch is
    // seeded to insert_epoch so the field is well-defined; the SIEVE
    // hand will manage `visited` from W8 onward.
    m.insert_epoch = kv.inserted_epoch;
    m.last_access_epoch = kv.inserted_epoch;
    m.visited = false;
    out.push_back(m);
  }
  return out;
}

}  // namespace lethe
