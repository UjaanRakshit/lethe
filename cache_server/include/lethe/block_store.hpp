#pragma once
// Single-tier block store. Owns the actual KV bytes for blocks resident in
// one storage tier and provides thread-safe get/put/erase. TieredStore
// composes three of these.

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

class BlockStore {
 public:
  BlockStore(Tier tier, std::size_t capacity_bytes);

  BlockStore(const BlockStore&) = delete;
  BlockStore& operator=(const BlockStore&) = delete;

  // Inserts a block. Returns false if the store is full (caller should evict
  // before retrying).
  bool Put(KvBlock block);

  // Borrowed view of the block data; caller must not retain past the next
  // mutating call. Returns nullopt on miss.
  std::optional<std::span<const std::byte>> Get(const BlockId& id);

  // Removes the named block. Returns bytes freed (0 if absent).
  std::size_t Erase(const BlockId& id);

  // Total capacity and current usage in bytes.
  std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
  std::size_t used_bytes() const noexcept;

  // Metadata snapshot - used by the evictor to pick victims.
  std::vector<BlockMeta> Snapshot() const;

  Tier tier() const noexcept { return tier_; }

 private:
  Tier tier_;
  std::size_t capacity_bytes_;

  mutable std::shared_mutex mu_;
  std::unordered_map<BlockId, KvBlock, BlockIdHash> blocks_;
  std::atomic<std::size_t> used_bytes_{0};
};

}  // namespace lethe
