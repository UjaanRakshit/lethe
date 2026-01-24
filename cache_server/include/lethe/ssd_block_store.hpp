#pragma once
// Lethe — SSD-backed block store (W7).
//
// File-backed slot allocator on top of a single mmap'd region per node.
// One 64 KiB slot per block (configurable); the slot header carries the
// content hash + payload size + tier so we can rebuild the in-memory
// index by scanning the file at startup. No fsync — host crash loses
// recently-written SSD blocks, which is acceptable per the W0 design
// (the source of truth is the model weights, which can recompute).
//
// Threading: shared_mutex with the same reader-favored shape as
// BlockStore. The mmap region itself is shared by all threads; each
// slot's payload bytes are only mutated under the write lock during
// Put/Erase.
//
// What this is NOT:
//   - Not a real on-disk hash table. Index lookups happen against an
//     in-memory unordered_map<BlockId, slot_index>. The index is
//     reconstructed by scanning live slot headers at startup; the
//     file format is "raw slot array" not "B-tree on disk."
//   - Not fragmentation-resistant. The free list is a flat vector of
//     freed slot indexes; allocator is bump-then-freelist. Workloads
//     that churn many small blocks could fragment; if W11 chaos
//     surfaces it we revisit. For W7 acceptance this is sufficient.
//   - Not durable through host crash. See above.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

// On-disk slot layout (POD; little-endian; field order matches the file
// byte order). 64 bytes total leaves room for a few future fields without
// growing the slot beyond a single cache line.
//
// Field order is chosen for natural alignment: the 8-byte
// `inserted_epoch` follows `hash` so it lands at offset 40 (multiple of
// 8) without compiler-inserted padding. The lone 32-bit fields tail
// the struct so the post-pad is just the 4-byte `reserved` slot.
struct SsdSlotHeader {
  // Magic byte to distinguish "live block" (0xA5) from "free" (0x00) and
  // from "torn write" (anything else). Set last when writing so partial
  // writes show up as torn rather than live-with-wrong-bytes.
  std::uint8_t magic;            // offset 0
  std::uint8_t tier;             // offset 1
  std::uint16_t flags;           // offset 2-3
  std::uint32_t payload_size;    // offset 4-7
  std::array<std::byte, 32> hash; // offset 8-39
  std::uint64_t inserted_epoch;  // offset 40-47 (naturally 8-aligned)
  std::uint32_t layer;           // offset 48-51
  std::uint32_t head_group;      // offset 52-55
  std::uint32_t model_id;        // offset 56-59
  std::array<std::byte, 4> reserved;  // offset 60-63
};
static_assert(sizeof(SsdSlotHeader) == 64,
              "SsdSlotHeader must be exactly 64 bytes for slot alignment");

class SsdBlockStore {
 public:
  // capacity_bytes is the total file size; slot_bytes is the per-slot
  // (header + payload) size. Both must be passed at construction; we
  // don't grow the file at runtime.
  SsdBlockStore(std::filesystem::path file,
                std::size_t capacity_bytes,
                std::size_t slot_bytes = 64ULL * 1024);
  ~SsdBlockStore();

  SsdBlockStore(const SsdBlockStore&) = delete;
  SsdBlockStore& operator=(const SsdBlockStore&) = delete;

  // Inserts a block. Returns false if the store is full OR if the block
  // payload exceeds slot capacity (slot_bytes - sizeof(header)).
  bool Put(KvBlock block);

  // Returns an OWNED copy of the block. SSD blocks can't lend spans
  // safely across mutations because the slot may be reused; callers get
  // their own vector and the SSD layer keeps no reference.
  std::optional<KvBlock> GetCopy(const BlockId& id);

  // Removes the block. Returns bytes freed (payload only, not slot
  // overhead).
  std::size_t Erase(const BlockId& id);

  // Used bytes counts ONLY payload bytes of live blocks (consistent with
  // BlockStore::used_bytes). Capacity is the file size minus the
  // overhead of slot headers — we expose what the upper tier can
  // actually use.
  std::size_t capacity_bytes() const noexcept { return usable_bytes_; }
  std::size_t used_bytes() const noexcept;

  std::vector<BlockMeta> Snapshot() const;

  // Test-only seam: how many slots have a live block. Used by
  // test_tiered_store.cpp to verify persistence across restart.
  std::size_t live_slot_count() const;

 private:
  // Returns the offset (bytes from file start) of slot N.
  std::size_t slot_offset(std::size_t n) const { return n * slot_bytes_; }
  SsdSlotHeader* slot_header(std::size_t n);
  std::byte* slot_payload(std::size_t n);

  // Tries to claim a free slot for a new block. Caller holds the
  // unique write lock. Returns nullopt if none available.
  std::optional<std::size_t> AllocSlot();

  // Walks the mmap region scanning for live slot headers and rebuilds
  // the in-memory index + free list. Called once at construction.
  void RebuildIndexFromMmap();

  std::filesystem::path file_;
  std::size_t capacity_bytes_;
  std::size_t slot_bytes_;
  std::size_t slot_payload_bytes_;
  std::size_t total_slots_;
  std::size_t usable_bytes_;

  int fd_ = -1;
  void* mmap_base_ = nullptr;

  mutable std::shared_mutex mu_;
  std::unordered_map<BlockId, std::size_t, BlockIdHash> index_;  // BlockId → slot
  std::vector<std::size_t> free_slots_;                          // freed slot indexes
  std::size_t bump_next_ = 0;                                    // next unused slot
  std::atomic<std::size_t> used_bytes_{0};
};

}  // namespace lethe
