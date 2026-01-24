// Lethe — SSD block store (W7).
//
// One mmap'd file per node. Slot-based allocator: every block lives in
// exactly one slot, slots are fixed-size (default 64 KiB), header at the
// start of the slot, payload follows.
//
// Persistence model: NO fsync. Per the W0 design decision documented in
// DESIGN.md "What's intentionally fragile" — host crash loses recently-
// written SSD blocks; recovery falls back to recompute via the model
// weights. The acceptance test is process-restart durability (kill -9
// the cache server, restart, read the block back) — the page cache
// keeps the bytes around for that. Host crash is a separate, harder
// promise we explicitly don't make.
//
// Index rebuild on startup: scan every slot's header; if magic byte is
// the live sentinel (0xA5), pull the BlockId into the in-memory index
// and account the payload bytes. This is O(total_slots) at startup but
// happens once per process lifetime. The torn-write window (header
// partially written but magic not set) is invisible to us — those
// slots come back as "free" on restart, which is correct: a block we
// can't read intact is one we never had.

#include "lethe/ssd_block_store.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace lethe {

namespace {

constexpr std::uint8_t kSlotMagicLive = 0xA5;
constexpr std::uint8_t kSlotMagicFree = 0x00;

}  // namespace

SsdBlockStore::SsdBlockStore(std::filesystem::path file,
                             std::size_t capacity_bytes,
                             std::size_t slot_bytes)
    : file_(std::move(file)),
      capacity_bytes_(capacity_bytes),
      slot_bytes_(slot_bytes),
      slot_payload_bytes_(slot_bytes - sizeof(SsdSlotHeader)),
      total_slots_(capacity_bytes / slot_bytes),
      usable_bytes_(total_slots_ * slot_payload_bytes_) {
  if (slot_bytes <= sizeof(SsdSlotHeader)) {
    throw std::invalid_argument("ssd slot_bytes must exceed header size");
  }
  if (total_slots_ == 0) {
    throw std::invalid_argument("ssd capacity_bytes < slot_bytes; no slots");
  }

  // Ensure parent directory exists. The test fixture and run_3node.sh
  // already do this; defensive call here so single-node smoke runs work
  // without external setup.
  if (file_.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file_.parent_path(), ec);
    // ec ignored: if creation fails for a real reason, open() below
    // surfaces it with a clearer errno.
  }

  // Open + size the file. O_CREAT preserves an existing file's contents
  // (so the persistence acceptance test works); ftruncate to the
  // configured capacity grows or trims as needed. Trimming an oversized
  // file loses any blocks past the new boundary — acceptable; this is
  // a process startup decision, not a runtime resize.
  fd_ = ::open(file_.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd_ < 0) {
    throw std::runtime_error(std::string("ssd open failed: ") +
                             std::strerror(errno) + " (" + file_.string() + ")");
  }
  // Check current size; only ftruncate if different so we don't blow
  // away a pre-existing populated file on a clean restart.
  struct stat st{};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error(std::string("ssd fstat failed: ") +
                             std::strerror(errno));
  }
  const off_t want = static_cast<off_t>(total_slots_ * slot_bytes_);
  if (st.st_size != want) {
    if (::ftruncate(fd_, want) != 0) {
      ::close(fd_);
      throw std::runtime_error(std::string("ssd ftruncate failed: ") +
                               std::strerror(errno));
    }
  }

  mmap_base_ = ::mmap(nullptr, total_slots_ * slot_bytes_,
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mmap_base_ == MAP_FAILED) {
    mmap_base_ = nullptr;
    ::close(fd_);
    throw std::runtime_error(std::string("ssd mmap failed: ") +
                             std::strerror(errno));
  }

  RebuildIndexFromMmap();
}

SsdBlockStore::~SsdBlockStore() {
  if (mmap_base_ != nullptr) {
    // No msync — host-crash durability is explicitly not a promise.
    ::munmap(mmap_base_, total_slots_ * slot_bytes_);
    mmap_base_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

SsdSlotHeader* SsdBlockStore::slot_header(std::size_t n) {
  return reinterpret_cast<SsdSlotHeader*>(
      static_cast<char*>(mmap_base_) + slot_offset(n));
}

std::byte* SsdBlockStore::slot_payload(std::size_t n) {
  return reinterpret_cast<std::byte*>(
      static_cast<char*>(mmap_base_) + slot_offset(n) + sizeof(SsdSlotHeader));
}

void SsdBlockStore::RebuildIndexFromMmap() {
  // Single-threaded at construction; no lock needed.
  std::size_t live = 0;
  std::size_t live_bytes = 0;
  for (std::size_t n = 0; n < total_slots_; ++n) {
    const SsdSlotHeader* h = slot_header(n);
    if (h->magic != kSlotMagicLive) {
      free_slots_.push_back(n);
      continue;
    }
    if (h->payload_size > slot_payload_bytes_) {
      // Corrupt header (size larger than slot). Treat as free; this
      // slot is recoverable by a future Put. Log so we're not silent
      // about it.
      std::cerr << "[lethe] ssd: corrupt header at slot " << n
                << " (payload_size=" << h->payload_size << "); reclaiming\n";
      free_slots_.push_back(n);
      continue;
    }
    BlockId id;
    id.hash = h->hash;
    id.layer = h->layer;
    id.head_group = h->head_group;
    id.model_id = h->model_id;
    index_.emplace(id, n);
    live_bytes += h->payload_size;
    ++live;
  }
  used_bytes_.store(live_bytes, std::memory_order_relaxed);
  // bump_next_ stays at 0 — we rebuilt free_slots_ exhaustively, so
  // every unused slot is on the free list. Bump path is unused on a
  // restored file; it only fires on a fresh-allocator path where we
  // skipped the scan (we don't, so this is fine).
  bump_next_ = total_slots_;
}

std::optional<std::size_t> SsdBlockStore::AllocSlot() {
  if (!free_slots_.empty()) {
    const std::size_t n = free_slots_.back();
    free_slots_.pop_back();
    return n;
  }
  if (bump_next_ < total_slots_) {
    return bump_next_++;
  }
  return std::nullopt;
}

bool SsdBlockStore::Put(KvBlock block) {
  // Pre-flight reject: payload too big for a slot.
  if (block.data.size() > slot_payload_bytes_) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mu_);

  // Idempotent on existing id (same semantics as DRAM BlockStore).
  if (auto it = index_.find(block.id); it != index_.end()) {
    return true;
  }

  auto slot_opt = AllocSlot();
  if (!slot_opt.has_value()) {
    return false;
  }
  const std::size_t n = *slot_opt;
  SsdSlotHeader* h = slot_header(n);
  std::byte* payload = slot_payload(n);

  // Write payload FIRST, then header fields, then the magic last. If
  // we crash mid-write, the magic stays 0x00 (or whatever stale value)
  // and the slot reads as free on restart. The persistence acceptance
  // test only relies on the page cache outliving process exit, not on
  // surviving a real crash — but writing in this order costs nothing
  // and makes the future fsync wiring straightforward.
  std::memcpy(payload, block.data.data(), block.data.size());
  h->tier = static_cast<std::uint8_t>(Tier::SSD);
  h->flags = 0;
  h->payload_size = static_cast<std::uint32_t>(block.data.size());
  h->hash = block.id.hash;
  h->layer = block.id.layer;
  h->head_group = block.id.head_group;
  h->model_id = block.id.model_id;
  h->inserted_epoch = block.inserted_epoch;
  for (auto& b : h->reserved) b = std::byte{0};
  // Magic last.
  h->magic = kSlotMagicLive;

  index_.emplace(block.id, n);
  used_bytes_.fetch_add(block.data.size(), std::memory_order_relaxed);
  return true;
}

std::optional<KvBlock> SsdBlockStore::GetCopy(const BlockId& id) {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = index_.find(id);
  if (it == index_.end()) {
    return std::nullopt;
  }
  const std::size_t n = it->second;
  SsdSlotHeader* h = slot_header(n);
  // Header magic check is defensive — the index claims this slot is
  // live; if the on-disk magic is anything else, the file got
  // externally clobbered, treat as miss.
  if (h->magic != kSlotMagicLive) {
    return std::nullopt;
  }
  KvBlock out;
  out.id.hash = h->hash;
  out.id.layer = h->layer;
  out.id.head_group = h->head_group;
  out.id.model_id = h->model_id;
  out.tier = Tier::SSD;
  out.inserted_epoch = h->inserted_epoch;
  out.data.resize(h->payload_size);
  std::memcpy(out.data.data(), slot_payload(n), h->payload_size);
  return out;
}

std::size_t SsdBlockStore::Erase(const BlockId& id) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  auto it = index_.find(id);
  if (it == index_.end()) {
    return 0;
  }
  const std::size_t n = it->second;
  SsdSlotHeader* h = slot_header(n);
  const std::size_t freed = h->payload_size;
  // Clear magic first so a concurrent reader (we're under unique lock,
  // so there shouldn't be one, but defensive) sees the slot as free.
  h->magic = kSlotMagicFree;
  h->payload_size = 0;
  index_.erase(it);
  free_slots_.push_back(n);
  used_bytes_.fetch_sub(freed, std::memory_order_relaxed);
  return freed;
}

std::size_t SsdBlockStore::used_bytes() const noexcept {
  return used_bytes_.load(std::memory_order_relaxed);
}

std::vector<BlockMeta> SsdBlockStore::Snapshot() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::vector<BlockMeta> out;
  out.reserve(index_.size());
  for (const auto& [id, slot_idx] : index_) {
    // Read the header under the shared lock — Put/Erase hold unique, so
    // no concurrent mutation. Cast away const for header access; the
    // method is logically const because we don't modify the bytes.
    const SsdSlotHeader* h = const_cast<SsdBlockStore*>(this)->slot_header(slot_idx);
    BlockMeta m;
    m.id = id;
    m.tier = Tier::SSD;
    m.size_bytes = h->payload_size;
    m.insert_epoch = h->inserted_epoch;
    m.last_access_epoch = h->inserted_epoch;
    m.visited = false;
    out.push_back(m);
  }
  return out;
}

std::size_t SsdBlockStore::live_slot_count() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return index_.size();
}

}  // namespace lethe
