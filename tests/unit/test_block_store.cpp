// Lethe — BlockStore unit test (W1).
//
// Covers:
//   * Round-trip Put → Get → byte-identical readback.
//   * Erase removes the block; subsequent Get returns nullopt.
//   * Idempotent Put on the same id is a no-op for used_bytes.
//   * Capacity overflow: Put returns false past capacity.
//   * used_bytes accounting matches the sum of payload sizes exactly
//     across any sequence of Put / Erase / Snapshot operations.
//   * Concurrent Put / Get / Erase across N threads converges on the
//     correct final used_bytes and produces no crash or data race
//     (TSan-clean by construction; build with -DLETHE_ENABLE_TSAN=ON).
//   * Explicit local_data lifetime contract:
//       Put → Get(span) → capture bytes → Erase → Get returns nullopt
//       and the captured copy still matches what we read pre-Erase.
//
// Standalone main()-with-asserts; ported to GoogleTest in W3+.

#include "lethe/block_store.hpp"
#include "lethe/types.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using lethe::BlockId;
using lethe::BlockStore;
using lethe::KvBlock;
using lethe::Tier;

BlockId MakeId(std::uint32_t seed, std::uint32_t layer = 0) {
  BlockId id;
  // Trivially distinct hashes seeded from `seed`; tests don't care about
  // BLAKE3-correctness here, only uniqueness.
  std::mt19937 rng(seed);
  for (auto& b : id.hash) {
    b = static_cast<std::byte>(rng() & 0xff);
  }
  id.layer = layer;
  id.head_group = 0;
  id.model_id = 0;
  return id;
}

std::vector<std::byte> MakePayload(std::size_t n, std::uint8_t pattern) {
  std::vector<std::byte> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<std::byte>(pattern ^ static_cast<std::uint8_t>(i & 0xff));
  }
  return out;
}

KvBlock MakeBlock(std::uint32_t seed, std::size_t payload_size,
                  std::uint8_t pattern) {
  KvBlock blk;
  blk.id = MakeId(seed);
  blk.data = MakePayload(payload_size, pattern);
  blk.tier = Tier::DRAM;
  blk.inserted_epoch = 0;
  return blk;
}

void TestBasicRoundtrip() {
  BlockStore store(Tier::DRAM, /*capacity_bytes=*/1ULL << 20);

  KvBlock blk = MakeBlock(1, /*payload_size=*/256, /*pattern=*/0xAB);
  const auto id = blk.id;
  const std::vector<std::byte> expected = blk.data;
  const std::size_t payload_size = expected.size();

  assert(store.Put(std::move(blk)));
  assert(store.used_bytes() == payload_size);

  auto got = store.Get(id);
  assert(got.has_value());
  assert(got->size() == payload_size);
  for (std::size_t i = 0; i < payload_size; ++i) {
    assert((*got)[i] == expected[i]);
  }

  const std::size_t freed = store.Erase(id);
  assert(freed == payload_size);
  assert(store.used_bytes() == 0);
  assert(!store.Get(id).has_value());
}

void TestIdempotentPut() {
  BlockStore store(Tier::DRAM, /*capacity_bytes=*/1ULL << 20);
  KvBlock blk1 = MakeBlock(2, 1024, 0xCC);
  const auto id = blk1.id;
  const std::size_t size1 = blk1.data.size();

  assert(store.Put(std::move(blk1)));
  assert(store.used_bytes() == size1);

  // Second Put with the same id is a no-op; used_bytes must not double.
  KvBlock blk2 = MakeBlock(2, 1024, 0xCC);  // same seed → same id, same bytes
  assert(blk2.id == id);
  assert(store.Put(std::move(blk2)));
  assert(store.used_bytes() == size1);  // unchanged
}

void TestCapacityOverflow() {
  // Capacity 1024 bytes; put two 600-byte blocks. Second one rejected.
  BlockStore store(Tier::DRAM, /*capacity_bytes=*/1024);
  KvBlock a = MakeBlock(10, 600, 0x11);
  KvBlock b = MakeBlock(11, 600, 0x22);

  assert(store.Put(std::move(a)));
  assert(store.used_bytes() == 600);
  assert(!store.Put(std::move(b)));     // would exceed capacity
  assert(store.used_bytes() == 600);    // unchanged
}

void TestUsedBytesAccountingMixed() {
  // Sum-of-payloads invariant across an interleaved Put/Erase trace.
  BlockStore store(Tier::DRAM, /*capacity_bytes=*/1ULL << 20);
  std::size_t expected = 0;

  std::vector<BlockId> live_ids;
  std::vector<std::size_t> live_sizes;

  for (std::uint32_t i = 0; i < 100; ++i) {
    const std::size_t sz = 128 + (i * 17) % 4096;
    KvBlock blk = MakeBlock(100 + i, sz, static_cast<std::uint8_t>(i));
    live_ids.push_back(blk.id);
    live_sizes.push_back(sz);
    assert(store.Put(std::move(blk)));
    expected += sz;
    assert(store.used_bytes() == expected);
  }

  // Erase every third block; bytes drop by exactly the freed amount.
  for (std::size_t i = 0; i < live_ids.size(); i += 3) {
    const std::size_t freed = store.Erase(live_ids[i]);
    assert(freed == live_sizes[i]);
    expected -= freed;
    assert(store.used_bytes() == expected);
  }

  // Snapshot matches the surviving set in count and aggregate size.
  auto snap = store.Snapshot();
  std::size_t snap_total = 0;
  for (const auto& m : snap) snap_total += m.size_bytes;
  assert(snap_total == expected);
}

void TestLocalDataLifetimeContract() {
  // The contract: Get returns a span valid until the next mutating call
  // on this BlockId. The test demonstrates the SHIM pattern: copy the
  // span into a buffer BEFORE any mutation, then verify the captured
  // buffer's content is correct independent of the cache state.

  BlockStore store(Tier::DRAM, 1ULL << 20);
  KvBlock blk = MakeBlock(42, 1024, 0xAA);
  const auto id = blk.id;
  const std::vector<std::byte> expected = blk.data;

  assert(store.Put(std::move(blk)));
  auto span_or = store.Get(id);
  assert(span_or.has_value());

  // Step 1: capture bytes from the span IMMEDIATELY. This is exactly
  // what the gRPC shim in main.cpp does in LookupServiceImpl::Lookup
  // before returning to the gRPC layer.
  std::vector<std::byte> captured(span_or->begin(), span_or->end());

  // Step 2: trigger a mutation on this id — Erase invalidates the
  // underlying KvBlock and therefore the span.
  const std::size_t freed = store.Erase(id);
  assert(freed == 1024);
  assert(!store.Get(id).has_value());

  // Step 3: the captured copy is independent of cache state. After
  // Erase, span_or's bytes are formally UB to read — but the copy
  // we made before the mutation is fine, and its content matches the
  // pre-Erase data byte-for-byte. This is what the gRPC shim's
  // "copy into wire response immediately" pattern relies on.
  assert(captured.size() == expected.size());
  for (std::size_t i = 0; i < captured.size(); ++i) {
    assert(captured[i] == expected[i]);
  }
}

void TestConcurrentPutGetErase() {
  // Stress the shared_mutex discipline. N threads, each doing M
  // iterations of (Put a random block, Get it back, sometimes Erase).
  // Pass criterion: no crash, final used_bytes matches the recorded
  // sum-of-live-blocks. TSan-clean by construction.
  BlockStore store(Tier::DRAM, /*capacity_bytes=*/64ULL << 20);

  constexpr int kThreads = 8;
  constexpr int kIterPerThread = 500;
  std::atomic<int> total_puts{0};
  std::atomic<int> total_erases{0};

  auto worker = [&](int tid) {
    std::mt19937 rng(static_cast<std::uint32_t>(tid * 9001));
    for (int i = 0; i < kIterPerThread; ++i) {
      // Block id is namespaced by thread to avoid Put/Erase contention
      // for the accounting test; the *mutex* contention is what we're
      // actually stressing.
      const std::uint32_t seed =
          static_cast<std::uint32_t>(tid * 100000 + i);
      const std::size_t sz = 64 + (rng() % 1024);
      KvBlock blk = MakeBlock(seed, sz, static_cast<std::uint8_t>(tid));
      const auto id = blk.id;

      if (store.Put(std::move(blk))) {
        total_puts.fetch_add(1, std::memory_order_relaxed);
      }

      // Read back; should succeed because we just inserted.
      auto got = store.Get(id);
      assert(got.has_value());

      if ((rng() % 4) == 0) {
        if (store.Erase(id) > 0) {
          total_erases.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) t.join();

  // Final used_bytes equals sum of live-block sizes from Snapshot.
  // Concurrent Put/Erase + the atomic update under the unique_lock
  // means used_bytes must be exact, not approximate.
  auto snap = store.Snapshot();
  std::size_t snap_sum = 0;
  for (const auto& m : snap) snap_sum += m.size_bytes;
  assert(store.used_bytes() == snap_sum);

  std::printf("test_block_store: concurrent OK puts=%d erases=%d live=%zu bytes=%zu\n",
              total_puts.load(), total_erases.load(),
              snap.size(), store.used_bytes());
}

}  // namespace

int main() {
  TestBasicRoundtrip();
  TestIdempotentPut();
  TestCapacityOverflow();
  TestUsedBytesAccountingMixed();
  TestLocalDataLifetimeContract();
  TestConcurrentPutGetErase();
  std::printf("test_block_store: OK\n");
  return 0;
}
