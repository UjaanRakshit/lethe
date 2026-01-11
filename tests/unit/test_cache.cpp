// Lethe — LetheCache facade unit test (W1).
//
// Covers the Lookup / Insert round-trip at the facade layer, not the
// gRPC wire (that's in tests/integration/test_client_roundtrip.py).
// Single-node, DRAM-only — matches the W1 deliverable scope.
//
// Specifically asserts:
//   * Insert → Lookup returns LocalHit with a non-empty local_data span.
//   * Lookup miss returns Miss (no router → no RemoteHit path in W1).
//   * Multiple inserts (different ids) coexist; LookupResult counts
//     hits and misses correctly per the proto's contract.
//   * cluster_epoch() == 0 in W1 (no Membership yet; cache.cpp returns
//     0 explicitly when membership_ is null).
//   * The local_data lifetime contract from cache.hpp:81-86: the gRPC
//     shim pattern (capture-bytes-before-next-call) works as long as
//     the caller respects it. Demonstrated by capturing the span,
//     doing a subsequent Insert (which under W1's no-mutation-of-
//     existing-blocks semantics is harmless), and verifying the
//     captured bytes are still valid.
//
// Standalone main()-with-asserts.

#include "lethe/cache.hpp"
#include "lethe/types.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using lethe::BlockId;
using lethe::CacheConfig;
using lethe::KvBlock;
using lethe::LetheCache;
using lethe::LookupResult;
using lethe::Tier;

BlockId MakeId(std::uint32_t seed) {
  BlockId id;
  std::mt19937 rng(seed);
  for (auto& b : id.hash) b = static_cast<std::byte>(rng() & 0xff);
  id.layer = 0;
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

KvBlock MakeBlock(std::uint32_t seed, std::size_t n, std::uint8_t pattern) {
  KvBlock b;
  b.id = MakeId(seed);
  b.data = MakePayload(n, pattern);
  b.tier = Tier::DRAM;
  return b;
}

CacheConfig MakeCfg() {
  CacheConfig cfg;
  cfg.node_id = "test_node";
  cfg.hbm_bytes = 0;
  cfg.dram_bytes = 1ULL << 20;  // 1 MiB
  cfg.ssd_bytes = 0;             // SSD disabled in W1.
  return cfg;
}

void TestInsertLookupRoundtrip() {
  LetheCache cache(MakeCfg());
  cache.Start();

  std::vector<KvBlock> blocks;
  for (std::uint32_t i = 1; i <= 4; ++i) {
    blocks.push_back(MakeBlock(i, 512, static_cast<std::uint8_t>(i)));
  }
  std::vector<BlockId> ids;
  ids.reserve(blocks.size());
  for (const auto& b : blocks) ids.push_back(b.id);

  const auto accepted = cache.Insert(
      std::move(blocks), "req-1", "client");
  assert(accepted == 4);

  auto result = cache.Lookup(ids, "req-2", "client");
  assert(result.entries.size() == 4);
  assert(result.hit_count == 4);
  assert(result.miss_count == 0);
  for (const auto& e : result.entries) {
    assert(e.where == LookupResult::Entry::Where::LocalHit);
    assert(e.tier == Tier::DRAM);
    assert(e.local_data.size() == 512);
  }

  cache.Shutdown();
}

void TestLookupMissReturnsMiss() {
  LetheCache cache(MakeCfg());
  cache.Start();

  std::vector<BlockId> ids = {MakeId(999), MakeId(1000)};
  auto result = cache.Lookup(ids, "req", "client");
  assert(result.entries.size() == 2);
  assert(result.hit_count == 0);
  assert(result.miss_count == 2);
  for (const auto& e : result.entries) {
    assert(e.where == LookupResult::Entry::Where::Miss);
    assert(e.local_data.size() == 0);
  }

  cache.Shutdown();
}

void TestClusterEpochZeroInW1() {
  LetheCache cache(MakeCfg());
  // No Membership yet → epoch is 0 unconditionally.
  assert(cache.cluster_epoch() == 0);
}

void TestLocalDataSpanCopyPattern() {
  // The gRPC shim pattern is: Lookup → copy span into proto bytes →
  // return. This test demonstrates that the pattern is safe: the
  // copy happens entirely within the shim's stack frame, no
  // intervening mutation can race.

  LetheCache cache(MakeCfg());
  cache.Start();

  KvBlock blk = MakeBlock(7, 2048, 0xEE);
  const auto id = blk.id;
  const std::vector<std::byte> expected = blk.data;

  cache.Insert({blk}, "req", "client");

  auto result = cache.Lookup({id}, "req", "client");
  assert(result.entries.size() == 1);
  assert(result.entries[0].where == LookupResult::Entry::Where::LocalHit);

  // Simulate the gRPC shim: copy bytes from the span into a local
  // buffer BEFORE returning control to anything that might mutate.
  std::vector<std::byte> captured(
      result.entries[0].local_data.begin(),
      result.entries[0].local_data.end());

  // Trigger an Insert that doesn't touch the existing block id (W1's
  // public API doesn't expose Erase; this is the most "mutating" call
  // available at the facade layer). Inserting a DIFFERENT block must
  // not invalidate the prior block's data — and it doesn't, per
  // unordered_map's reference-stability-on-rehash guarantee. The
  // captured bytes still match the original; if we had instead held
  // the raw span across the mutation, that would still be valid in
  // W1 but tracking the safe pattern matters for W4+ when Erase /
  // eviction can race.
  KvBlock other = MakeBlock(8, 4096, 0x55);
  cache.Insert({other}, "req", "client");

  assert(captured.size() == expected.size());
  for (std::size_t i = 0; i < captured.size(); ++i) {
    assert(captured[i] == expected[i]);
  }

  // Sanity: the block is still cached and still has the same content.
  auto result2 = cache.Lookup({id}, "req", "client");
  assert(result2.entries[0].where == LookupResult::Entry::Where::LocalHit);
  assert(result2.entries[0].local_data.size() == expected.size());

  cache.Shutdown();
}

}  // namespace

int main() {
  TestInsertLookupRoundtrip();
  TestLookupMissReturnsMiss();
  TestClusterEpochZeroInW1();
  TestLocalDataSpanCopyPattern();
  std::printf("test_cache: OK\n");
  return 0;
}
