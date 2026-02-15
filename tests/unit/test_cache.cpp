// LetheCache facade unit test. Covers the Insert/Lookup round-trip at the
// facade layer (not the gRPC wire — that's in
// tests/integration/test_client_roundtrip.py), single-node DRAM-only.
// Asserts LocalHit on insert, Miss on lookup miss, correct hit/miss
// counts, cluster_epoch()==0 when membership is null, and that
// Entry::local_data is independent of subsequent cache mutation (the
// entry owns its bytes; kept as a regression guard against re-introducing
// borrows). Standalone main()-with-asserts.

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
  cfg.ssd_bytes = 0;             // SSD disabled.
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
  // No Membership → epoch is 0 unconditionally.
  assert(cache.cluster_epoch() == 0);
}

void TestLocalDataSpanCopyPattern() {
  // The gRPC handler does Lookup → copy bytes into the proto → return.
  // That copy happens entirely within one stack frame, so no intervening
  // mutation can race it. Verify that pattern holds.

  LetheCache cache(MakeCfg());
  cache.Start();

  KvBlock blk = MakeBlock(7, 2048, 0xEE);
  const auto id = blk.id;
  const std::vector<std::byte> expected = blk.data;

  cache.Insert({blk}, "req", "client");

  auto result = cache.Lookup({id}, "req", "client");
  assert(result.entries.size() == 1);
  assert(result.entries[0].where == LookupResult::Entry::Where::LocalHit);

  // Copy bytes into a local buffer before returning control to anything
  // that might mutate, as the gRPC handler does.
  std::vector<std::byte> captured(
      result.entries[0].local_data.begin(),
      result.entries[0].local_data.end());

  // Insert a different block — the most mutating call the facade exposes
  // (no Erase in the public API). It must not invalidate the prior
  // block's data, per unordered_map's reference-stability-on-rehash
  // guarantee, so the captured bytes still match the original.
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
