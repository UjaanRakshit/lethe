// TieredStore + SsdBlockStore unit tests. Standalone main with assert()s.
//
// Coverage map:
//   - Two-tier (hbm=0, dram>0, ssd=0): Put/Get/Erase round-trip.
//   - Three-tier with SSD path: tier-hint fallthrough when DRAM full.
//   - Promotion: SSD → DRAM after `promotion_access_threshold` hits;
//     blocks move and the cold copy is dropped.
//   - Demotion: explicit Demote moves DRAM → SSD; SSD demote returns
//     false (no slower tier).
//   - Erase wipes access_counts_: re-Insert starts the access count over.
//   - SSD persistence across process restart: write a block, destroy the
//     TieredStore (closing the mmap), reconstruct with the same path, and
//     verify the block is still readable. close+mmap+open is the boundary
//     the SsdBlockStore claims to survive, minus the actual fork.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "lethe/ssd_block_store.hpp"
#include "lethe/tiered_store.hpp"
#include "lethe/types.hpp"

using namespace lethe;

namespace {

// Per-test temp directory under /tmp; cleaned at process exit (we leak
// it on failure, on purpose, so the human can poke at the file).
std::filesystem::path MakeTempDir(const std::string& tag) {
  auto base = std::filesystem::temp_directory_path() /
              ("lethe_test_" + tag + "_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  std::filesystem::create_directories(base);
  return base;
}

BlockId MakeBlockId(std::uint64_t seed) {
  BlockId id;
  for (std::size_t i = 0; i < id.hash.size(); ++i) {
    id.hash[i] = static_cast<std::byte>((seed >> (i % 8 * 8)) & 0xFF);
  }
  id.layer = 0;
  id.head_group = 0;
  id.model_id = 0;
  return id;
}

KvBlock MakeBlock(std::uint64_t seed, std::size_t size) {
  KvBlock b;
  b.id = MakeBlockId(seed);
  b.data.resize(size);
  std::mt19937_64 rng(seed);
  for (std::size_t i = 0; i < size; ++i) {
    b.data[i] = static_cast<std::byte>(rng() & 0xFF);
  }
  b.tier = Tier::DRAM;
  b.inserted_epoch = 7;  // arbitrary, propagates through the SSD header
  return b;
}

void TestTwoTierBasic() {
  TieredStoreConfig cfg;
  cfg.hbm_bytes = 0;
  cfg.dram_bytes = 1 << 20;  // 1 MiB
  cfg.ssd_bytes = 0;
  TieredStore ts(cfg);

  auto blk = MakeBlock(/*seed=*/1, /*size=*/4096);
  const BlockId id = blk.id;
  auto bytes_in = blk.data;  // owned copy for later compare

  auto landed = ts.Put(blk, Tier::DRAM);
  assert(landed.has_value());
  assert(*landed == Tier::DRAM);

  auto got = ts.Get(id);
  assert(got.has_value());
  assert(got->tier_found == Tier::DRAM);
  assert(got->data == bytes_in);

  auto freed = ts.Erase(id);
  assert(freed == bytes_in.size());
  assert(!ts.Get(id).has_value());
  std::cout << "  TestTwoTierBasic: ok\n";
}

void TestTierHintFallthrough() {
  // DRAM intentionally tiny so a block with hint=DRAM has to fall
  // through to SSD.
  auto dir = MakeTempDir("hint_fallthrough");
  TieredStoreConfig cfg;
  cfg.dram_bytes = 4096;            // exactly fits ONE 4 KiB block
  cfg.ssd_bytes = 8ULL << 20;       // 8 MiB
  cfg.ssd_path = dir.string();
  cfg.enable_promotion = false;     // we're testing Put, not Get
  TieredStore ts(cfg);

  // First block fills DRAM.
  auto b1 = MakeBlock(/*seed=*/100, /*size=*/4096);
  auto where1 = ts.Put(b1, Tier::DRAM);
  assert(where1.has_value() && *where1 == Tier::DRAM);

  // Second block: hint=DRAM but DRAM full → falls through to SSD.
  auto b2 = MakeBlock(/*seed=*/101, /*size=*/4096);
  auto where2 = ts.Put(b2, Tier::DRAM);
  assert(where2.has_value());
  assert(*where2 == Tier::SSD);  // hint was DRAM; fallthrough → SSD

  // Both are findable.
  assert(ts.Get(b1.id).has_value());
  auto got2 = ts.Get(b2.id);
  assert(got2.has_value());
  assert(got2->tier_found == Tier::SSD);
  std::cout << "  TestTierHintFallthrough: ok\n";
}

void TestPromotionOnAccessThreshold() {
  auto dir = MakeTempDir("promotion");
  TieredStoreConfig cfg;
  cfg.dram_bytes = 64 << 10;
  cfg.ssd_bytes = 1ULL << 20;
  cfg.ssd_path = dir.string();
  cfg.enable_promotion = true;
  cfg.promotion_access_threshold = 2;  // promote on the 2nd Get
  TieredStore ts(cfg);

  auto blk = MakeBlock(/*seed=*/200, /*size=*/4096);
  const BlockId id = blk.id;
  auto where = ts.Put(blk, Tier::SSD);
  assert(where.has_value() && *where == Tier::SSD);

  // First Get: count becomes 1 (< threshold); no promotion yet.
  auto g1 = ts.Get(id);
  assert(g1.has_value());
  assert(g1->tier_found == Tier::SSD);
  assert(!g1->promoted);
  assert(ts.used_bytes(Tier::SSD) == 4096);
  assert(ts.used_bytes(Tier::DRAM) == 0);

  // Second Get: count becomes 2 (>= threshold); promotes to DRAM.
  auto g2 = ts.Get(id);
  assert(g2.has_value());
  assert(g2->tier_found == Tier::SSD);    // tier reported is where we
                                          // FOUND it, before promote
  assert(g2->promoted);

  // After promotion: SSD copy is gone, DRAM copy exists.
  assert(ts.used_bytes(Tier::SSD) == 0);
  assert(ts.used_bytes(Tier::DRAM) == 4096);

  // Third Get: finds it on DRAM now.
  auto g3 = ts.Get(id);
  assert(g3.has_value());
  assert(g3->tier_found == Tier::DRAM);
  // Already at DRAM - promotion target would be HBM, but HBM disabled
  // (hbm_bytes=0). No further promotion.
  assert(!g3->promoted);
  std::cout << "  TestPromotionOnAccessThreshold: ok\n";
}

void TestDemoteDramToSsd() {
  auto dir = MakeTempDir("demote");
  TieredStoreConfig cfg;
  cfg.dram_bytes = 1 << 20;
  cfg.ssd_bytes = 8ULL << 20;
  cfg.ssd_path = dir.string();
  cfg.enable_promotion = false;
  TieredStore ts(cfg);

  auto blk = MakeBlock(/*seed=*/300, /*size=*/4096);
  const BlockId id = blk.id;
  ts.Put(blk, Tier::DRAM);
  assert(ts.used_bytes(Tier::DRAM) == 4096);
  assert(ts.used_bytes(Tier::SSD) == 0);

  bool ok = ts.Demote(id);
  assert(ok);
  assert(ts.used_bytes(Tier::DRAM) == 0);
  assert(ts.used_bytes(Tier::SSD) == 4096);

  // Demote from SSD: no slower tier; returns false.
  ok = ts.Demote(id);
  assert(!ok);
  // Block still findable on SSD.
  assert(ts.Get(id).has_value());
  std::cout << "  TestDemoteDramToSsd: ok\n";
}

void TestEraseWipesAccessCounts() {
  TieredStoreConfig cfg;
  cfg.dram_bytes = 1 << 20;
  cfg.ssd_bytes = 0;
  TieredStore ts(cfg);

  auto blk = MakeBlock(/*seed=*/400, /*size=*/1024);
  const BlockId id = blk.id;
  ts.Put(blk, Tier::DRAM);

  (void)ts.Get(id);
  (void)ts.Get(id);
  assert(ts.access_count_for_testing(id) == 2);

  ts.Erase(id);
  // After Erase, access count entry is gone (returns 0 for absent).
  assert(ts.access_count_for_testing(id) == 0);

  // Re-Insert + Get behaves as fresh - count starts from 1.
  ts.Put(blk, Tier::DRAM);
  (void)ts.Get(id);
  assert(ts.access_count_for_testing(id) == 1);
  std::cout << "  TestEraseWipesAccessCounts: ok\n";
}

void TestSsdPersistenceAcrossDestruct() {
  // Write a block, drop the TieredStore (which closes the mmap), recreate
  // against the same file, read the block back.
  auto dir = MakeTempDir("persistence");
  TieredStoreConfig cfg;
  cfg.dram_bytes = 0;                  // force SSD path
  cfg.ssd_bytes = 1ULL << 20;
  cfg.ssd_path = dir.string();
  cfg.enable_promotion = false;

  auto blk = MakeBlock(/*seed=*/500, /*size=*/8192);
  const BlockId id = blk.id;
  const auto bytes_in = blk.data;

  {
    // dram_bytes=0 → no DRAM. Put with hint=SSD lands on SSD.
    // The TieredStoreConfig validation accepts dram_bytes=0; the
    // BlockStore for DRAM is constructed with capacity 0 and rejects
    // every Put with `false`. The Put-fallthrough chain still lands
    // the block on SSD.
    TieredStore ts(cfg);
    auto where = ts.Put(blk, Tier::SSD);
    assert(where.has_value() && *where == Tier::SSD);
    auto got = ts.Get(id);
    assert(got.has_value());
    assert(got->data == bytes_in);
  }

  // Reconstruct against the same path.
  {
    TieredStore ts(cfg);
    auto got = ts.Get(id);
    assert(got.has_value());
    assert(got->tier_found == Tier::SSD);
    assert(got->data == bytes_in);
    assert(ts.used_bytes(Tier::SSD) == bytes_in.size());
  }
  std::cout << "  TestSsdPersistenceAcrossDestruct: ok\n";
}

void TestSsdRejectsOversizedBlock() {
  auto dir = MakeTempDir("oversized");
  TieredStoreConfig cfg;
  cfg.dram_bytes = 0;
  cfg.ssd_bytes = 4ULL << 20;
  cfg.ssd_path = dir.string();
  cfg.ssd_slot_bytes = 4096;  // tiny slots → payload cap ~ 4032 bytes
  TieredStore ts(cfg);

  // Block larger than slot capacity must be rejected.
  auto blk = MakeBlock(/*seed=*/600, /*size=*/8192);
  auto where = ts.Put(blk, Tier::SSD);
  assert(!where.has_value());
  assert(!ts.Get(blk.id).has_value());
  std::cout << "  TestSsdRejectsOversizedBlock: ok\n";
}

void TestSnapshotPerTier() {
  auto dir = MakeTempDir("snapshot");
  TieredStoreConfig cfg;
  cfg.dram_bytes = 1 << 20;
  cfg.ssd_bytes = 1ULL << 20;
  cfg.ssd_path = dir.string();
  TieredStore ts(cfg);

  for (int i = 0; i < 5; ++i) {
    ts.Put(MakeBlock(700 + i, 1024), Tier::DRAM);
  }
  for (int i = 0; i < 3; ++i) {
    ts.Put(MakeBlock(800 + i, 1024), Tier::SSD);
  }
  assert(ts.Snapshot(Tier::DRAM).size() == 5);
  assert(ts.Snapshot(Tier::SSD).size() == 3);
  assert(ts.Snapshot(Tier::HBM).empty());
  std::cout << "  TestSnapshotPerTier: ok\n";
}

}  // namespace

int main() {
  std::cout << "test_tiered_store:\n";
  TestTwoTierBasic();
  TestTierHintFallthrough();
  TestPromotionOnAccessThreshold();
  TestDemoteDramToSsd();
  TestEraseWipesAccessCounts();
  TestSsdPersistenceAcrossDestruct();
  TestSsdRejectsOversizedBlock();
  TestSnapshotPerTier();
  std::cout << "test_tiered_store: ALL PASS\n";
  return 0;
}
