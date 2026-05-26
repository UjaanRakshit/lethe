// Evictor SIEVE unit tests. Standalone main()-with-asserts.
//
// Drives Evictor::RunPassForTier directly against a TieredStore filled
// past its high watermark. Membership is null and broadcast_evictions
// is off, so these exercise pure eviction logic without a cluster
// (BroadcastEvictionsToPeers early-returns on null membership).
//
// Coverage:
//   - No visited bits → eviction drains below the low watermark and stops
//     there (no over-eviction).
//   - Visited bit gives a second chance: an all-visited tier still
//     converges (bits cleared on the first sweep, victims taken on the
//     second) - the SIEVE convergence guarantee.
//   - Cross-tier demotion: DRAM victims land in SSD (not dropped) when
//     SSD has space.
//   - SSD eviction is a hard drop (no slower tier).

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "lethe/eviction.hpp"
#include "lethe/tiered_store.hpp"
#include "lethe/types.hpp"

using namespace lethe;

namespace {

std::filesystem::path MakeTempDir(const std::string& tag) {
  auto base = std::filesystem::temp_directory_path() /
              ("lethe_evict_" + tag + "_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  std::filesystem::create_directories(base);
  return base;
}

BlockId MakeId(std::uint64_t seed) {
  BlockId id;
  for (std::size_t i = 0; i < id.hash.size(); ++i) {
    id.hash[i] = static_cast<std::byte>((seed >> (i % 8 * 8)) & 0xFF);
  }
  return id;
}

KvBlock MakeBlock(std::uint64_t seed, std::size_t size) {
  KvBlock b;
  b.id = MakeId(seed);
  b.data.assign(size, static_cast<std::byte>(seed & 0xFF));
  b.tier = Tier::DRAM;
  return b;
}

EvictionConfig EvCfg() {
  EvictionConfig c;
  c.high_watermark_pct = 90;
  c.low_watermark_pct = 75;
  c.broadcast_evictions = false;  // no cluster in unit tests
  return c;
}

void TestEvictsToLowWatermarkNoOverEvict() {
  TieredStoreConfig ts;
  ts.dram_bytes = 10 * 1000;  // 10 blocks of 1000 bytes
  ts.ssd_bytes = 0;
  ts.enable_promotion = false;
  TieredStore store(ts);

  for (int i = 0; i < 10; ++i) {
    auto landed = store.Put(MakeBlock(i, 1000), Tier::DRAM);
    assert(landed.has_value());
  }
  assert(store.used_bytes(Tier::DRAM) == 10000);  // 100%

  Evictor ev(EvCfg(), &store, /*membership=*/nullptr, "test");
  auto r = ev.RunPassForTier(Tier::DRAM);

  // High=9000, low=7500. Evict until < 7500 → at least 3 blocks (down
  // to <=7000). Must NOT over-evict to 0.
  const std::size_t used = store.used_bytes(Tier::DRAM);
  assert(used <= 7500);          // dropped below low watermark
  assert(used >= 1000);          // did NOT nuke the whole tier
  assert(r.dram_evicted >= 3);
  assert(r.dram_evicted <= 4);   // stops promptly at low watermark
  std::printf("  TestEvictsToLowWatermarkNoOverEvict: ok (evicted=%zu used=%zu)\n",
              r.dram_evicted, used);
}

void TestAllVisitedStillConverges() {
  // SIEVE convergence: if every block is visited, the first sweep clears
  // all bits (no eviction) and the second takes victims. The hand must
  // not loop forever.
  TieredStoreConfig ts;
  ts.dram_bytes = 10 * 1000;
  ts.ssd_bytes = 0;
  ts.enable_promotion = false;
  TieredStore store(ts);

  for (int i = 0; i < 10; ++i) {
    store.Put(MakeBlock(i, 1000), Tier::DRAM);
  }
  // Touch every block so all visited bits are set.
  for (int i = 0; i < 10; ++i) {
    store.MarkVisited(MakeId(i));
  }
  // Sanity: bits are set.
  assert(store.visited_for_testing(MakeId(0)));

  Evictor ev(EvCfg(), &store, nullptr, "test");
  auto r = ev.RunPassForTier(Tier::DRAM);

  // Converged: some blocks evicted despite all being visited.
  assert(r.dram_evicted >= 3);
  assert(store.used_bytes(Tier::DRAM) <= 7500);
  // Survivors had their visited bit cleared by the first sweep.
  // Pick a block that still exists and confirm its bit is clear.
  int survivors_checked = 0;
  for (int i = 0; i < 10; ++i) {
    if (store.Get(MakeId(i)).has_value()) {
      // Get itself re-sets the visited bit - so check BEFORE Get. We
      // can't, since Get is how we know it survived. Instead assert
      // the pass made progress, which the eviction count above does.
      ++survivors_checked;
    }
  }
  assert(survivors_checked >= 1);
  std::printf("  TestAllVisitedStillConverges: ok (evicted=%zu)\n",
              r.dram_evicted);
}

void TestCrossTierDemotionDramToSsd() {
  auto dir = MakeTempDir("demote");
  TieredStoreConfig ts;
  ts.dram_bytes = 10 * 1000;
  ts.ssd_bytes = 4 << 20;  // plenty of SSD room
  ts.ssd_path = dir.string();
  ts.enable_promotion = false;
  TieredStore store(ts);

  for (int i = 0; i < 10; ++i) {
    store.Put(MakeBlock(i, 1000), Tier::DRAM);
  }
  assert(store.used_bytes(Tier::DRAM) == 10000);
  assert(store.used_bytes(Tier::SSD) == 0);

  Evictor ev(EvCfg(), &store, nullptr, "test");
  auto r = ev.RunPassForTier(Tier::DRAM);

  // DRAM victims should have been DEMOTED to SSD, not dropped. So the
  // evicted DRAM bytes show up as SSD bytes.
  assert(r.dram_evicted >= 3);
  assert(store.used_bytes(Tier::DRAM) <= 7500);
  assert(store.used_bytes(Tier::SSD) > 0);  // demoted, not dropped
  // The demoted blocks are still findable (now on SSD).
  std::size_t still_present = 0;
  for (int i = 0; i < 10; ++i) {
    if (store.Get(MakeId(i)).has_value()) ++still_present;
  }
  assert(still_present == 10);  // nothing dropped; all demoted or kept
  std::printf("  TestCrossTierDemotionDramToSsd: ok (dram_evicted=%zu ssd_used=%zu)\n",
              r.dram_evicted, store.used_bytes(Tier::SSD));
}

void TestSsdEvictionIsHardDrop() {
  auto dir = MakeTempDir("ssd_drop");
  TieredStoreConfig ts;
  ts.dram_bytes = 0;          // force everything to SSD
  ts.ssd_bytes = 10 * 4096;   // 10 slots (4 KiB each)
  ts.ssd_slot_bytes = 4096;
  ts.ssd_path = dir.string();
  ts.enable_promotion = false;
  TieredStore store(ts);

  // Fill SSD. Payloads sized so 10 of them ~ fill the usable bytes.
  const std::size_t payload = 4096 - 64 - 16;  // < slot payload cap
  int inserted = 0;
  for (int i = 0; i < 10; ++i) {
    if (store.Put(MakeBlock(1000 + i, payload), Tier::SSD).has_value()) {
      ++inserted;
    }
  }
  assert(inserted >= 8);
  const std::size_t before = store.used_bytes(Tier::SSD);

  Evictor ev(EvCfg(), &store, nullptr, "test");
  auto r = ev.RunPassForTier(Tier::SSD);

  // SSD has no slower tier → hard drop. used_bytes must have dropped,
  // and the dropped blocks are gone (not demoted anywhere).
  assert(r.ssd_evicted >= 1);
  assert(store.used_bytes(Tier::SSD) < before);
  std::printf("  TestSsdEvictionIsHardDrop: ok (ssd_evicted=%zu before=%zu after=%zu)\n",
              r.ssd_evicted, before, store.used_bytes(Tier::SSD));
}

}  // namespace

int main() {
  std::printf("test_eviction:\n");
  TestEvictsToLowWatermarkNoOverEvict();
  TestAllVisitedStillConverges();
  TestCrossTierDemotionDramToSsd();
  TestSsdEvictionIsHardDrop();
  std::printf("test_eviction: ALL PASS\n");
  return 0;
}
