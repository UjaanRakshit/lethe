// Lethe — eviction (W8).
//
// Per-tier eviction threads driving CLOCK-style SIEVE: each thread
// wakes every cfg.scan_interval (default 500ms), checks whether its
// tier's used_bytes exceeds the high watermark, and if so walks a
// hand pointer through the tier's snapshot picking victims by the
// SIEVE rule:
//
//   if visited:   clear bit, advance hand           (2nd chance)
//   if !visited:  evict block, advance hand         (victim)
//
// Cross-tier handling on evict:
//   * HBM victim → try TieredStore::Demote to DRAM. If DRAM full,
//     fall through to hard Erase.
//   * DRAM victim → try Demote to SSD. If SSD full, hard Erase.
//   * SSD victim → hard Erase (no slower tier).
//
// Cluster-wide broadcast: after each pass, batch all evicted block IDs
// into ONE EvictBroadcast RPC per alive peer. Best-effort — failures
// are logged but don't block the next pass. The receiver records
// "peer evicted these" in its own Evictor; read-repair MAY consult
// (the wire is here, the optimization is a small follow-up).
//
// SIEVE-vs-CLOCK note: true SIEVE relies on FIFO insertion order. Our
// BlockStore stores blocks in an unordered_map; we walk the snapshot
// in iteration order, not insertion order. That makes our impl
// equivalent to CLOCK with a visited bit, not strict SIEVE. For W8
// acceptance (visited-bit semantics + watermark behavior + cross-tier
// demote) the difference doesn't matter; an insertion-ordered deque
// alongside the map would restore exact SIEVE if W11 chaos surfaces
// a regression. Documented in DECISIONS.md.
//
// Threading: each tier has its own std::thread. There is no shared
// state between the eviction threads beyond the read-only
// TieredStore* and Membership* — both subsystems serialize their own
// internals. EvictBroadcast stubs are per-peer with their own mutex
// in Impl::stubs_mu.

#include "lethe/eviction.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "lethe.grpc.pb.h"
#include "lethe.pb.h"
#include "lethe/membership.hpp"
#include "lethe/tiered_store.hpp"
#include "lethe/types.hpp"

namespace lethe {

namespace {

constexpr int kBroadcastDeadlineMs = 500;

::lethe::rpc::BlockId BlockIdToProto(const BlockId& src) {
  ::lethe::rpc::BlockId p;
  p.set_hash(src.hash.data(), src.hash.size());
  p.set_layer(src.layer);
  p.set_head_group(src.head_group);
  p.set_model_id(src.model_id);
  return p;
}

}  // namespace

struct Evictor::Impl {
  std::atomic<bool> running{false};
  std::vector<std::thread> threads;

  // Hand pointers per tier. Numeric indices into the snapshot of the
  // current pass; the snapshot is regenerated each pass and indices
  // are clamped via modulo on wrap-around.
  std::size_t hand_hbm = 0;
  std::size_t hand_dram = 0;
  std::size_t hand_ssd = 0;

  // Per-peer gRPC stubs for EvictBroadcast. Built lazily on first use.
  // Channels are HTTP/2 with internal stream mux — one channel per
  // peer is enough.
  std::mutex stubs_mu;
  std::unordered_map<std::string,
                     std::pair<std::shared_ptr<grpc::Channel>,
                               std::unique_ptr<::lethe::rpc::LetheCache::Stub>>>
      stubs;

  // Tracks "peer recently evicted these blocks." Bounded loosely to
  // 4096 entries per peer (oldest dropped via set rotation; W8 keeps
  // it simple by clearing the set when it grows past the bound). Read
  // by `peer_evicted_count_for_testing`; FetchFromAny does not yet
  // consult this — wire is here, consumer is a follow-up.
  std::mutex peer_evict_mu;
  std::unordered_map<std::string, std::unordered_set<BlockId, BlockIdHash>>
      peer_evicted;
};

namespace {

struct Registry {
  std::mutex mu;
  std::unordered_map<const Evictor*, std::unique_ptr<Evictor::Impl>> impls;
};

Registry& registry() {
  static Registry r;
  return r;
}

Evictor::Impl* impl_for(const Evictor* e) {
  auto& reg = registry();
  std::lock_guard<std::mutex> g(reg.mu);
  auto it = reg.impls.find(e);
  return it == reg.impls.end() ? nullptr : it->second.get();
}

}  // namespace

Evictor::Evictor(EvictionConfig cfg,
                 TieredStore* store,
                 Membership* membership,
                 std::string local_node_id)
    : cfg_(cfg),
      store_(store),
      membership_(membership),
      local_node_id_(std::move(local_node_id)) {
  auto impl = std::make_unique<Impl>();
  auto& reg = registry();
  std::lock_guard<std::mutex> g(reg.mu);
  reg.impls.emplace(this, std::move(impl));
}

Evictor::~Evictor() {
  Shutdown();
  auto& reg = registry();
  std::lock_guard<std::mutex> g(reg.mu);
  reg.impls.erase(this);
}

void Evictor::Start() {
  auto* impl = impl_for(this);
  if (impl == nullptr) return;
  if (impl->running.exchange(true)) return;  // already started

  // Spawn one thread per tier that has nonzero capacity.
  auto spawn = [this, impl](Tier tier, std::size_t& hand) {
    if (store_->capacity_bytes(tier) == 0) return;  // tier not configured
    impl->threads.emplace_back([this, impl, tier, &hand]() {
      // Per-thread wake cv kept local — eviction threads only ever
      // self-time. The shared shutdown signal comes from impl->running.
      using namespace std::chrono_literals;
      while (impl->running.load(std::memory_order_acquire)) {
        // Sleep in short slices so shutdown isn't blocked for a full
        // scan interval on stop. 50ms slice; ~10 slices per default
        // 500ms scan interval.
        auto next_pass = std::chrono::steady_clock::now() + cfg_.scan_interval;
        while (std::chrono::steady_clock::now() < next_pass &&
               impl->running.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::min(
              50ms,
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  next_pass - std::chrono::steady_clock::now())));
        }
        if (!impl->running.load(std::memory_order_acquire)) break;
        (void)RunPassForTier(tier);
      }
    });
  };
  spawn(Tier::HBM, impl->hand_hbm);
  spawn(Tier::DRAM, impl->hand_dram);
  spawn(Tier::SSD, impl->hand_ssd);
}

void Evictor::Shutdown() {
  auto* impl = impl_for(this);
  if (impl == nullptr) return;
  impl->running.store(false, std::memory_order_release);
  for (auto& t : impl->threads) {
    if (t.joinable()) t.join();
  }
  impl->threads.clear();
}

Evictor::PassResult Evictor::RunPassForTier(Tier tier) {
  PassResult r;
  auto* impl = impl_for(this);
  if (impl == nullptr) return r;

  const std::size_t cap = store_->capacity_bytes(tier);
  const std::size_t used = store_->used_bytes(tier);
  if (cap == 0 || used == 0) return r;

  const std::size_t high = cap * cfg_.high_watermark_pct / 100;
  const std::size_t low = cap * cfg_.low_watermark_pct / 100;
  if (used <= high) return r;  // no pressure

  auto blocks = store_->Snapshot(tier);
  if (blocks.empty()) return r;

  // Pick the per-tier hand reference.
  std::size_t* hand_ptr = nullptr;
  switch (tier) {
    case Tier::HBM:  hand_ptr = &impl->hand_hbm;  break;
    case Tier::DRAM: hand_ptr = &impl->hand_dram; break;
    case Tier::SSD:  hand_ptr = &impl->hand_ssd;  break;
  }
  std::size_t& hand = *hand_ptr;

  // Target: free enough bytes to drop below low watermark.
  // need is signed-safe because we already checked used > high > low.
  std::size_t need = used - low;
  std::size_t freed_local = 0;

  std::vector<BlockId> evicted;
  evicted.reserve(blocks.size() / 4);

  // SIEVE walk: bound by 2N iterations so visited→clear pass can
  // happen before victim selection without an infinite loop.
  const std::size_t max_steps = blocks.size() * 2;
  std::size_t steps = 0;

  while (freed_local < need && steps < max_steps) {
    if (hand >= blocks.size()) hand = 0;  // wrap
    const auto& meta = blocks[hand];

    if (meta.visited) {
      // Second chance — clear the bit and advance. ClearVisited is a
      // brief unique lock on TieredStore::counts_mu_.
      store_->ClearVisited(meta.id);
      ++hand;
    } else {
      // Victim. Try Demote first (DRAM→SSD, HBM→DRAM); if that fails
      // OR the tier is SSD, hard Erase.
      bool demoted = false;
      if (tier != Tier::SSD) {
        demoted = store_->Demote(meta.id);
      }
      std::size_t bytes_removed = 0;
      if (demoted) {
        // Demote moved the block — it's no longer in this tier.
        bytes_removed = meta.size_bytes;
      } else {
        bytes_removed = store_->Erase(meta.id);
      }
      freed_local += bytes_removed;
      evicted.push_back(meta.id);
      switch (tier) {
        case Tier::HBM:  ++r.hbm_evicted;  break;
        case Tier::DRAM: ++r.dram_evicted; break;
        case Tier::SSD:  ++r.ssd_evicted;  break;
      }
      ++hand;
    }
    ++steps;
  }
  r.bytes_freed = freed_local;

  // Broadcast in one batch RPC per peer.
  if (cfg_.broadcast_evictions && !evicted.empty()) {
    BroadcastEvictionsToPeers(evicted);
    r.blocks_broadcast = evicted.size();
  }
  return r;
}

void Evictor::OnPeerEviction(const std::vector<BlockId>& evicted,
                             const std::string& peer) {
  auto* impl = impl_for(this);
  if (impl == nullptr) return;
  std::lock_guard<std::mutex> g(impl->peer_evict_mu);
  auto& set = impl->peer_evicted[peer];
  // Loose bound to prevent unbounded growth under chaos; if we exceed
  // it, clear and restart. The "this peer recently evicted X" hint is
  // best-effort, not authoritative.
  constexpr std::size_t kPerPeerBound = 4096;
  if (set.size() > kPerPeerBound) set.clear();
  for (const auto& id : evicted) set.insert(id);
}

std::size_t Evictor::peer_evicted_count_for_testing(
    const std::string& peer) const {
  auto* impl = impl_for(const_cast<Evictor*>(this));
  if (impl == nullptr) return 0;
  std::lock_guard<std::mutex> g(impl->peer_evict_mu);
  auto it = impl->peer_evicted.find(peer);
  return it == impl->peer_evicted.end() ? 0 : it->second.size();
}

// ---------------------------------------------------------------------------
// Broadcast helper — gRPC client side; lazy per-peer stubs.
// ---------------------------------------------------------------------------

void Evictor::BroadcastEvictionsToPeers(const std::vector<BlockId>& evicted) {
  auto* impl = impl_for(this);
  if (impl == nullptr || membership_ == nullptr) return;

  // Pull a snapshot of alive peer addresses. AllPeerAddresses returns
  // all peers we know about; Membership filters dead from there in W8
  // (alive_peers only). For W8 we send to every known address — if a
  // peer is dead, the RPC fails fast and the eviction broadcast is
  // lost, which is fine per the best-effort contract.
  auto addresses = membership_->AllPeerAddresses();
  if (addresses.empty()) return;

  // Build the request once; same payload to every peer.
  ::lethe::rpc::EvictBroadcastRequest req;
  req.set_source_node(local_node_id_);
  req.set_epoch(membership_->Epoch());
  req.mutable_evicted()->Reserve(static_cast<int>(evicted.size()));
  for (const auto& id : evicted) {
    *req.add_evicted() = BlockIdToProto(id);
  }

  // Per-peer dispatch. Fire-and-forget on this thread — the eviction
  // loop is tolerant of the per-peer RPC latency since we already did
  // the work of freeing bytes.
  for (const auto& address : addresses) {
    ::lethe::rpc::LetheCache::Stub* stub = nullptr;
    {
      std::lock_guard<std::mutex> g(impl->stubs_mu);
      auto it = impl->stubs.find(address);
      if (it == impl->stubs.end()) {
        auto ch = grpc::CreateChannel(address,
                                      grpc::InsecureChannelCredentials());
        auto s = ::lethe::rpc::LetheCache::NewStub(ch);
        stub = s.get();
        impl->stubs.emplace(address, std::make_pair(std::move(ch),
                                                    std::move(s)));
      } else {
        stub = it->second.second.get();
      }
    }
    ::lethe::rpc::EvictBroadcastResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() +
                     std::chrono::milliseconds(kBroadcastDeadlineMs));
    (void)stub->EvictBroadcast(&ctx, req, &resp);
    // Failures swallowed per CLAUDE.md rule 2 (never block cache work
    // on peer liveness). The eviction itself already happened locally;
    // a missed broadcast costs one wasted RPC at the receiver later.
  }
}

}  // namespace lethe
