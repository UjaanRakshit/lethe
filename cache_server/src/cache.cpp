// Lethe — top-level cache facade (W1 → W3-W4).
//
// W1 scope was single-node, DRAM-only. W3 adds Router (consistent
// hash, owns ring) and Membership (holds the static peer table,
// returns it on heartbeat). W4 adds Replicator (async push to
// replica successors + read-repair pull from primary on local
// miss). Evictor, transport, and Metrics remain stubbed until
// their respective weeks (W8, W5-6, W10).
//
// Routing semantics:
//   - Insert: write LOCALLY first, ACK to caller. If Replicator
//     is present and we are primary, kick off async ReplicateOut
//     to the R-1 successors. Inserts on a non-primary node still
//     write locally (W1 behavior) — the W4 client routes to the
//     primary, but the proto allows out-of-band Inserts and we
//     accept them so the server doesn't reject the gRPC.
//   - Lookup: per-block routing. If local IS primary and we have
//     it → LocalHit. If local is primary but miss → try read-
//     repair against the R-1 replicas. If local is a replica
//     (not primary) and we have it → LocalHit (free hit; the
//     primary may not even know). Otherwise → RemoteHit pointing
//     at primary, OR Miss if there's no Router yet.
//   - IngestStreamedBlock: stores blindly into the local tier.
//     The push-shaped flows (replication, read-repair pull,
//     prefill→decode) all land here.
//
// Lifetime contract (cache.hpp::LookupResult::Entry::local_data):
// since W7, `local_data` is an OWNED std::vector<std::byte> that the
// Lookup path moves into the entry. The W1 "borrowed span, valid
// until next mutation" contract is retired — the SSD tier can't
// safely lend spans into mmap'd slots across reuse, and uniform
// ownership beats per-tier dispatch (one memcpy per Get is in the
// noise at 64 KiB block sizes). Callers can hold the entry as long
// as they need; gRPC shim copies into the wire bytes and discards.

#include "lethe/cache.hpp"

#include <algorithm>
#include <utility>

// Every forward-declared subsystem in cache.hpp needs its complete
// type visible at the point where LetheCache's destructor instantiates
// the unique_ptr destructors. Even though they're null in W1, the
// compiler still needs to be able to delete them in principle.
#include "lethe/eviction.hpp"
#include "lethe/kv_transport.hpp"
#include "lethe/membership.hpp"
#include "lethe/metrics.hpp"
#include "lethe/replication.hpp"
#include "lethe/routing.hpp"
#include "lethe/tiered_store.hpp"

namespace lethe {

LetheCache::LetheCache(CacheConfig cfg) : cfg_(std::move(cfg)) {
  TieredStoreConfig ts;
  ts.hbm_bytes = cfg_.hbm_bytes;          // W1: ignored unless 0.
  ts.dram_bytes = cfg_.dram_bytes;
  ts.ssd_bytes = cfg_.ssd_bytes;          // W7.
  ts.ssd_path = cfg_.ssd_path;
  store_ = std::make_unique<TieredStore>(std::move(ts));

  // W3 (corrected at W8): Router needs the FULL cluster peer set
  // including the local node — otherwise IsLocalPrimary / IsLocalReplica
  // never return true on this node and Lookup's read-repair branch
  // (and W8's TriggerReReplication "we_in_route" heuristic) never
  // fire. main.cpp's --peers parser strips self from the seed list
  // (the on-wire format names OTHER nodes); we add ourselves back
  // here so the ring matches the Python client's HashRing
  // construction, which is built with the full peer list.
  // Empty seed_peers → single-node mode; the ring contains only us
  // and routes everything locally.
  router_ = std::make_unique<Router>(
      cfg_.node_id,
      cfg_.virtual_nodes_per_peer,
      cfg_.replication_factor);
  {
    std::vector<std::string> peer_ids;
    peer_ids.reserve(cfg_.seed_peers.size() + 1);
    peer_ids.push_back(cfg_.node_id);   // self always in the ring
    for (const auto& p : cfg_.seed_peers) {
      if (p.node_id != cfg_.node_id) peer_ids.push_back(p.node_id);
    }
    router_->SetPeers(std::move(peer_ids));
  }

  // W5-6: transport built BEFORE Replicator (Replicator now delegates
  // all peer-to-peer block movement through KvTransport). GATE #1 of
  // W5-6 fired on WSL2 (no infiniband subsystem in the dev kernel),
  // so we ship GrpcStreamTransport as the production data path. The
  // IbverbsTransport class still exists but is a defined-symbol stub
  // that aborts if constructed; the factory below never picks it
  // because RdmaIsAvailable() is always-false in the absence of real
  // hardware. See docs/decisions/W5_rdma_fallback.md.
  RdmaConfig rdma_cfg;
  rdma_cfg.device_name = cfg_.rdma_device;
  rdma_cfg.listen_port = cfg_.rdma_port;
  transport_ = std::make_unique<GrpcStreamTransport>(rdma_cfg,
                                                     KvTransport::OnReceiveFn{});
  transport_->Start();

  // W4: Replicator must exist BEFORE Membership (Membership holds a
  // pointer to it for W8's membership-change re-replication driving).
  // The transport's connection pool is populated here from the static
  // peer set so ReplicateOut on the very first Insert can route
  // immediately (no warm-up RPC needed). EnsurePeerClient delegates to
  // transport_->Connect.
  replicator_ = std::make_unique<Replicator>(
      cfg_.node_id, router_.get(), store_.get(), transport_.get());
  for (const auto& p : cfg_.seed_peers) {
    if (p.node_id == cfg_.node_id) continue;
    replicator_->EnsurePeerClient(p.node_id, p.address);
  }

  // W3 / W8: Membership constructed with the static peer set.
  // W8 lights up the real heartbeat thread + failure-detector loop;
  // Start() below kicks them off. Membership holds Router* +
  // Replicator* and calls into both on membership-change events.
  MembershipConfig mcfg{};
  membership_ = std::make_unique<Membership>(
      mcfg,
      cfg_.node_id,
      cfg_.seed_peers,
      router_.get(),
      replicator_.get());

  // W8: Evictor runs per-tier SIEVE scans and gossips evictions via
  // EvictBroadcast. Takes TieredStore + Membership for the alive-peer
  // address list. Watermark + scan-interval default; tune via
  // EvictionConfig if pressure shape changes.
  EvictionConfig ecfg{};
  ecfg.high_watermark_pct = cfg_.eviction_high_watermark_pct;
  ecfg.low_watermark_pct = cfg_.eviction_low_watermark_pct;
  evictor_ = std::make_unique<Evictor>(
      ecfg, store_.get(), membership_.get(), cfg_.node_id);
}

LetheCache::~LetheCache() {
  Shutdown();
}

void LetheCache::Start() {
  if (running_.exchange(true)) return;  // idempotent
  // Start order matches the dependency chain: Membership before
  // Evictor (Evictor reads Membership::AllPeerAddresses for
  // broadcast). transport_->Start() was already called in the ctor.
  if (membership_) membership_->Start();
  if (evictor_)    evictor_->Start();
}

void LetheCache::Shutdown() {
  if (!running_.exchange(false)) return;  // already stopped
  // Reverse of Start. Evictor first (joins its tier threads) — those
  // threads can call into Membership::AllPeerAddresses during a
  // final pass, so Membership stays alive. Then Membership
  // (joins heartbeat thread). Then close transport channels.
  // Member destruction handles the unique_ptr cleanup in the field
  // order encoded by cache.hpp.
  if (evictor_)    evictor_->Shutdown();
  if (membership_) membership_->Shutdown();
  if (transport_)  transport_->Shutdown();
}

LookupResult LetheCache::Lookup(const std::vector<BlockId>& ids,
                                const std::string& /*request_id*/,
                                const std::string& /*requesting_node*/) {
  LookupResult result;
  result.entries.reserve(ids.size());
  for (const auto& id : ids) {
    LookupResult::Entry e;
    e.id = id;

    // First branch: try the local store. A LocalHit short-circuits
    // even if this node isn't the primary for the block — having the
    // bytes locally is better than a network round-trip. W7: Get
    // returns OWNED bytes (vector) because the SSD tier can't lend
    // spans safely; move them straight into the entry.
    if (auto got = store_->Get(id); got.has_value()) {
      e.where = LookupResult::Entry::Where::LocalHit;
      e.tier = got->tier_found;
      e.local_data = std::move(got->data);
      ++result.hit_count;
      result.entries.push_back(std::move(e));
      continue;
    }

    // Local miss. Consult the Router.
    if (router_) {
      const auto route = router_->Route(id);
      const bool we_are_primary =
          !route.primary.empty() && route.primary == cfg_.node_id;
      const bool we_are_in_route =
          we_are_primary ||
          std::any_of(route.replicas.begin(), route.replicas.end(),
                      [&](const std::string& p) { return p == cfg_.node_id; });

      // Read-repair branch: we are in the route (primary or replica)
      // but locally missed. Try FetchFromAny against the other peers
      // in the route. On success, write locally + return as LocalHit.
      // On failure, fall through to Miss.
      if (we_are_in_route && replicator_ != nullptr) {
        std::vector<std::string> repair_peers;
        if (route.primary != cfg_.node_id && !route.primary.empty()) {
          repair_peers.push_back(route.primary);
        }
        for (const auto& p : route.replicas) {
          if (p != cfg_.node_id) repair_peers.push_back(p);
        }
        if (!repair_peers.empty()) {
          auto fetched = replicator_->FetchFromAny(id, repair_peers);
          if (fetched.has_value()) {
            // Repair: write to local store. ID was preserved by
            // FetchFromAny so the next Get will hit.
            store_->Put(*fetched, Tier::DRAM);
            // Re-Get so the returned local_data is an owned copy of
            // the freshly-stored bytes (the Put consumed `fetched`).
            if (auto repaired = store_->Get(id); repaired.has_value()) {
              e.where = LookupResult::Entry::Where::LocalHit;
              e.tier = repaired->tier_found;
              e.local_data = std::move(repaired->data);
              ++result.hit_count;
              result.entries.push_back(std::move(e));
              continue;
            }
          }
        }
      }

      // Not in the route (e.g. client's ring was stale during a
      // membership change). Report RemoteHit pointing at the primary
      // so the client can re-route.
      if (!route.primary.empty() && route.primary != cfg_.node_id) {
        e.where = LookupResult::Entry::Where::RemoteHit;
        e.remote_node = route.primary;
        ++result.hit_count;
        result.entries.push_back(e);
        continue;
      }
    }

    e.where = LookupResult::Entry::Where::Miss;
    ++result.miss_count;
    result.entries.push_back(e);
  }
  return result;
}

std::uint32_t LetheCache::Insert(std::vector<KvBlock> blocks,
                                 const std::string& /*request_id*/,
                                 const std::string& /*source_node*/,
                                 InsertOptions /*opts*/) {
  // W4 async replication + W7 tier-aware insert: write locally first
  // (using blk.tier as the hint, defaulting to DRAM) → ACK → kick off
  // background ReplicateOut. opts.sync_replicate is still accepted but
  // unimplemented; the W0 decision (async by default) holds.
  //
  // "Accepted" semantics: counts NEWLY-INSERTED blocks. A Put against
  // an already-present BlockId is idempotent (content-addressed; same
  // hash → same bytes) and counts as 0, NOT 1. The W4 client roundtrip
  // test (test_repeated_insert_is_idempotent) enforces this. We detect
  // newness by total-used-bytes delta across all tiers — Put can land
  // a new block in any tier (HBM/DRAM/SSD per the fallthrough chain),
  // so we sum the deltas rather than watching one tier.
  auto total_used = [&]() -> std::size_t {
    return store_->used_bytes(Tier::HBM) + store_->used_bytes(Tier::DRAM) +
           store_->used_bytes(Tier::SSD);
  };

  std::uint32_t accepted = 0;
  for (auto& blk : blocks) {
    // Capture the hint before std::move-ing blk into Put. The hint
    // comes from the InsertRequest.Block.tier_hint (main.cpp shim sets
    // it). Default for callers that don't set it: DRAM.
    const Tier hint = blk.tier;
    // Save a copy of the block for the replication queue BEFORE we
    // std::move it into the local store. The replication task needs
    // its own owned bytes since it runs after Insert returns.
    KvBlock to_replicate = blk;
    const std::size_t before = total_used();
    const auto landed = store_->Put(std::move(blk), hint);
    if (landed.has_value() && total_used() > before) {
      ++accepted;
      if (replicator_ != nullptr) {
        // Fire-and-forget. ReplicateOut returns the peer_ids it
        // queued the push to; we don't wait. Failures are logged
        // via the Replicator's overflow_drops / replicate_failures
        // counters (W10 will surface these as metrics).
        replicator_->ReplicateOut(to_replicate);
      }
    }
  }
  return accepted;
}

void LetheCache::IngestStreamedBlock(BlockId id,
                                     std::vector<std::byte> payload,
                                     StreamPurpose /*purpose*/) {
  // W1: route every streamed block into DRAM. Purpose is recorded by
  // the caller's metric (W10) but doesn't change tier selection yet.
  KvBlock blk;
  blk.id = id;
  blk.data = std::move(payload);
  blk.tier = Tier::DRAM;
  blk.inserted_epoch = cluster_epoch();
  store_->Put(std::move(blk), Tier::DRAM);
}

HeartbeatReply LetheCache::OnHeartbeat(const std::string& peer_id,
                                       std::uint64_t peer_epoch) {
  if (membership_) {
    return membership_->OnHeartbeat(peer_id, peer_epoch);
  }
  // W1: no membership → return an empty reply with epoch=0. The gRPC
  // shim ships this as HeartbeatResponse(cluster_epoch=0, alive_peers=[]).
  return HeartbeatReply{};
}

void LetheCache::OnEvictBroadcast(const std::vector<BlockId>& evicted,
                                  const std::string& source) {
  // W8: delegate to Evictor::OnPeerEviction so the read-repair routing
  // table can later skip peers that just evicted the same block. The
  // skip-peer consumer in FetchFromAny is deferred; the wire is in
  // place and the data is tracked.
  if (evictor_) evictor_->OnPeerEviction(evicted, source);
}

std::uint64_t LetheCache::cluster_epoch() const noexcept {
  // Source of truth is Membership::Epoch() (W0 contract). When
  // membership_ is null (W1, pre-cluster), there is no cluster — the
  // epoch is by definition 0.
  return membership_ ? membership_->Epoch() : 0;
}

}  // namespace lethe
