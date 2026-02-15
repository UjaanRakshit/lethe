// Top-level cache facade. Owns Router (consistent hash + ring), Membership
// (static peer table), Replicator (async push + read-repair), Evictor,
// transport, and Metrics, and delegates all RPC handlers to them.
//
// Routing semantics:
//   - Insert: write LOCALLY first, ACK to caller. If we're primary, kick off
//     async ReplicateOut to the R-1 successors. Inserts on a non-primary node
//     still write locally — the client routes to the primary, but the proto
//     allows out-of-band Inserts and we accept them.
//   - Lookup: per-block routing. Local hit short-circuits even on a replica
//     (free hit). Otherwise RemoteHit pointing at primary, or Miss if there's
//     no Router.
//   - IngestStreamedBlock: stores blindly into the local tier. The push-shaped
//     flows (replication, read-repair pull, prefill→decode) all land here.
//
// Lifetime (cache.hpp::LookupResult::Entry::local_data): `local_data` is an
// OWNED vector the Lookup path moves into the entry. The SSD tier can't safely
// lend spans into mmap'd slots across reuse, and uniform ownership beats
// per-tier dispatch (one memcpy per Get is noise at 64 KiB blocks). The gRPC
// shim copies into the wire bytes and discards.

#include "lethe/cache.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>

// Every forward-declared subsystem in cache.hpp needs its complete type
// visible where LetheCache's destructor instantiates the unique_ptr
// destructors.
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
  ts.hbm_bytes = cfg_.hbm_bytes;
  ts.dram_bytes = cfg_.dram_bytes;
  ts.ssd_bytes = cfg_.ssd_bytes;
  ts.ssd_path = cfg_.ssd_path;
  store_ = std::make_unique<TieredStore>(std::move(ts));

  // Router needs the FULL cluster peer set including the local node —
  // otherwise IsLocalPrimary / IsLocalReplica never return true here and
  // Lookup's route-aware branch (and TriggerReReplication's "we_in_route"
  // heuristic) never fire. main.cpp's --peers parser strips self from the
  // seed list (the on-wire format names OTHER nodes); we add ourselves back
  // so the ring matches the Python client's HashRing, built with the full
  // peer list. Empty seed_peers → single-node mode; the ring contains only
  // us and routes everything locally.
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

  // Transport is built BEFORE Replicator (Replicator delegates all
  // peer-to-peer block movement through KvTransport). The dev kernel has no
  // infiniband subsystem, so GrpcStreamTransport is the default data path.
  // IbverbsTransport still exists but is a defined-symbol stub that aborts if
  // constructed; the factory below never picks it because RdmaIsAvailable() is
  // always-false without real hardware. See docs/decisions/W5_rdma_fallback.md.
  RdmaConfig rdma_cfg;
  rdma_cfg.device_name = cfg_.rdma_device;
  rdma_cfg.listen_port = cfg_.rdma_port;
#ifdef LETHE_ENABLE_RDMA
  // Real IbverbsTransport, selected at runtime by LETHE_USE_RDMA so a single
  // binary serves the gRPC-vs-RDMA A/B benchmark. RDMA receives don't flow
  // through the gRPC service handlers, so the transport gets an on_receive
  // callback that lands a pushed block exactly as the gRPC Insert handler
  // would — same Insert() path, same idempotent re-replication bounce.
  if (std::getenv("LETHE_USE_RDMA") != nullptr) {
    transport_ = std::make_unique<IbverbsTransport>(
        rdma_cfg,
        [this](BlockId id, std::vector<std::byte> bytes, StreamPurpose) {
          KvBlock blk;
          blk.id = id;
          blk.data = std::move(bytes);
          blk.tier = Tier::DRAM;
          std::vector<KvBlock> one;
          one.push_back(std::move(blk));
          this->Insert(std::move(one), "", "rdma-recv");
        });
  } else
#endif
  {
    transport_ = std::make_unique<GrpcStreamTransport>(
        rdma_cfg, KvTransport::OnReceiveFn{});
  }
  transport_->Start();

  // Metrics is a leaf — everyone records, no one reads. Construct it before
  // the subsystems that record into it (Replicator, Membership, Evictor). Its
  // ctor spawns the /metrics HTTP thread on cfg_.metrics_port; bind failure is
  // non-fatal (see metrics.cpp), so a shared-host multi-node test with
  // colliding ports still runs.
  //
  // Gated on LETHE_ENABLE_METRICS (CMake option, default ON). When OFF the
  // flag is undefined, metrics_ stays null, and every Record* call is a no-op
  // via its `if (metrics_)` guard — the hot path pays nothing. Unit tests link
  // cache.cpp without the flag, so they skip the HTTP server (test_metrics.cpp
  // drives Metrics directly).
#ifdef LETHE_ENABLE_METRICS
  metrics_ = std::make_unique<Metrics>(
      "0.0.0.0:" + std::to_string(cfg_.metrics_port), cfg_.node_id);
#endif

  // Replicator must exist BEFORE Membership (Membership holds a pointer to it
  // to drive membership-change re-replication). The transport's connection
  // pool is populated here from the static peer set so ReplicateOut on the
  // very first Insert routes immediately (no warm-up RPC). EnsurePeerClient
  // delegates to transport_->Connect.
  replicator_ = std::make_unique<Replicator>(
      cfg_.node_id, router_.get(), store_.get(), transport_.get(),
      metrics_.get());
  for (const auto& p : cfg_.seed_peers) {
    if (p.node_id == cfg_.node_id) continue;
    replicator_->EnsurePeerClient(p.node_id, p.address);
  }

  // Membership constructed with the static peer set; Start() below kicks off
  // the heartbeat thread + failure-detector loop. Membership holds Router* +
  // Replicator* and calls into both on membership-change events.
  MembershipConfig mcfg{};
  membership_ = std::make_unique<Membership>(
      mcfg,
      cfg_.node_id,
      cfg_.seed_peers,
      router_.get(),
      replicator_.get(),
      metrics_.get());

  // Evictor runs per-tier SIEVE scans and gossips evictions via
  // EvictBroadcast. Takes TieredStore + Membership for the alive-peer address
  // list. Watermark + scan-interval default; tune via EvictionConfig if
  // pressure shape changes.
  EvictionConfig ecfg{};
  ecfg.high_watermark_pct = cfg_.eviction_high_watermark_pct;
  ecfg.low_watermark_pct = cfg_.eviction_low_watermark_pct;
  evictor_ = std::make_unique<Evictor>(
      ecfg, store_.get(), membership_.get(), cfg_.node_id, metrics_.get());
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
                                const std::string& request_id,
                                const std::string& requesting_node) {
  LatencyTimer timer;
  LookupResult result;
  result.entries.reserve(ids.size());
  for (const auto& id : ids) {
    LookupResult::Entry e;
    e.id = id;

    // Try the local store first. A LocalHit short-circuits even if this node
    // isn't the primary — having the bytes locally beats a network
    // round-trip. Get returns OWNED bytes because the SSD tier can't lend
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

      // Server-side read-repair (FetchFromAny on local miss) is DISABLED.
      // Two reasons:
      //
      // 1. Cold-cache amplification. For a Lookup whose blocks miss on every
      //    node, each in-route block triggers FetchFromAny → Fetch RPC to
      //    each replica. With N blocks × R-1 peers × per-call overhead, a
      //    30-block cold lookup blew the per-primary-batching budget (30s vs
      //    1s) once router self-inclusion made we_are_in_route true.
      //
      // 2. Redundant with the client's transparent fetch. LetheClient already
      //    implements the "RemoteHit → transparent Fetch from source_node"
      //    path. Server-side read-repair only saved one round-trip on hits
      //    that landed on the wrong peer; in the cold case it adds unbounded
      //    fan-out.
      //
      // The right place for read-repair is the chunk-level bulk-pull path
      // (StreamBlocks), where the hit-rate justifies it and the budget
      // tolerates the extra hop. The route-aware miss handling below stays
      // intact — we still return RemoteHit so the client can route correctly.

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

  // The load-bearing hit-rate metric: a flat-zero
  // lethe_requests_total{result="hit"} surfaces a dead load path on the
  // dashboard. hit_count counts LocalHit + RemoteHit; miss_count the rest.
  if (metrics_) {
    metrics_->RecordLookup(result.hit_count, result.miss_count,
                           timer.elapsed());
  }
  // Structured trace: one greppable JSON line per Lookup (routing summary).
  // Per-block detail is deferred; this summary is enough now that read-repair
  // is off at the Lookup path.
  std::cerr << "{\"evt\":\"lookup\",\"node\":\"" << cfg_.node_id
            << "\",\"request_id\":\"" << request_id
            << "\",\"requesting_node\":\"" << requesting_node
            << "\",\"blocks\":" << ids.size()
            << ",\"hits\":" << result.hit_count
            << ",\"misses\":" << result.miss_count
            << ",\"latency_ms\":"
            << std::chrono::duration<double, std::milli>(timer.elapsed()).count()
            << "}\n";
  return result;
}

std::uint32_t LetheCache::Insert(std::vector<KvBlock> blocks,
                                 const std::string& /*request_id*/,
                                 const std::string& /*source_node*/,
                                 InsertOptions /*opts*/) {
  LatencyTimer timer;
  // Async replication + tier-aware insert: write locally first (using blk.tier
  // as the hint, defaulting to DRAM) → ACK → kick off background ReplicateOut.
  // opts.sync_replicate is accepted but unimplemented; replication is async by
  // default.
  //
  // "Accepted" semantics: counts NEWLY-INSERTED blocks. A Put against an
  // already-present BlockId is idempotent (content-addressed; same hash → same
  // bytes) and counts as 0, NOT 1 (test_repeated_insert_is_idempotent enforces
  // this). We detect newness by total-used-bytes delta across all tiers — Put
  // can land a new block in any tier per the fallthrough chain, so we sum the
  // deltas rather than watching one tier.
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
        // Fire-and-forget. ReplicateOut returns the peer_ids it queued the
        // push to; we don't wait. Failures show up in the Replicator's
        // overflow_drops / replicate_failures counters.
        replicator_->ReplicateOut(to_replicate);
      }
    }
  }
  if (metrics_) metrics_->RecordInsert(accepted, timer.elapsed());
  return accepted;
}

std::optional<LetheCache::LocalFetchResult> LetheCache::FetchLocal(
    const BlockId& id) {
  // Non-recursive: only consults the local TieredStore. No router,
  // no read-repair. This exists so the gRPC Fetch handler can ask
  // "do I have this block locally?" without recursing back into
  // Lookup's read-repair branch (which would issue Fetch RPCs to
  // peers whose Fetch handlers would then call Lookup again, etc.).
  // See cache.hpp:LocalFetchResult docstring.
  if (auto got = store_->Get(id); got.has_value()) {
    LocalFetchResult r;
    r.data = std::move(got->data);
    r.tier = got->tier_found;
    return r;
  }
  return std::nullopt;
}

void LetheCache::IngestStreamedBlock(BlockId id,
                                     std::vector<std::byte> payload,
                                     StreamPurpose /*purpose*/) {
  // Route every streamed block into DRAM. Purpose is recorded by the caller's
  // metric but doesn't change tier selection.
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
  // No membership → empty reply with epoch=0. The gRPC shim ships this as
  // HeartbeatResponse(cluster_epoch=0, alive_peers=[]).
  return HeartbeatReply{};
}

void LetheCache::OnEvictBroadcast(const std::vector<BlockId>& evicted,
                                  const std::string& source) {
  // Delegate to Evictor::OnPeerEviction so read-repair routing can later skip
  // peers that just evicted the same block. The skip-peer consumer in
  // FetchFromAny is deferred; the wire is in place and the data is tracked.
  if (evictor_) evictor_->OnPeerEviction(evicted, source);
}

std::uint64_t LetheCache::cluster_epoch() const noexcept {
  // Source of truth is Membership::Epoch(). When membership_ is null there is
  // no cluster, so the epoch is by definition 0.
  return membership_ ? membership_->Epoch() : 0;
}

}  // namespace lethe
