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
// Lifetime contract (cache.hpp:81-86): LookupResult::Entry::local_data
// is a borrowed span into TieredStore's BlockStore. The gRPC service
// shim in main.cpp serializes it into the wire response immediately;
// no mutating cache call can happen between Lookup returning and the
// shim's serialization on the same thread because the shim is sync.

#include "lethe/cache.hpp"

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

  // W3: Router built from the static peer set. The static set always
  // includes the local node (main.cpp injects it before constructing
  // CacheConfig). Empty seed_peers → single-node mode; Router has no
  // peers and Lookup falls back to LocalHit/Miss only.
  router_ = std::make_unique<Router>(
      cfg_.node_id,
      cfg_.virtual_nodes_per_peer,
      cfg_.replication_factor);
  if (!cfg_.seed_peers.empty()) {
    std::vector<std::string> peer_ids;
    peer_ids.reserve(cfg_.seed_peers.size());
    for (const auto& p : cfg_.seed_peers) peer_ids.push_back(p.node_id);
    router_->SetPeers(std::move(peer_ids));
  }

  // W3: Membership constructed with the static peer set. W3-W4 treats
  // OnHeartbeat as a peer-list view only (no failure detection, no
  // epoch advancing). W8 adds the heartbeat loop. Pass router_ and
  // (future) replicator_ pointers for W8's membership-change driving.
  MembershipConfig mcfg{};
  membership_ = std::make_unique<Membership>(
      mcfg,
      cfg_.node_id,
      cfg_.seed_peers,
      router_.get(),
      /*replicator=*/nullptr);  // W4 wires this when Replicator is built.

  // Replicator: W4 (next commits) constructs and wires it in.
  // Evictor/transport/metrics: still deferred to their weeks.
}

LetheCache::~LetheCache() {
  Shutdown();
}

void LetheCache::Start() {
  running_.store(true, std::memory_order_relaxed);
  // W4+: evictor_->Start(); membership_->Start(); transport_->Start();
}

void LetheCache::Shutdown() {
  running_.store(false, std::memory_order_relaxed);
  // W4+: tear down workers. Subsystem dtors run when the unique_ptrs
  // are released by the destructor; for W1 they are all null.
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
    // bytes locally is better than a network round-trip.
    if (auto got = store_->Get(id); got.has_value()) {
      e.where = LookupResult::Entry::Where::LocalHit;
      e.tier = got->tier_found;
      e.local_data = got->data;
      ++result.hit_count;
      result.entries.push_back(e);
      continue;
    }

    // Local miss. Consult the Router. If we are the primary, the W4
    // read-repair branch will try to fetch from a replica before
    // accepting a miss; that lands in the next commit. If we are NOT
    // the primary, report RemoteHit with the primary's node_id and
    // let the client batch-fetch from the source.
    if (router_) {
      const auto route = router_->Route(id);
      if (!route.primary.empty() && route.primary != cfg_.node_id) {
        e.where = LookupResult::Entry::Where::RemoteHit;
        e.remote_node = route.primary;
        ++result.hit_count;  // counts as "located"; bytes come via Fetch
        result.entries.push_back(e);
        continue;
      }
      // route.primary == cfg_.node_id: we ARE primary but missed.
      // W4 will fall through to read-repair via FetchFromAny here.
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
  // W1: no replication (replicator_ is null). The opts.sync_replicate
  // flag is accepted but has no effect; W4 lights up real semantics.
  std::uint32_t accepted = 0;
  for (auto& blk : blocks) {
    // BlockStore::Put returns false on capacity overflow; we treat
    // that as a rejection (Insert returns the accepted count).
    const std::size_t before = store_->used_bytes(Tier::DRAM);
    store_->Put(std::move(blk), Tier::DRAM);
    const std::size_t after = store_->used_bytes(Tier::DRAM);
    if (after > before) ++accepted;
    else {
      // Either the block was already present (idempotent Put) or the
      // store was full. Both cases count as "did not newly accept";
      // a future audit may want to distinguish these — for now the
      // count semantics match the gRPC accepted_count proto field.
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

void LetheCache::OnEvictBroadcast(const std::vector<BlockId>& /*evicted*/,
                                  const std::string& /*source*/) {
  // W1: no evictor → drop the broadcast on the floor. W8 wires this
  // into Evictor::OnPeerEviction so we don't read-repair from peers
  // that just evicted the same block.
}

std::uint64_t LetheCache::cluster_epoch() const noexcept {
  // Source of truth is Membership::Epoch() (W0 contract). When
  // membership_ is null (W1, pre-cluster), there is no cluster — the
  // epoch is by definition 0.
  return membership_ ? membership_->Epoch() : 0;
}

}  // namespace lethe
