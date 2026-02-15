// Replication + read-repair.
//
// Replicator holds a KvTransport* and delegates ALL peer-to-peer block
// movement through the abstraction — this is the swap point for the IB
// hardware transition: change the factory in main.cpp to construct an
// IbverbsTransport instead of GrpcStreamTransport and the data path switches
// transparently.
//
//   * Async replication policy: fire-and-forget, bounded queue, 4 workers.
//   * Queue overflow → drop with overflow_drops bump; re-replication catches
//     up later.
//   * FetchFromAny: parallel fan-out (std::async per peer) to replicas, first
//     non-empty wins, abandon laggards. The 500ms per-call deadline lives in
//     the transport; the wait-loop budget stays here.
//
// Wire format is the same Insert / Fetch RPCs regardless of transport, so the
// GrpcStreamTransport path resolves Send/Fetch futures synchronously inside
// the call while an ibverbs transport resolves them on its CQ-polling thread;
// the queue + N-workers limit keeps N concurrent Sends in flight either way.

#include "lethe/replication.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include "lethe/kv_transport.hpp"
#include "lethe/metrics.hpp"
#include "lethe/routing.hpp"
#include "lethe/tiered_store.hpp"
#include "lethe/types.hpp"

namespace lethe {

namespace {

constexpr std::size_t kReplicationQueueMax = 1024;
constexpr int kReplicationWorkers = 4;
constexpr int kFetchTimeoutMs = 500;

// Re-replication is dispatched in bounded batches so a single membership
// change doesn't enqueue an unbounded burst. The sweep thread drains
// successive batches every kReReplicationSweepInterval until the round's whole
// candidate set is covered. (An earlier version ran one capped pass with
// nothing to re-trigger, so working sets > kReReplicationBatch stayed
// under-replicated indefinitely.)
constexpr std::size_t kReReplicationBatch = 256;
constexpr auto kReReplicationSweepInterval = std::chrono::milliseconds(250);

// Diagnostic gate; set LETHE_DEBUG_REREP=1 in the environment to get
// stderr prints from TriggerReReplication and the worker loop.
// Disabled in normal builds so production logs aren't polluted.
bool DebugRerepEnabled() {
  static const bool en = []{ const char* v = std::getenv("LETHE_DEBUG_REREP");
                             return v != nullptr && *v != '0'; }();
  return en;
}

}  // namespace

// ---------------------------------------------------------------------------
// Internal pool state (pimpl via TU-local registry)
// ---------------------------------------------------------------------------

namespace {

struct ReplicateTask {
  std::string peer_id;
  KvBlock block;  // owns its data
};

}  // namespace

struct ReplicatorPoolState {
  std::mutex mu;  // guards the worker queue (queue/cv)
  std::condition_variable cv;
  std::deque<ReplicateTask> queue;
  std::vector<std::thread> workers;
  std::atomic<bool> running{true};
  std::atomic<std::uint64_t> overflow_drops{0};
  std::atomic<std::uint64_t> replicate_attempts{0};
  std::atomic<std::uint64_t> replicate_failures{0};

  // Periodic re-replication sweep. `rerep_mu` guards the round state. A round
  // is the full set of in-route blocks this node holds at the time of a
  // membership change; the sweep thread drains it in kReReplicationBatch-sized,
  // queue-back-pressured chunks (advancing the cursor only for blocks actually
  // enqueued) until the cursor reaches the end. Lock order is ALWAYS rerep_mu
  // -> mu (queue); workers take only mu.
  std::mutex rerep_mu;
  std::vector<BlockId> rerep_round;
  std::size_t rerep_cursor = 0;
  bool rerep_active = false;
  std::chrono::steady_clock::time_point rerep_start;
  std::thread sweep_thread;
};

namespace {

struct PoolRegistry {
  std::mutex mu;
  std::unordered_map<const Replicator*, std::unique_ptr<ReplicatorPoolState>>
      pools;
};

PoolRegistry& pool_registry() {
  static PoolRegistry r;
  return r;
}

ReplicatorPoolState* pool_for(const Replicator* r) {
  auto& reg = pool_registry();
  std::lock_guard<std::mutex> g(reg.mu);
  auto it = reg.pools.find(r);
  return it == reg.pools.end() ? nullptr : it->second.get();
}

}  // namespace

// ---------------------------------------------------------------------------
// Replicator
// ---------------------------------------------------------------------------

Replicator::Replicator(std::string local_node_id,
                       Router* router,
                       TieredStore* store,
                       KvTransport* transport,
                       Metrics* metrics)
    : local_node_id_(std::move(local_node_id)),
      router_(router),
      store_(store),
      transport_(transport),
      metrics_(metrics) {
  // store_ is stashed for TriggerReReplication (scans the local store for
  // under-replicated blocks after a peer is declared dead). The (void) cast
  // satisfies clang's -Wunused-private-field without the cross-compiler quirks
  // of [[maybe_unused]] on a data member; same pattern as Membership.
  (void)store_;
  auto state = std::make_unique<ReplicatorPoolState>();
  ReplicatorPoolState* raw = state.get();
  {
    auto& reg = pool_registry();
    std::lock_guard<std::mutex> g(reg.mu);
    reg.pools.emplace(this, std::move(state));
  }
  for (int i = 0; i < kReplicationWorkers; ++i) {
    raw->workers.emplace_back([this, raw]() {
      for (;;) {
        ReplicateTask task;
        {
          std::unique_lock<std::mutex> lock(raw->mu);
          raw->cv.wait(lock, [raw]() {
            return !raw->queue.empty() ||
                   !raw->running.load(std::memory_order_acquire);
          });
          if (raw->queue.empty() &&
              !raw->running.load(std::memory_order_acquire)) {
            return;
          }
          task = std::move(raw->queue.front());
          raw->queue.pop_front();
        }

        const std::uint64_t attempt_n =
            raw->replicate_attempts.fetch_add(1, std::memory_order_relaxed);
        if (transport_ == nullptr) {
          raw->replicate_failures.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        if (DebugRerepEnabled() && (attempt_n < 10 || attempt_n % 50 == 0)) {
          std::fprintf(stderr, "[lethe %s] worker send #%llu → %s\n",
                       local_node_id_.c_str(),
                       static_cast<unsigned long long>(attempt_n),
                       task.peer_id.c_str());
        }
        // Dispatch via transport. Block on the future — the worker IS the
        // concurrency unit here (queue + N workers = N in-flight Sends).
        // GrpcStreamTransport resolves the future synchronously inside
        // the call; an ibverbs transport would resolve when the
        // completion arrives on its CQ-polling thread. Either way the
        // worker is correctly back-pressured.
        std::span<const std::byte> data(task.block.data.data(),
                                        task.block.data.size());
        bool ok = false;
        const auto send_t0 = std::chrono::steady_clock::now();
        try {
          auto fut = transport_->Send(task.peer_id, task.block.id,
                                      StreamPurpose::ReplicationPush, data);
          ok = fut.get();
        } catch (...) {
          // Transport threw (shouldn't, but defensive). Treat as failure.
          ok = false;
        }
        if (ok && metrics_ != nullptr) {
          // Record the bytes + latency of this block transfer.
          metrics_->RecordStreamBytes(
              task.block.data.size(),
              std::chrono::steady_clock::now() - send_t0);
        }
        if (!ok) {
          const std::uint64_t fail_n =
              raw->replicate_failures.fetch_add(1, std::memory_order_relaxed);
          if (DebugRerepEnabled() && (fail_n < 10 || fail_n % 50 == 0)) {
            std::fprintf(stderr, "[lethe %s] worker send FAIL #%llu → %s\n",
                         local_node_id_.c_str(),
                         static_cast<unsigned long long>(fail_n),
                         task.peer_id.c_str());
          }
          // Never block scheduling on cache liveness. Drop on the floor; the
          // periodic sweep re-pushes it on the next round / membership change.
        }
      }
    });
  }

  // Re-replication sweep thread. Drives DrainReReplication on a fixed cadence
  // so a re-replication round larger than one batch fully drains across
  // successive ticks. A dedicated low-frequency Replicator-owned timer keeps
  // re-replication self-contained (no Evictor->Replicator dependency), and its
  // lifetime is trivially correct (joined in ~Replicator, while
  // store_/router_/transport_ — all declared before replicator_ in cache.hpp —
  // are still alive). It is a no-op (one mutex acquire + bool check) whenever
  // no round is active.
  raw->sweep_thread = std::thread([this, raw]() {
    using namespace std::chrono_literals;
    while (raw->running.load(std::memory_order_acquire)) {
      // Sleep in short slices so Shutdown isn't blocked for a full
      // interval (mirrors the Evictor's per-tier sleep pattern).
      const auto wake = std::chrono::steady_clock::now() +
                        kReReplicationSweepInterval;
      while (std::chrono::steady_clock::now() < wake &&
             raw->running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::min<std::chrono::milliseconds>(
            50ms, std::chrono::duration_cast<std::chrono::milliseconds>(
                      wake - std::chrono::steady_clock::now())));
      }
      if (!raw->running.load(std::memory_order_acquire)) break;
      DrainReReplication();
    }
  });
}

Replicator::~Replicator() {
  auto* state = pool_for(this);
  if (state == nullptr) return;
  {
    std::lock_guard<std::mutex> g(state->mu);
    state->running.store(false, std::memory_order_release);
  }
  state->cv.notify_all();
  for (auto& w : state->workers) {
    if (w.joinable()) w.join();
  }
  // Stop the re-replication sweep before the pool is torn down. `running` is
  // already false above; the sweep checks it each 50ms slice.
  if (state->sweep_thread.joinable()) state->sweep_thread.join();
  auto& reg = pool_registry();
  std::lock_guard<std::mutex> g(reg.mu);
  reg.pools.erase(this);
}

std::vector<std::string> Replicator::ReplicateOut(const KvBlock& block) {
  std::vector<std::string> submitted;
  if (router_ == nullptr || transport_ == nullptr) return submitted;
  const auto route = router_->Route(block.id);
  // Replicate to the (R-1) successor peers. For R=2, that's exactly one
  // peer. Skip self defensively — the ring shouldn't return us as a
  // replica when we're primary, but if the client routed an Insert to
  // a non-primary by mistake, we still want to push to the correct
  // successors and avoid a self-loop.
  for (const auto& peer_id : route.replicas) {
    if (peer_id == local_node_id_) continue;
    submitted.push_back(peer_id);
  }
  if (submitted.empty()) return submitted;

  auto* state = pool_for(this);
  if (state == nullptr) return {};

  for (const auto& peer_id : submitted) {
    std::lock_guard<std::mutex> lock(state->mu);
    if (state->queue.size() >= kReplicationQueueMax) {
      state->overflow_drops.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    state->queue.push_back(ReplicateTask{peer_id, block});
    state->cv.notify_one();
  }
  return submitted;
}

std::optional<KvBlock> Replicator::FetchFromAny(
    const BlockId& id,
    const std::vector<std::string>& peers) {
  if (peers.empty() || transport_ == nullptr) return std::nullopt;

  // Fan out: one std::async per peer calling transport_->Fetch. The
  // transport's Fetch returns a future, but for the GrpcStreamTransport
  // case it's a ready future (sync-inside-call) — so the parallelism
  // comes from our std::async layer, not the transport. An ibverbs
  // transport would return a truly-async future, and our wait loop
  // below handles both shapes identically.
  std::vector<std::future<std::optional<KvBlock>>> futs;
  futs.reserve(peers.size());
  for (const auto& peer_id : peers) {
    KvTransport* t = transport_;
    BlockId bid_copy = id;
    std::string pid_copy = peer_id;
    futs.push_back(std::async(std::launch::async,
                              [t, pid_copy = std::move(pid_copy),
                               bid_copy]() -> std::optional<KvBlock> {
      try {
        return t->Fetch(pid_copy, bid_copy).get();
      } catch (...) {
        return std::nullopt;
      }
    }));
  }

  // First non-empty wins. Poll futures with short waits so we don't
  // starve waiting on a slow peer when a fast one already answered.
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kFetchTimeoutMs * 2);
  while (std::chrono::steady_clock::now() < deadline) {
    bool any_ready = false;
    for (auto& f : futs) {
      if (!f.valid()) continue;
      if (f.wait_for(std::chrono::milliseconds(5)) ==
          std::future_status::ready) {
        any_ready = true;
        auto got = f.get();
        if (got.has_value()) return got;
      }
    }
    if (!any_ready) {
      bool all_done = true;
      for (auto& f : futs) {
        if (f.valid()) { all_done = false; break; }
      }
      if (all_done) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  return std::nullopt;
}

void Replicator::TriggerReReplication(
    const std::vector<std::string>& lost_peers) {
  // Called by Membership::OnMembershipChange after a peer is declared dead
  // (and after Router::SetPeers has installed the new ring). We build the FULL
  // candidate set — every block this node holds and is in-route for under the
  // new ring — and hand it to the sweep as a "re-replication round."
  // DrainReReplication then dispatches it in bounded, queue-back-pressured
  // batches across successive sweep ticks until the cursor reaches the end.
  //
  // Idempotent: the receiver's Insert dedups on BlockId, so re-pushing a block
  // a peer already has is a no-op — safe to over-include (we don't know which
  // blocks a surviving peer is missing without an ACK channel; adding one is a
  // deferred optimization).
  if (router_ == nullptr || transport_ == nullptr || store_ == nullptr) {
    if (DebugRerepEnabled()) {
      std::fprintf(stderr, "[lethe %s] TriggerReReplication: nullptr dep; skipping\n",
                   local_node_id_.c_str());
    }
    return;
  }
  if (lost_peers.empty()) {
    if (DebugRerepEnabled()) {
      std::fprintf(stderr, "[lethe %s] TriggerReReplication: lost_peers empty\n",
                   local_node_id_.c_str());
    }
    return;
  }

  // Scan ALL tiers (no cap — covering the whole working set is the point)
  // and collect the IDs of blocks we hold and are in-route for. Snapshot
  // is shared-locked per BlockStore; we only read IDs here, the byte copy
  // happens lazily in DrainReReplication via TieredStore::Get.
  std::vector<BlockId> candidates;
  for (Tier t : {Tier::HBM, Tier::DRAM, Tier::SSD}) {
    for (auto& meta : store_->Snapshot(t)) {
      const auto route = router_->Route(meta.id);
      bool in_route = (route.primary == local_node_id_);
      if (!in_route) {
        for (const auto& r : route.replicas) {
          if (r == local_node_id_) { in_route = true; break; }
        }
      }
      if (in_route) candidates.push_back(meta.id);
    }
  }

  auto* state = pool_for(this);
  if (state == nullptr) return;
  std::size_t round_size = 0;
  {
    std::lock_guard<std::mutex> g(state->rerep_mu);
    state->rerep_round = std::move(candidates);
    state->rerep_cursor = 0;
    state->rerep_active = !state->rerep_round.empty();
    state->rerep_start = std::chrono::steady_clock::now();
    round_size = state->rerep_round.size();
  }
  if (DebugRerepEnabled()) {
    std::fprintf(stderr, "[lethe %s] TriggerReReplication: round=%zu (lost=%zu)\n",
                 local_node_id_.c_str(), round_size, lost_peers.size());
  }
  // Live deficit gauge: the count still pending, zeroed by DrainReReplication
  // on completion.
  if (metrics_ != nullptr) {
    metrics_->RecordUnderReplicated(round_size);
  }
  // Kick the first batch inline so recovery doesn't wait a sweep tick.
  // (rerep_mu is released above, so this re-acquire is safe.)
  DrainReReplication();
}

void Replicator::DrainReReplication() {
  auto* state = pool_for(this);
  if (state == nullptr || router_ == nullptr || transport_ == nullptr ||
      store_ == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> g(state->rerep_mu);
  if (!state->rerep_active) return;  // no-op when idle (the common case)

  std::size_t dispatched_this_tick = 0;
  while (dispatched_this_tick < kReReplicationBatch &&
         state->rerep_cursor < state->rerep_round.size()) {
    const BlockId id = state->rerep_round[state->rerep_cursor];
    const auto route = router_->Route(id);

    // FULL target set = every route member except self: a surviving replica
    // may be the only holder of a block whose new primary lacks it, so
    // pushing only to replicas would never converge the primary.
    std::vector<std::string> targets;
    if (!route.primary.empty() && route.primary != local_node_id_) {
      targets.push_back(route.primary);
    }
    for (const auto& r : route.replicas) {
      if (r != local_node_id_) targets.push_back(r);
    }
    if (targets.empty()) {  // nothing to push for this block; it's covered
      ++state->rerep_cursor;
      continue;
    }

    // Back-pressure: advance the cursor ONLY for blocks we can fully
    // enqueue. If the worker queue lacks room for all of this block's
    // pushes, stop this tick and retry from the SAME cursor next tick —
    // this throttles the sweep to the worker drain rate and guarantees no
    // overflow-drop silently skips a block.
    {
      std::lock_guard<std::mutex> qg(state->mu);
      if (state->queue.size() + targets.size() > kReplicationQueueMax) break;
    }

    auto got = store_->Get(id);
    if (!got.has_value()) {  // evicted since the round was built; skip
      ++state->rerep_cursor;
      continue;
    }
    KvBlock blk;
    blk.id = id;
    blk.data = std::move(got->data);
    blk.tier = got->tier_found;
    {
      std::lock_guard<std::mutex> qg(state->mu);
      for (const auto& peer_id : targets) {
        state->queue.push_back(ReplicateTask{peer_id, blk});
        state->cv.notify_one();
      }
    }
    ++state->rerep_cursor;
    ++dispatched_this_tick;
  }

  const std::size_t remaining =
      state->rerep_round.size() - state->rerep_cursor;
  if (state->rerep_cursor >= state->rerep_round.size()) {
    // Round complete — every candidate dispatched. Zero the deficit gauge and
    // record the dispatch-completion time: this feeds
    // lethe_failover_recovery_seconds, measuring detection→full-dispatch, the
    // locally-observable recovery signal — ACK-confirmed R=2 lags by the queue
    // drain, which the chaos suite measures behaviorally.
    state->rerep_active = false;
    if (metrics_ != nullptr) {
      metrics_->RecordUnderReplicated(0);
      metrics_->RecordFailoverRecovery(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - state->rerep_start));
    }
    if (DebugRerepEnabled()) {
      std::fprintf(stderr, "[lethe %s] DrainReReplication: round complete\n",
                   local_node_id_.c_str());
    }
  } else if (metrics_ != nullptr) {
    metrics_->RecordUnderReplicated(remaining);
  }
}

void Replicator::EnsurePeerClient(const std::string& peer_id,
                                  const std::string& address) {
  if (transport_ == nullptr) return;
  transport_->Connect(peer_id, address);
}

void Replicator::DropPeerClient(const std::string& peer_id) {
  if (transport_ == nullptr) return;
  transport_->Disconnect(peer_id);
}

}  // namespace lethe
