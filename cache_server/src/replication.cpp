// Lethe — replication + read-repair (W4, refactored W5-6 for KvTransport).
//
// W5-6 change: Replicator no longer owns gRPC channels directly. It now
// holds a KvTransport* and delegates ALL peer-to-peer block movement
// through the abstraction. This is the swap point for the W12 IB
// hardware transition: change the factory in main.cpp to construct an
// IbverbsTransport instead of GrpcStreamTransport and the Replicator
// transparently switches data paths. See
// docs/decisions/W5_rdma_fallback.md.
//
// What stayed the same:
//   * Async replication policy: fire-and-forget, bounded queue, 4 workers.
//   * Queue overflow → drop with overflow_drops bump; W8 re-replication
//     catches up.
//   * FetchFromAny semantics: parallel fan-out to replicas, first
//     non-empty wins, abandon laggards. 500ms per-call deadline (the
//     deadline now lives inside the transport implementation; the
//     wait-loop budget stays here).
//
// What changed:
//   * No more PeerClient class (it moved into GrpcStreamTransport).
//   * No more pool_mu_ in Replicator (channels live in the transport).
//   * Workers call transport_->Send and block on the returned future
//     (which is ready-by-return for GrpcStreamTransport's
//     synchronous-inside-call shape). The queue + N-workers limit
//     keeps N concurrent Sends in flight.
//   * FetchFromAny uses std::async per peer to call
//     transport_->Fetch in parallel — same fan-out shape as W4.
//
// Why this isn't a breaking change for W4 tests:
//   * Wire format is unchanged. GrpcStreamTransport invokes the same
//     Insert / Fetch RPCs the W4 Replicator did directly.
//   * Async-replication semantics (fire-and-forget, bounded queue) are
//     preserved. Workers still pop, Send, swallow the result.
//   * EnsurePeerClient / DropPeerClient still on Replicator's API;
//     LetheCache's ctor calls them per seed peer as before.

#include "lethe/replication.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include "lethe/kv_transport.hpp"
#include "lethe/routing.hpp"
#include "lethe/tiered_store.hpp"
#include "lethe/types.hpp"

namespace lethe {

namespace {

constexpr std::size_t kReplicationQueueMax = 1024;
constexpr int kReplicationWorkers = 4;
constexpr int kFetchTimeoutMs = 500;

}  // namespace

// ---------------------------------------------------------------------------
// Internal pool state (pimpl via TU-local registry — same pattern as W4)
// ---------------------------------------------------------------------------

namespace {

struct ReplicateTask {
  std::string peer_id;
  KvBlock block;  // owns its data
};

}  // namespace

struct ReplicatorPoolState {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<ReplicateTask> queue;
  std::vector<std::thread> workers;
  std::atomic<bool> running{true};
  std::atomic<std::uint64_t> overflow_drops{0};
  std::atomic<std::uint64_t> replicate_attempts{0};
  std::atomic<std::uint64_t> replicate_failures{0};
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
                       KvTransport* transport)
    : local_node_id_(std::move(local_node_id)),
      router_(router),
      store_(store),
      transport_(transport) {
  // store_ is stashed for W8's TriggerReReplication (scans the local
  // store for under-replicated blocks after a peer is declared dead).
  // The (void) cast satisfies clang's -Wunused-private-field without
  // the cross-compiler quirks of [[maybe_unused]] on a data member;
  // same pattern as Membership::router_ / replicator_ (commit
  // 8b5b7f5).
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

        raw->replicate_attempts.fetch_add(1, std::memory_order_relaxed);
        if (transport_ == nullptr) {
          raw->replicate_failures.fetch_add(1, std::memory_order_relaxed);
          continue;
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
        try {
          auto fut = transport_->Send(task.peer_id, task.block.id,
                                      StreamPurpose::ReplicationPush, data);
          ok = fut.get();
        } catch (...) {
          // Transport threw (shouldn't, but defensive). Treat as failure.
          ok = false;
        }
        if (!ok) {
          raw->replicate_failures.fetch_add(1, std::memory_order_relaxed);
          // Per CLAUDE.md rule 2: never block scheduling on cache
          // liveness. Drop on the floor; W8 re-replication heals.
        }
      }
    });
  }
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
    const std::vector<std::string>& /*lost_peers*/) {
  // W8: scan local store for blocks where the ring now shows fewer than
  // R replicas alive (i.e. the lost peers were on the replica list).
  // Re-push to the new R-1 successors via transport_->Send. For W4-W6
  // this is a no-op — failure detection (Membership) doesn't run yet.
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
