// Lethe — replication + read-repair (W4).
//
// PeerClient: per-peer gRPC channel + stub. Lazily opened on first
// use, cached for the process lifetime. grpc::Channel multiplexes
// concurrent RPCs internally (it's an HTTP/2 connection), so one
// channel per peer is enough — we don't pool more. If a peer goes
// down, the channel transitions to TRANSIENT_FAILURE and recovers
// automatically when the peer comes back; we don't tear it down on
// transient errors.
//
// Async replication policy (W0 decision, locked):
//   Insert returns to the caller after the LOCAL tier write. Pushing
//   to replicas happens on a background thread pool. A bounded queue
//   (max depth 1024 per the W4 prompt) keeps memory bounded under
//   bursty Insert load; queue overflow drops with a warning and W8's
//   re-replication catches up. This means Insert ACK latency is
//   independent of replica health, which is the whole point.
//
// Thread pool sizing: N=4 workers, mirroring the W1.4 client-side
// connector ThreadPoolExecutor. The bound is "small enough that
// thread-context overhead is negligible, large enough to overlap
// gRPC latency across the R-1=1 successor for a 3-node R=2 setup
// plus a couple of concurrent client Insert flows." Bump if W11
// chaos shows starvation.
//
// FetchFromAny (read-repair pull):
//   Synchronous to the caller (Lookup blocks waiting for a hit).
//   Fans out concurrent Fetch RPCs to all named replicas; first
//   non-empty response wins. Other in-flight requests are abandoned;
//   their results are discarded. Per-RPC timeout 500ms (read-repair
//   on a cache miss should not exceed a couple of network RTTs;
//   beyond that the caller is better off treating it as a Miss and
//   recomputing).

#include "lethe/replication.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <thread>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "lethe.grpc.pb.h"
#include "lethe.pb.h"
#include "lethe/routing.hpp"
#include "lethe/tiered_store.hpp"
#include "lethe/types.hpp"

namespace lethe {

namespace {

constexpr std::size_t kReplicationQueueMax = 1024;
constexpr int kReplicationWorkers = 4;
constexpr int kFetchTimeoutMs = 500;
constexpr int kInsertTimeoutMs = 2000;

::lethe::rpc::BlockId BlockIdToProto(const BlockId& src) {
  ::lethe::rpc::BlockId p;
  p.set_hash(src.hash.data(), src.hash.size());
  p.set_layer(src.layer);
  p.set_head_group(src.head_group);
  p.set_model_id(src.model_id);
  return p;
}

BlockId BlockIdFromProto(const ::lethe::rpc::BlockId& src) {
  BlockId out;
  const std::string& h = src.hash();
  const std::size_t n = std::min<std::size_t>(h.size(), out.hash.size());
  std::memcpy(out.hash.data(), h.data(), n);
  if (n < out.hash.size()) {
    std::memset(out.hash.data() + n, 0, out.hash.size() - n);
  }
  out.layer = src.layer();
  out.head_group = src.head_group();
  out.model_id = src.model_id();
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// PeerClient — thin gRPC channel + stub holder.
// ---------------------------------------------------------------------------

class PeerClient {
 public:
  PeerClient(std::string peer_id, std::string address)
      : peer_id_(std::move(peer_id)), address_(std::move(address)) {
    // grpc::CreateChannel returns a shared_ptr<Channel>. Insecure for
    // W4 (single-rack loopback or LAN); TLS lands when this thing
    // leaves the lab. Default keepalive is fine for loopback.
    channel_ = grpc::CreateChannel(address_, grpc::InsecureChannelCredentials());
    stub_ = ::lethe::rpc::LetheCache::NewStub(channel_);
  }

  const std::string& peer_id() const { return peer_id_; }
  const std::string& address() const { return address_; }
  ::lethe::rpc::LetheCache::Stub* stub() { return stub_.get(); }

 private:
  std::string peer_id_;
  std::string address_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<::lethe::rpc::LetheCache::Stub> stub_;
};

// ---------------------------------------------------------------------------
// Internal queue + worker pool (declared here so the header stays
// small; pimpl-shaped state lives inside the Replicator implementation).
// ---------------------------------------------------------------------------

namespace {

struct ReplicateTask {
  std::string peer_id;
  KvBlock block;  // owns its data
  std::string request_id;
};

}  // namespace

// We hang the pool state off the Replicator class via a pimpl-style
// internal struct. Header keeps the Replicator API intact.
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

// Translation-unit-local registry mapping Replicator* → pool state.
// Avoids changing the header's member layout (which would force
// a header-level type leak of grpc / condition_variable). This is
// the same trick used by Metrics::Impl in metrics.cpp.
//
// Lookups happen rarely (once per ctor / dtor / ReplicateOut /
// FetchFromAny call) and the Replicator population is tiny (one
// per server process), so a flat vector under a mutex is fine.
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
                       TieredStore* store)
    : local_node_id_(std::move(local_node_id)),
      router_(router),
      store_(store) {
  auto state = std::make_unique<ReplicatorPoolState>();
  // Spawn workers AFTER the unique_ptr is in the registry so the
  // worker lambda can look up its state safely. Capture `this` by
  // value (it's a pointer, cheap and stable for the worker's
  // lifetime — the Replicator outlives the workers, which we join
  // in the dtor).
  ReplicatorPoolState* raw = state.get();
  {
    auto& reg = pool_registry();
    std::lock_guard<std::mutex> g(reg.mu);
    reg.pools.emplace(this, std::move(state));
  }
  for (int i = 0; i < kReplicationWorkers; ++i) {
    raw->workers.emplace_back([this, raw]() {
      // Worker body: pop tasks, look up PeerClient, issue Insert RPC.
      // No header-level WorkerLoop_ method — kept inline so the
      // .hpp surface stays minimal. Captures `this` for access to
      // pool_mu_ + peer_clients_ + local_node_id_.
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

        PeerClient* peer = nullptr;
        {
          std::lock_guard<std::mutex> g(pool_mu_);
          auto it = peer_clients_.find(task.peer_id);
          if (it != peer_clients_.end()) peer = it->second.get();
        }
        raw->replicate_attempts.fetch_add(1, std::memory_order_relaxed);
        if (peer == nullptr) {
          raw->replicate_failures.fetch_add(1, std::memory_order_relaxed);
          continue;
        }

        ::lethe::rpc::InsertRequest req;
        req.set_request_id(task.request_id);
        req.set_source_node(local_node_id_);
        auto* b = req.add_blocks();
        *b->mutable_id() = BlockIdToProto(task.block.id);
        b->set_kv_data(task.block.data.data(), task.block.data.size());
        b->set_tier_hint(static_cast<std::uint32_t>(task.block.tier));

        ::lethe::rpc::InsertResponse resp;
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(kInsertTimeoutMs));
        auto status = peer->stub()->Insert(&ctx, req, &resp);
        if (!status.ok()) {
          raw->replicate_failures.fetch_add(1, std::memory_order_relaxed);
          // CLAUDE.md rule: never block scheduling on cache liveness.
          // Drop on the floor; W8 re-replication catches up after a
          // peer's declared-dead transition.
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
  if (router_ == nullptr) return submitted;
  const auto route = router_->Route(block.id);
  // Replicate to the (R-1) successor peers on the ring. For R=2,
  // that's exactly one peer. We skip the local node defensively —
  // the ring shouldn't return us as a replica when we're primary
  // (Route() picks distinct peers), but if this server is acting
  // as a non-primary writer (e.g. the client routed to us by
  // mistake during a transient ring drift), we still want to push
  // to the correct successors and avoid a self-loop.
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
      // Bounded queue, drop on overflow. The W4 prompt accepts this:
      // synchronous backpressure on Insert would defeat the
      // async-replication policy. Drops are logged via the counter;
      // W8's re-replication eventually heals.
      state->overflow_drops.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    state->queue.push_back(ReplicateTask{
        peer_id,
        block,  // copies; necessary because the caller's reference is
                // local to Insert and the worker runs later.
        std::string{},
    });
    state->cv.notify_one();
  }
  return submitted;
}

std::optional<KvBlock> Replicator::FetchFromAny(
    const BlockId& id,
    const std::vector<std::string>& peers) {
  if (peers.empty()) return std::nullopt;

  // Snapshot the PeerClient pointers under the pool mutex. We don't
  // hold the mutex across the network calls.
  std::vector<PeerClient*> clients;
  clients.reserve(peers.size());
  {
    std::lock_guard<std::mutex> g(pool_mu_);
    for (const auto& peer_id : peers) {
      auto it = peer_clients_.find(peer_id);
      if (it != peer_clients_.end()) clients.push_back(it->second.get());
    }
  }
  if (clients.empty()) return std::nullopt;

  // Fan out parallel Fetch RPCs via std::async. The first responder
  // wins; outstanding RPCs are abandoned (their contexts go out of
  // scope and grpc cancels). Per-call deadline keeps the worst case
  // bounded.
  std::vector<std::future<std::optional<KvBlock>>> futs;
  futs.reserve(clients.size());
  auto pb_id = BlockIdToProto(id);
  for (auto* c : clients) {
    futs.push_back(std::async(std::launch::async,
                              [c, pb_id]() -> std::optional<KvBlock> {
      ::lethe::rpc::FetchRequest req;
      *req.mutable_id() = pb_id;
      req.set_requesting_node("lethe-readrepair");
      ::lethe::rpc::FetchResponse resp;
      grpc::ClientContext ctx;
      ctx.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(kFetchTimeoutMs));
      auto status = c->stub()->Fetch(&ctx, req, &resp);
      if (!status.ok() || !resp.found()) return std::nullopt;
      KvBlock blk;
      blk.id = BlockIdFromProto(resp.id());
      const std::string& d = resp.kv_data();
      blk.data.resize(d.size());
      std::memcpy(blk.data.data(), d.data(), d.size());
      blk.tier = Tier::DRAM;
      return blk;
    }));
  }

  // First non-empty wins. We poll the futures with short waits so
  // we don't starve waiting on a slow peer when a fast one already
  // answered.
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
    // All futures either consumed or pending; if none ready in this
    // sweep, sleep briefly and try again. Cheaper than a hot spin.
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
  // W8: scan local store for blocks where the ring now shows fewer
  // than R replicas alive (i.e. the lost peers were on the replica
  // list). Re-push to the new R-1 successors. For W4 this is a
  // no-op — failure detection (Membership) doesn't run yet.
}

void Replicator::EnsurePeerClient(const std::string& peer_id,
                                  const std::string& address) {
  std::lock_guard<std::mutex> g(pool_mu_);
  auto it = peer_clients_.find(peer_id);
  if (it != peer_clients_.end()) return;  // already pooled
  peer_clients_.emplace(
      peer_id, std::make_unique<PeerClient>(peer_id, address));
}

void Replicator::DropPeerClient(const std::string& peer_id) {
  std::lock_guard<std::mutex> g(pool_mu_);
  peer_clients_.erase(peer_id);
}

}  // namespace lethe
