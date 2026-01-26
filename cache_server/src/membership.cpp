// Lethe — membership (W3-W8).
//
// W3-W4 surface (per-peer last_seen_epoch through HeartbeatReply,
// static seed list) carried forward intact. W8 adds:
//
//   * Heartbeat thread: every cfg.heartbeat_interval (default 200ms),
//     fan out Heartbeat RPCs to every known peer (alive AND not-self).
//     Parallel via std::async per peer; per-RPC deadline equal to the
//     heartbeat interval itself.
//
//   * Failure detection: EvaluateSuspicions runs each tick. Peers
//     whose last_seen is older than cfg.suspect_after (1s) become
//     suspected; older than cfg.dead_after (3s) get removed from the
//     alive set, the cluster epoch bumps, and OnMembershipChange
//     fires.
//
//   * OnMembershipChange: bumps epoch, calls Router::SetPeers with
//     the new alive list, calls Replicator::TriggerReReplication
//     with the just-died peer list so re-replication can scope the
//     work to blocks whose replicas just disappeared.
//
//   * Peer resurrection: a dead peer that sends a successful
//     heartbeat (or replies to ours) transitions back to alive at the
//     current epoch. Same OnMembershipChange path; the lost_peers
//     vector is empty in that direction.
//
// Threading: ONE std::thread (member of the class). Per-tick RPC
// parallelism uses std::async (one thread per active peer for the
// duration of the tick). The W0 design explicitly separates the
// heartbeat thread from the gRPC server's worker pool — heartbeat
// latency must not be coupled to RPC service load.
//
// gRPC client stubs live in a TU-local registry keyed by Membership*
// (same pImpl-via-registry pattern as Replicator / Evictor) to keep
// gRPC types out of membership.hpp.

#include "lethe/membership.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "lethe.grpc.pb.h"
#include "lethe.pb.h"
#include "lethe/replication.hpp"
#include "lethe/routing.hpp"

namespace lethe {

namespace {

constexpr int kHeartbeatDeadlineMs = 250;  // <= heartbeat_interval default

struct MembershipImpl {
  std::mutex stubs_mu;
  std::unordered_map<std::string,
                     std::pair<std::shared_ptr<grpc::Channel>,
                               std::unique_ptr<::lethe::rpc::LetheCache::Stub>>>
      stubs;

  // wake_cv lets Shutdown poke the heartbeat thread out of its
  // wait_for sleep instead of waiting up to heartbeat_interval.
  std::mutex wake_mu;
  std::condition_variable wake_cv;
};

struct Registry {
  std::mutex mu;
  std::unordered_map<const Membership*, std::unique_ptr<MembershipImpl>> impls;
};

Registry& registry() {
  static Registry r;
  return r;
}

MembershipImpl* impl_for(const Membership* m) {
  auto& reg = registry();
  std::lock_guard<std::mutex> g(reg.mu);
  auto it = reg.impls.find(m);
  return it == reg.impls.end() ? nullptr : it->second.get();
}

::lethe::rpc::LetheCache::Stub* stub_for_peer(MembershipImpl* impl,
                                              const std::string& address) {
  std::lock_guard<std::mutex> g(impl->stubs_mu);
  auto it = impl->stubs.find(address);
  if (it != impl->stubs.end()) return it->second.second.get();
  auto ch = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto s = ::lethe::rpc::LetheCache::NewStub(ch);
  auto* p = s.get();
  impl->stubs.emplace(address, std::make_pair(std::move(ch), std::move(s)));
  return p;
}

}  // namespace

Membership::Membership(MembershipConfig cfg,
                       std::string local_node_id,
                       std::vector<StaticPeer> seed_peers,
                       Router* router,
                       Replicator* replicator)
    : cfg_(cfg),
      local_node_id_(std::move(local_node_id)),
      router_(router),
      replicator_(replicator) {
  auto& reg = registry();
  std::lock_guard<std::mutex> g(reg.mu);
  reg.impls.emplace(this, std::make_unique<MembershipImpl>());

  auto now = std::chrono::steady_clock::now();
  for (auto& peer : seed_peers) {
    PeerInfo info;
    info.node_id = peer.node_id;
    info.address = peer.address;
    info.last_seen = now;
    info.last_seen_epoch = 0;
    info.alive = true;
    peers_.emplace(info.node_id, std::move(info));
  }
}

Membership::~Membership() {
  Shutdown();
  auto& reg = registry();
  std::lock_guard<std::mutex> g(reg.mu);
  reg.impls.erase(this);
}

void Membership::Start() {
  if (running_.exchange(true)) return;  // already started
  thread_ = std::thread([this]() { HeartbeatLoop(); });
}

void Membership::Shutdown() {
  if (!running_.exchange(false)) {
    // Was already stopped; still need to join if a thread was spawned
    // and we somehow got here through a double-shutdown path.
    if (thread_.joinable()) thread_.join();
    return;
  }
  // Wake the heartbeat thread immediately.
  if (auto* impl = impl_for(this)) {
    std::lock_guard<std::mutex> g(impl->wake_mu);
    impl->wake_cv.notify_all();
  }
  if (thread_.joinable()) thread_.join();
}

void Membership::HeartbeatLoop() {
  auto* impl = impl_for(this);
  if (impl == nullptr) return;

  while (running_.load(std::memory_order_acquire)) {
    // Send heartbeats to all known peers (alive AND dead — dead peers
    // can resurrect via successful round-trip).
    SendHeartbeatsToAllPeers();

    // Evaluate suspicions: any peers crossing thresholds?
    auto lost = EvaluateSuspicions();
    if (!lost.empty()) {
      OnMembershipChange(lost);
    }

    // Sleep until next tick or shutdown wake-up.
    std::unique_lock<std::mutex> lock(impl->wake_mu);
    impl->wake_cv.wait_for(lock, cfg_.heartbeat_interval, [this]() {
      return !running_.load(std::memory_order_acquire);
    });
  }
}

// ---------------------------------------------------------------------------
// Heartbeat fan-out (W8 client side)
// ---------------------------------------------------------------------------

void Membership::SendHeartbeatsToAllPeers() {
  auto* impl = impl_for(this);
  if (impl == nullptr) return;

  // Snapshot target list (id, address) under mu_, then release.
  std::vector<std::pair<std::string, std::string>> targets;
  {
    std::lock_guard<std::mutex> lock(mu_);
    targets.reserve(peers_.size());
    for (const auto& [id, info] : peers_) {
      if (id == local_node_id_) continue;
      if (info.address.empty()) continue;
      targets.emplace_back(id, info.address);
    }
  }
  if (targets.empty()) return;

  // Fan out in parallel via std::async. One thread per peer for the
  // duration of the RPC; bounded by N=cluster_size (3 today).
  const std::uint64_t epoch_at_send = epoch_.load(std::memory_order_relaxed);
  std::vector<std::future<std::pair<std::string, bool>>> futs;
  futs.reserve(targets.size());
  for (auto& [peer_id, address] : targets) {
    futs.emplace_back(std::async(std::launch::async,
                                 [this, impl, peer_id, address,
                                  epoch_at_send]() -> std::pair<std::string, bool> {
      auto* stub = stub_for_peer(impl, address);
      if (stub == nullptr) return {peer_id, false};
      ::lethe::rpc::HeartbeatRequest req;
      req.set_node_id(local_node_id_);
      req.set_epoch(epoch_at_send);
      // known_peers / load fields stay default; W8 doesn't gossip the
      // full peer view yet — that's W11 chaos work. The wire fields
      // exist (proto/lethe.proto) so future additions don't need a
      // proto bump.
      ::lethe::rpc::HeartbeatResponse resp;
      grpc::ClientContext ctx;
      ctx.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(kHeartbeatDeadlineMs));
      auto status = stub->Heartbeat(&ctx, req, &resp);
      return {peer_id, status.ok()};
    }));
  }

  // Wait for all replies with an outer deadline of one heartbeat
  // interval; futures past the deadline are abandoned (their threads
  // continue to completion but we don't trust the result).
  auto deadline = std::chrono::steady_clock::now() + cfg_.heartbeat_interval;
  std::vector<std::string> succeeded;
  for (auto& f : futs) {
    auto status = f.wait_until(deadline);
    if (status != std::future_status::ready) continue;
    auto [peer_id, ok] = f.get();
    if (ok) succeeded.push_back(std::move(peer_id));
  }

  // Update PeerInfo for successful replies in one lock acquisition.
  if (!succeeded.empty()) {
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> resurrected;  // peers that came back from dead
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (const auto& peer_id : succeeded) {
        auto it = peers_.find(peer_id);
        if (it == peers_.end()) continue;
        it->second.last_seen = now;
        it->second.last_seen_epoch = epoch_.load(std::memory_order_relaxed);
        it->second.suspected = false;
        if (!it->second.alive) {
          it->second.alive = true;
          resurrected.push_back(peer_id);
        }
      }
    }
    // Resurrection is a membership change. Re-replication kicks in to
    // rebalance now that we have more capacity again.
    if (!resurrected.empty()) {
      OnMembershipChange(/*lost_peers=*/{});
    }
  }
}

// ---------------------------------------------------------------------------
// Failure detection
// ---------------------------------------------------------------------------

std::vector<std::string> Membership::EvaluateSuspicions() {
  std::vector<std::string> newly_dead;
  std::lock_guard<std::mutex> lock(mu_);
  auto now = std::chrono::steady_clock::now();
  for (auto& [id, info] : peers_) {
    if (id == local_node_id_) continue;
    if (!info.alive) continue;
    const auto since = now - info.last_seen;
    if (since > cfg_.dead_after) {
      info.alive = false;
      info.suspected = true;
      newly_dead.push_back(id);
    } else if (since > cfg_.suspect_after) {
      info.suspected = true;
    } else {
      info.suspected = false;
    }
  }
  return newly_dead;
}

void Membership::OnMembershipChange(
    const std::vector<std::string>& lost_peers) {
  // Bump epoch FIRST so any concurrent caller sees the new value.
  epoch_.fetch_add(1, std::memory_order_release);

  // Compute the new alive peer-id list under mu_, then drop the lock
  // before calling into Router (its own lock) and Replicator (its
  // own queue + worker pool). No lock holds across subsystem calls.
  std::vector<std::string> alive_ids;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [id, info] : peers_) {
      if (info.alive) alive_ids.push_back(id);
    }
  }

  if (router_ != nullptr) {
    router_->SetPeers(alive_ids);
  }
  if (replicator_ != nullptr && !lost_peers.empty()) {
    replicator_->TriggerReReplication(lost_peers);
  }

  std::cout << "[lethe] membership change: epoch=" << epoch_.load()
            << " alive=" << alive_ids.size()
            << " lost=" << lost_peers.size() << "\n";
}

// ---------------------------------------------------------------------------
// Heartbeat RPC handler (server side)
// ---------------------------------------------------------------------------

HeartbeatReply Membership::OnHeartbeat(const std::string& peer_id,
                                       std::uint64_t /*peer_epoch*/) {
  // W8 still ignores the peer_epoch field for gossip-convergence
  // purposes (that's a W11 chaos-suite refinement); we use it only as
  // an opportunity to refresh the peer's last_seen.
  HeartbeatReply reply;
  bool resurrected = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto now = std::chrono::steady_clock::now();
    auto [it, inserted] = peers_.try_emplace(peer_id);
    if (inserted) {
      it->second.node_id = peer_id;
      // New peer joining IS a membership change; W8 fires it via the
      // resurrected/lost paths below. We don't have its gRPC address
      // here (the request's HeartbeatRequest schema doesn't carry
      // address explicitly — only node_id). For W8 the static seed
      // list provides addresses; new peers without prior seed entry
      // can't be reached for outbound heartbeats until the seed list
      // is reconfigured. Surface that limitation in logs.
      std::cerr << "[lethe] heartbeat from unknown peer " << peer_id
                << " (no seed entry; cannot send return heartbeats)\n";
    }
    const bool was_dead = !it->second.alive;
    it->second.last_seen = now;
    it->second.last_seen_epoch = epoch_.load(std::memory_order_relaxed);
    it->second.suspected = false;
    it->second.alive = true;
    if (was_dead) resurrected = true;

    reply.cluster_epoch = epoch_.load(std::memory_order_relaxed);
    reply.alive_peers.reserve(peers_.size());
    for (const auto& [_, info] : peers_) {
      if (!info.alive) continue;
      reply.alive_peers.push_back(PeerStatus{
          info.node_id, info.last_seen_epoch, info.suspected});
    }
  }
  if (resurrected) {
    // Fire membership change OUTSIDE the lock so Router/Replicator
    // calls don't reenter Membership while we're still holding mu_.
    OnMembershipChange(/*lost_peers=*/{});
  }
  return reply;
}

// ---------------------------------------------------------------------------
// Read-side accessors (unchanged from W3-W4)
// ---------------------------------------------------------------------------

std::vector<std::string> Membership::AlivePeers() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> out;
  out.reserve(peers_.size());
  for (const auto& [_, info] : peers_) {
    if (info.alive) out.push_back(info.node_id);
  }
  return out;
}

std::vector<std::string> Membership::AllPeerAddresses() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> out;
  out.reserve(peers_.size());
  for (const auto& [_, info] : peers_) {
    if (info.address.empty()) continue;
    if (!info.alive) continue;  // W8: only broadcast to alive peers
    out.push_back(info.address);
  }
  return out;
}

void Membership::BumpEpochForTesting() {
  epoch_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace lethe
