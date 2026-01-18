// Lethe — membership implementation.
//
// Scope right now: only the slice needed to surface per-peer
// last_seen_epoch through HeartbeatReply. The failure-detector loop,
// suspect/dead transitions, gossip convergence, and re-replication
// triggering are W8 work; see lethe/membership.hpp for the full
// contract and the TODO markers below.

#include "lethe/membership.hpp"

#include <utility>

namespace lethe {

Membership::Membership(MembershipConfig cfg,
                       std::string local_node_id,
                       std::vector<StaticPeer> seed_peers,
                       Router* router,
                       Replicator* replicator)
    : cfg_(cfg),
      local_node_id_(std::move(local_node_id)),
      router_(router),
      replicator_(replicator) {
  // Stash for W8 (membership change → router rebuild + re-
  // replication). W3-W4 never reads these from Membership; this
  // (void) cast tells the compiler "I know, I'll use it later"
  // without the cross-compiler quirks of [[maybe_unused]] on a
  // class member. See membership.hpp for the longer note.
  (void)router_;
  (void)replicator_;
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
}

void Membership::Start() {
  // TODO(W8): launch HeartbeatLoop thread that periodically pings
  // every known peer and drives EvaluateSuspicions.
  running_.store(true);
}

void Membership::Shutdown() {
  running_.store(false);
  if (thread_.joinable()) {
    thread_.join();
  }
}

HeartbeatReply Membership::OnHeartbeat(const std::string& peer_id,
                                       std::uint64_t /*peer_epoch*/) {
  // TODO(W8): use peer_epoch + req->known_peers + req->load (see proto
  // HeartbeatRequest fields 2-4) to drive gossip convergence and
  // admission-control signaling.
  HeartbeatReply reply;
  std::lock_guard<std::mutex> lock(mu_);

  auto now = std::chrono::steady_clock::now();
  auto [it, inserted] = peers_.try_emplace(peer_id);
  if (inserted) {
    it->second.node_id = peer_id;
    // TODO(W8): new peer joining is itself a membership change; the
    // real impl will bump cluster epoch and call OnMembershipChange()
    // here. The unit test uses BumpEpochForTesting() instead so this
    // slice stays minimal.
  }
  it->second.last_seen = now;
  it->second.last_seen_epoch = epoch_.load();
  it->second.suspected = false;
  it->second.alive = true;

  reply.cluster_epoch = epoch_.load();
  reply.alive_peers.reserve(peers_.size());
  for (const auto& [_, info] : peers_) {
    if (!info.alive) continue;
    reply.alive_peers.push_back(PeerStatus{
        info.node_id, info.last_seen_epoch, info.suspected});
  }
  return reply;
}

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
    if (!info.address.empty()) out.push_back(info.address);
  }
  return out;
}

void Membership::BumpEpochForTesting() {
  epoch_.fetch_add(1, std::memory_order_relaxed);
}

// TODO(W8): real implementations.
void Membership::HeartbeatLoop() {}
void Membership::EvaluateSuspicions() {}
void Membership::OnMembershipChange() {}

}  // namespace lethe
