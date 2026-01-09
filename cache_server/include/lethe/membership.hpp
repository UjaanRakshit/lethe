#pragma once
// Lethe — cluster membership and failure detection (W8).
//
// We do NOT run a real consensus protocol for membership. Production
// systems (Kafka, Cassandra, Mooncake itself) delegate this to
// ZooKeeper/etcd or use SWIM-style gossip. For Lethe we keep it simple:
//
//   - Static seed list at startup; peers exchange periodic heartbeats.
//   - Each heartbeat carries last-seen epochs for every known peer; a peer
//     not heard from in T_suspect ticks is marked suspected; in T_dead ticks
//     it's removed and we trigger Router::SetPeers + re-replication.
//
// This is sufficient for a 3-node demo. Anything more would be over-eng.

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

class Router;
class Replicator;

struct PeerInfo {
  std::string node_id;
  std::string address;            // host:port for gRPC
  std::chrono::steady_clock::time_point last_seen;
  bool suspected = false;
  bool alive = true;
};

// Defaults are deliberately conservative — `dead_after=3000ms` means false
// positives are vanishingly rare under stop-the-world pauses, GC spikes,
// and link blips, at the cost of slow detection.
//
// Recovery-budget interaction: the project's recovery-time goal is
// 3.5s end-to-end = `dead_after` (3000ms) + re-replication (~500ms).
// See CLAUDE.md "Architecture spine." The chaos suite (W11) is
// expected to tighten these for failover micro-benchmarks; do NOT tighten
// the production defaults to chase a number — the false-positive risk
// dominates the latency win for a 3-node cluster.
struct MembershipConfig {
  std::chrono::milliseconds heartbeat_interval{200};
  std::chrono::milliseconds suspect_after{1000};
  std::chrono::milliseconds dead_after{3000};
};

class Membership {
 public:
  Membership(MembershipConfig cfg,
             std::string local_node_id,
             std::vector<std::string> seed_peers,
             Router* router,
             Replicator* replicator);
  ~Membership();

  void Start();
  void Shutdown();

  // RPC entry point. Returns the local view of the cluster — the same
  // alive_peers + cluster_epoch that the gRPC shim ships as
  // HeartbeatResponse (see proto/lethe.proto).
  HeartbeatReply OnHeartbeat(const std::string& peer_id,
                             std::uint64_t peer_epoch);

  // Snapshot of currently alive peers.
  std::vector<std::string> AlivePeers() const;

  // Current cluster epoch (incremented on membership change).
  std::uint64_t Epoch() const noexcept { return epoch_.load(); }

  // For Evictor.BroadcastEvictions — returns peer gRPC addresses.
  std::vector<std::string> AllPeerAddresses() const;

 private:
  void HeartbeatLoop();
  void EvaluateSuspicions();
  void OnMembershipChange();   // bumps epoch, rebuilds router, triggers repl

  MembershipConfig cfg_;
  std::string local_node_id_;
  Router* router_;
  Replicator* replicator_;

  mutable std::mutex mu_;
  std::unordered_map<std::string, PeerInfo> peers_;

  std::atomic<std::uint64_t> epoch_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace lethe
