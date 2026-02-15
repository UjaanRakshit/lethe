#pragma once
// Cluster membership and failure detection.
//
// We do NOT run a real consensus protocol. Larger systems (Kafka,
// Cassandra, Mooncake) delegate this to ZooKeeper/etcd or use SWIM-style
// gossip; here we keep it simple:
//
//   - Static seed list at startup; peers exchange periodic heartbeats.
//   - Each heartbeat carries last-seen epochs for every known peer; a peer
//     not heard from in T_suspect ticks is marked suspected; in T_dead ticks
//     it's removed and we trigger Router::SetPeers + re-replication.
//
// Sufficient for a 3-node cluster; anything more would be over-engineering.

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
class Metrics;

struct PeerInfo {
  std::string node_id;
  std::string address;            // host:port for gRPC
  // Wall-clock timestamp of the last successful heartbeat. Used by
  // EvaluateSuspicions() to drive the suspect_after / dead_after timers
  // (units: steady_clock duration, not cluster epoch).
  std::chrono::steady_clock::time_point last_seen;
  // Cluster epoch at which we last received a heartbeat from this peer.
  // Distinct from `last_seen` above: that one is wall-clock-ish and
  // drives the suspect/dead detector; this one is the monotone cluster-
  // epoch counter and is what we ship to other nodes (and to the Python
  // client) so they can reason about staleness of our peer view across
  // membership changes. Populated by Membership::OnHeartbeat and
  // surfaced through HeartbeatReply::alive_peers[i].last_seen_epoch
  // (mirrors proto/lethe.proto message PeerStatus field 2).
  std::uint64_t last_seen_epoch = 0;
  bool suspected = false;
  bool alive = true;
  // Startup race guard: at process start other nodes' gRPC servers may not
  // be listening yet, so the first several heartbeats can fail. Without a
  // guard, EvaluateSuspicions would declare every peer dead within
  // dead_after of start. We skip peers whose ever_seen is false — "alive by
  // assumption" until first proven otherwise. First successful contact in
  // either direction (outbound reply OR inbound OnHeartbeat) flips ever_seen
  // true and starts the dead_after clock.
  bool ever_seen = false;
};

// Defaults are deliberately conservative — `dead_after=3000ms` makes false
// positives vanishingly rare under stop-the-world pauses, GC spikes, and
// link blips, at the cost of slow detection.
//
// Recovery budget: the recovery-time goal is 3.5s end-to-end = `dead_after`
// (3000ms) + re-replication (~500ms). The chaos suite may tighten these for
// failover micro-benchmarks, but do NOT tighten these defaults to chase a
// number — for a 3-node cluster the false-positive risk dominates the
// latency win.
struct MembershipConfig {
  std::chrono::milliseconds heartbeat_interval{200};
  std::chrono::milliseconds suspect_after{1000};
  std::chrono::milliseconds dead_after{3000};
};

class Membership {
 public:
  Membership(MembershipConfig cfg,
             std::string local_node_id,
             std::vector<StaticPeer> seed_peers,
             Router* router,
             Replicator* replicator,
             Metrics* metrics = nullptr);  // nullable for tests
  ~Membership();

  void Start();
  void Shutdown();

  // RPC entry point. Returns the local view of the cluster — the same
  // alive_peers + cluster_epoch the gRPC shim ships as HeartbeatResponse
  // (see proto/lethe.proto).
  //
  // The proto's HeartbeatRequest carries known_peers (gossip convergence)
  // and load (admission-control signaling), but this signature accepts only
  // peer_id and peer_epoch for now — a known shape mismatch with
  // proto/lethe.proto:128-149 to be widened later, not a contradiction.
  HeartbeatReply OnHeartbeat(const std::string& peer_id,
                             std::uint64_t peer_epoch);

  // Snapshot of currently alive peers.
  std::vector<std::string> AlivePeers() const;

  // Current cluster epoch (incremented on membership change).
  std::uint64_t Epoch() const noexcept { return epoch_.load(); }

  // For Evictor.BroadcastEvictions — returns peer gRPC addresses.
  std::vector<std::string> AllPeerAddresses() const;

  // Test-only seam: bump the cluster epoch by 1, simulating a membership
  // change. The real driver of epoch advance is OnMembershipChange; this
  // lets unit tests observe per-peer `last_seen_epoch` advancing without
  // standing up the full failure-detector loop. Do NOT call from non-test
  // code paths.
  void BumpEpochForTesting();

 private:
  void HeartbeatLoop();
  // Heartbeat fan-out: ONE std::async per active peer, parallel with
  // per-RPC deadline = heartbeat_interval. Replies that come back
  // before the outer deadline update PeerInfo timestamps and may
  // resurrect a dead peer.
  void SendHeartbeatsToAllPeers();
  // EvaluateSuspicions returns the list of peers that just transitioned
  // from alive→dead this tick. Caller (HeartbeatLoop) passes those to
  // OnMembershipChange so Replicator::TriggerReReplication knows which
  // ring positions to backfill from. Returns an empty vector when no
  // membership change happened.
  std::vector<std::string> EvaluateSuspicions();
  // Takes the lost-peer list so Replicator::TriggerReReplication can scope
  // the work to blocks whose replicas just disappeared. Bumps epoch, calls
  // Router::SetPeers with the surviving alive set, then calls
  // Replicator::TriggerReReplication(lost_peers).
  void OnMembershipChange(const std::vector<std::string>& lost_peers);

  MembershipConfig cfg_;
  std::string local_node_id_;
  // router_/replicator_ let membership changes drive Router::SetPeers and
  // Replicator::TriggerReReplication. The ctor body does a single
  // `(void)router_; (void)replicator_;` to satisfy clang's
  // -Wunused-private-field without paying the cross-compiler quirk of
  // [[maybe_unused]] on a data member (GCC warns -Wattributes "attribute
  // ignored").
  Router* router_;
  Replicator* replicator_;
  Metrics* metrics_;          // not owned; nullable

  mutable std::mutex mu_;
  std::unordered_map<std::string, PeerInfo> peers_;

  std::atomic<std::uint64_t> epoch_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace lethe
