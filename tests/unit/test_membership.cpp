// Lethe — Membership unit test.
//
// Covers the slice of Membership::OnHeartbeat that maintains per-peer
// `last_seen_epoch` and surfaces it through HeartbeatReply::alive_peers.
// The full failure-detector behavior (suspect/dead transitions, gossip
// convergence) lands in W8; this test will be extended then.
//
// Build wiring: this file is intentionally self-contained — a single
// main() that exits non-zero on assertion failure — so it can be built
// before the project picks a C++ test framework. When W3+ adds
// add_subdirectory(tests) and a real harness (likely GoogleTest), the
// asserts will be ported to TEST() macros without changing the test
// logic.
//
// Standalone build (assumes membership.cpp is on the link line):
//   cl /std:c++20 /EHsc /I cache_server/include \
//       tests/unit/test_membership.cpp cache_server/src/membership.cpp \
//       /Fe:test_membership.exe

#include "lethe/membership.hpp"
#include "lethe/types.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace {

// Find a named peer in HeartbeatReply::alive_peers and return its
// last_seen_epoch; std::nullopt if the peer is absent.
std::optional<std::uint64_t> FindLastSeenEpoch(
    const lethe::HeartbeatReply& reply, const std::string& node_id) {
  for (const auto& peer : reply.alive_peers) {
    if (peer.node_id == node_id) return peer.last_seen_epoch;
  }
  return std::nullopt;
}

void TestLastSeenEpochAdvancesAcrossHeartbeats() {
  lethe::MembershipConfig cfg;
  lethe::Membership m(cfg,
                      /*local_node_id=*/"node_self",
                      /*seed_peers=*/{{"nodeA", "127.0.0.1:0"}},
                      /*router=*/nullptr,
                      /*replicator=*/nullptr);

  // First heartbeat from A at cluster epoch 0.
  auto reply1 = m.OnHeartbeat("nodeA", /*peer_epoch=*/0);
  auto e1 = FindLastSeenEpoch(reply1, "nodeA");
  assert(e1.has_value() && "nodeA must appear in alive_peers");
  assert(*e1 == 0 && "first heartbeat at epoch 0 → last_seen_epoch=0");

  // Simulate a membership change between the two heartbeats; in
  // production, OnMembershipChange does this when a peer is declared
  // dead. The unit test uses the explicit seam so it does not have to
  // stand up the full detector loop.
  m.BumpEpochForTesting();
  m.BumpEpochForTesting();
  assert(m.Epoch() == 2);

  // Second heartbeat from A. cluster_epoch is now 2, so A's
  // last_seen_epoch in the reply must have advanced.
  auto reply2 = m.OnHeartbeat("nodeA", /*peer_epoch=*/0);
  auto e2 = FindLastSeenEpoch(reply2, "nodeA");
  assert(e2.has_value());
  assert(*e2 > *e1 &&
         "two heartbeats from the same peer at different cluster epochs "
         "must produce monotonically increasing last_seen_epoch");
  assert(*e2 == 2 && "second heartbeat at epoch 2 → last_seen_epoch=2");

  // HeartbeatReply::cluster_epoch surfaces the current cluster epoch.
  assert(reply2.cluster_epoch == 2);
}

void TestHeartbeatReplyMirrorsPeerStatus() {
  // Sanity: every PeerStatus in the reply has the suspected bit set
  // to its current value (false after a successful heartbeat).
  lethe::MembershipConfig cfg;
  lethe::Membership m(cfg, "node_self", {"nodeA"}, nullptr, nullptr);

  auto reply = m.OnHeartbeat("nodeA", 0);
  bool found = false;
  for (const auto& p : reply.alive_peers) {
    if (p.node_id == "nodeA") {
      assert(p.suspected == false &&
             "fresh heartbeat from nodeA must clear the suspected bit");
      found = true;
    }
  }
  assert(found);
}

}  // namespace

int main() {
  TestLastSeenEpochAdvancesAcrossHeartbeats();
  TestHeartbeatReplyMirrorsPeerStatus();
  std::printf("test_membership: OK\n");
  return 0;
}
