// Lethe — replication implementation (W4).
//
// W1 ships destructor stubs only — enough for LetheCache's
// unique_ptr<Replicator> to be destructible without forcing every
// translation unit that includes cache.hpp to also see the full
// PeerClient definition.

#include "lethe/replication.hpp"

namespace lethe {

// W1 placeholder for the pimpl-shaped peer-channel handle. The real
// PeerClient (gRPC channel + stub + connection state) lands in W4.
class PeerClient {};

Replicator::Replicator(std::string local_node_id,
                       Router* router,
                       TieredStore* store)
    : local_node_id_(std::move(local_node_id)),
      router_(router),
      store_(store) {}

Replicator::~Replicator() = default;

// W4: real bodies for ReplicateOut, FetchFromAny, TriggerReReplication,
// EnsurePeerClient, DropPeerClient.

}  // namespace lethe
