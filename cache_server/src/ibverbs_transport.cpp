// Lethe — IbverbsTransport implementation (compiled only when
// LETHE_ENABLE_RDMA=ON). See lethe/kv_transport.hpp for the contract.
// This file is intentionally a stub; methods are filled in across the
// W5–6 plan. GrpcStreamTransport lives in a separate translation unit
// because it is always built.

#include "lethe/kv_transport.hpp"

namespace lethe {

// TODO(W5–6): implement IbverbsTransport methods declared in
// lethe/kv_transport.hpp. Pinned buffer pool, QP per peer, completion
// polling thread, the usual ibverbs choreography.

}  // namespace lethe
