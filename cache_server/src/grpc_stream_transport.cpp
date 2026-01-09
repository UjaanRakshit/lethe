// Lethe — GrpcStreamTransport implementation. Default bulk-transport when
// LETHE_ENABLE_RDMA=OFF (or when ibverbs isn't usable at runtime).
//
// TODO: see lethe/kv_transport.hpp for the contract. This file is intentionally
// a stub for now; methods are filled in across the W1–W11 plan documented
// in the top-level README.

#include "lethe/kv_transport.hpp"

namespace lethe {

// TODO: implement GrpcStreamTransport methods declared in
// lethe/kv_transport.hpp. Wraps a bidi-stream client of StreamBlocks
// (proto/lethe.proto) per peer.

}  // namespace lethe
